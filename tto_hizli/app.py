# -*- coding: utf-8 -*-,
"""Trento Toplu Okuma masaüstü uygulaması — HIZLANDIRILMIŞ TEST SÜRÜMÜ.

TTO/app.py'nin kopyasıdır. Fark: sayım ekranındaki HIZLANDIRMA kutuları.
Hepsi kapalıyken bugünkü TTO yolu (process_image) birebir çalışır; biri
açıkken barkod/kayıt adımları hizli_islemci.py ile yapılır ve palet süresi
adım adım ölçülüp ekrana ve captures/sure_kayitlari.jsonl'e yazılır.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from tkinter import filedialog
import tkinter as tk

import cv2
import customtkinter as ctk
from PIL import Image, ImageDraw, ImageTk

BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parent
CAMERA_GUI_DIR = REPO_DIR / "Kamera_gui"

# TTO'nun kullanacağı YOLO modeli. Dosya yoksa ya da satır silinirse
# camera_gui_v2'nin varsayılan modeli kullanılır; ortamdan gelen
# ODAI_YOLO_MODEL değeri her zaman önceliklidir.
_TTO_YOLO_MODEL = REPO_DIR / "V8LAST.pt"
if not _TTO_YOLO_MODEL.is_file():   # i7-14 paketinde model models/ altında
    _TTO_YOLO_MODEL = REPO_DIR / "models" / "V8LAST.pt"
if _TTO_YOLO_MODEL.is_file():
    os.environ.setdefault("ODAI_YOLO_MODEL", str(_TTO_YOLO_MODEL))

# Arayüzdeki model seçicinin taradığı klasör: buradaki her .pt listeye girer.
MODELS_DIR = Path(r"C:\Users\Administrator\Desktop\Models")
# Kamera adları ve kart sırası: SERİ NUMARASINA yazılır. SDK'nın tarama
# sırası ağdan cevap gelme sırasıdır ve açılıştan açılışa değişebilir; bu
# dosya sayesinde aynı fiziksel kamera her zaman aynı karta ve aynı isme
# oturur. Kart başlığına çift dokunarak isim verilir.
CAMERA_NAMES_PATH = Path(__file__).resolve().parent / "kamera_isimleri.json"
DEFAULT_MODEL_LABEL = "Varsayılan (YOLOV11 FULLAUG)"
CAPTURE_DIR = BASE_DIR / "captures"
CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

if str(REPO_DIR) not in sys.path:   # aremak_barkod / app_ui / ocr_engine (i7-14 paketinde kökte)
    sys.path.append(str(REPO_DIR))
if str(CAMERA_GUI_DIR) not in sys.path:
    sys.path.insert(0, str(CAMERA_GUI_DIR))
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from camera_gui_v2 import (  # noqa: E402
    DetectionProcessor,
    pil_trento_logo_to_rgb,
    resolve_trento_logo_path,
)
from multi_camera_gui_2 import CrateAggregator, MultiCameraManager  # noqa: E402

# Ücretli barkod motoru: Aremak Code Reader (.NET DLL + USB dongle).
# Paket/DLL/dongle yoksa zxing-cpp'li DetectionProcessor'a düşülür.
AREMAK_IMPORT_ERROR: str | None = None
try:
    from multi_camera_gui_20260511 import (  # noqa: E402
        AREMAK_AVAILABLE,
        AremakDetectionProcessor,
    )
    if not AREMAK_AVAILABLE:
        AREMAK_IMPORT_ERROR = (
            "aremak_barkod paketi yüklenemedi (barkod_aremak/aremak_barkod "
            "klasörü ya da pythonnet kurulumu eksik olabilir)"
        )
except Exception as _aremak_ex:  # pragma: no cover
    AREMAK_AVAILABLE = False
    AremakDetectionProcessor = None  # type: ignore[assignment]
    AREMAK_IMPORT_ERROR = f"{type(_aremak_ex).__name__}: {_aremak_ex}"
    print(f"[UYARI] Aremak işlemcisi import edilemedi: {_aremak_ex}")

import theme  # noqa: E402
from widgets import CameraTile, InfoCard, MetricCard, ModeCard  # noqa: E402
import ocr_reader  # noqa: E402  (PaddleOCR modeli ilk kullanımda yüklenir)
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
import numpy as np  # noqa: E402
from hizli_islemci import (  # noqa: E402
    AYAR_ADLARI, AYAR_ETIKET, BarkodAkisi, HizliAyarlar, HizliBarkod,
)

HIZLI_AYAR_PATH = BASE_DIR / "hizli_ayarlar.json"
SURE_KAYIT_PATH = CAPTURE_DIR / "sure_kayitlari.jsonl"


MAX_CAMERAS = 6
PREVIEW_FPS = 3
# Bağlantıda tüm kameralara uygulanan sabit pozlama (µs). Otomatik pozlama
# sahne aydınlığına göre değişip çekimler arası tutarsızlık yaratıyor.
DEFAULT_EXPOSURE_US = 200000
# Aynı anda kare AKTARAN kamera sayısı üst sınırı. Kamera başına bant sınırı
# ~264 Mbps; PC'ye giden switch hattı 1 Gbit. 6 kamera birden tetiklenirse
# 6 × 264 ≈ 1.6 Gbps hatta sığmaz → switch GVSP (UDP) paketlerini düşürür ve
# kareler yırtık gelir. 2 kamera ≈ 530 Mbps → bol pay. (3 = ~800 Mbps, sınırda.)
MAX_CONCURRENT_CAPTURES = 2


def make_fullscreen(window) -> None:
    """Pencereyi ekranı kaplayacak biçimde açar.

    Tk'de `state("zoomed")` pencere HENÜZ HARİTALANMADAN çağrılırsa Windows
    bunu sessizce yok sayabiliyor — özellikle ürün seçimine dönüşte açılan
    İKİNCİ pencerede. Bu yüzden üç kez denenir: hemen, pencere ekrana
    geldiğinde (<Map>, tek seferlik) ve kısa bir gecikmeyle. Böylece uygulama
    her açılışta panelde tam ekran gelir.
    """
    def apply(_event=None):
        try:
            window.deiconify()
            window.state("zoomed")
        except tk.TclError:
            try:
                width = window.winfo_screenwidth()
                height = window.winfo_screenheight()
                window.geometry(f"{width}x{height}+0+0")
            except tk.TclError:
                return
        try:
            window.lift()
        except tk.TclError:
            pass

    # Tek seferlik: operatör pencereyi sonradan küçültürse zorlamayalım.
    # (unbind yerine bayrak — Tk sürümüne göre unbind başka bağlamaları da
    # silebiliyor; burada CTk'nın kendi <Map> bağlamasına dokunmuyoruz.)
    applied = {"done": False}

    def on_map(event=None):
        if applied["done"]:
            return
        applied["done"] = True
        apply(event)

    window.update_idletasks()
    apply()
    window.bind("<Map>", on_map, add="+")
    try:
        window.after(60, apply)
    except tk.TclError:
        pass


class ProductLauncher:
    """43 / 64 / karışık ürün seçim ekranı."""

    def __init__(self):
        self.selection = None
        self.root = ctk.CTk()
        self.root.title("Trento Toplu Okuma")
        self.root.minsize(1000, 700)
        self.root.configure(fg_color=theme.BG)
        self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)
        self._logo = None
        self._build()
        make_fullscreen(self.root)

    def _build_brand(self, parent):
        brand = ctk.CTkFrame(parent, fg_color="transparent")
        brand.pack(pady=(0, 8))
        logo_path = resolve_trento_logo_path()
        if logo_path:
            try:
                image = pil_trento_logo_to_rgb(Image.open(logo_path), theme.BG)
                max_h = 56
                new_w = max(1, int(image.width * max_h / image.height))
                image = image.resize((new_w, max_h), Image.Resampling.LANCZOS)
                self._logo = ctk.CTkImage(image, image, size=(new_w, max_h))
                ctk.CTkLabel(brand, image=self._logo, text="").pack()
            except OSError:
                self._logo = None
        ctk.CTkLabel(
            brand,
            text="TRENTO",
            text_color=theme.GREEN,
            font=ctk.CTkFont(theme.FONT, 12, "bold"),
        ).pack(pady=(8, 0))
        ctk.CTkLabel(
            brand,
            text="TOPLU OKUMA",
            text_color=theme.TEXT,
            font=ctk.CTkFont(theme.FONT, 38, "bold"),
        ).pack()
        ctk.CTkLabel(
            brand,
            text="Akıllı Kasa Sayım Sistemi",
            text_color=theme.TEXT_MUTED,
            font=ctk.CTkFont(theme.FONT, 13),
        ).pack(pady=(2, 0))

    def _build(self):
        ctk.CTkFrame(self.root, height=4, corner_radius=0, fg_color=theme.GREEN).pack(fill="x")
        page = ctk.CTkFrame(self.root, fg_color="transparent")
        page.pack(fill="both", expand=True, padx=54, pady=30)

        self._build_brand(page)
        ctk.CTkLabel(
            page,
            text="OKUMA TİPİNİ SEÇİN",
            text_color=theme.GREEN,
            font=ctk.CTkFont(theme.FONT, 10, "bold"),
        ).pack(pady=(15, 3))
        ctk.CTkLabel(
            page,
            text="Sayım yapılacak palet düzenini seçerek devam edin",
            text_color=theme.TEXT_SOFT,
            font=ctk.CTkFont(theme.FONT, 14),
        ).pack(pady=(0, 22))

        cards = ctk.CTkFrame(page, fg_color="transparent")
        cards.pack()
        ModeCard(cards, "43", "43 kasalık palet düzeni", False).grid(row=0, column=0, padx=12)
        ModeCard(
            cards,
            "64",
            "6 kamera ile toplu okuma",
            True,
            command=self._open_64,
        ).grid(row=0, column=1, padx=12)

        ctk.CTkButton(
            page,
            text="◈   KARIŞIK PALET   ·   YAKINDA",
            state="disabled",
            width=664,
            height=54,
            corner_radius=13,
            fg_color=theme.BG_RAISED,
            text_color=theme.TEXT_MUTED,
            border_width=1,
            border_color=theme.BORDER,
            font=ctk.CTkFont(theme.FONT, 12, "bold"),
        ).pack(pady=(20, 0))

        footer = ctk.CTkFrame(
            page,
            height=44,
            corner_radius=12,
            fg_color=theme.BG_RAISED,
            border_width=1,
            border_color=theme.BORDER,
        )
        footer.pack(fill="x", side="bottom")
        footer.pack_propagate(False)
        ctk.CTkLabel(
            footer,
            text="KÜTAHYA ÜRETİM TESİSİ",
            text_color=theme.TEXT_MUTED,
            font=ctk.CTkFont(theme.FONT, 9, "bold"),
        ).pack(side="left", padx=15)
        ctk.CTkLabel(
            footer,
            text="TTO  ·  Trento Toplu Okuma",
            text_color=theme.TEXT_MUTED,
            font=ctk.CTkFont(theme.FONT, 9),
        ).pack(side="right", padx=15)

    def _open_64(self):
        self.selection = "64"
        self.root.destroy()

    def run(self):
        self.root.mainloop()
        return self.selection


class TTOApplication:
    """64 kasa için tamamen yeniden tasarlanmış kamera ve sayım ekranı."""

    def __init__(self):
        self.manager = MultiCameraManager()
        if AREMAK_AVAILABLE and AremakDetectionProcessor is not None:
            self.detector = AremakDetectionProcessor()
            self.barcode_engine = "Aremak Code Reader (dongle)"
        else:
            self.detector = DetectionProcessor()
            self.barcode_engine = "zxing-cpp (yedek)"
        self.aggregator = CrateAggregator()
        # HIZLANDIRMA (test): aç/kapa kutuları, okuyucu havuzu, kayıt havuzu.
        self.hizli = HizliAyarlar(HIZLI_AYAR_PATH)
        self.hizli_barkod = (
            HizliBarkod(self.detector, self.log)
            if AREMAK_AVAILABLE and AremakDetectionProcessor is not None else None
        )
        self._io_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="io")
        self._hizli_vars = {}
        self._last_timing = None
        # Süre ölçümü: kamera iş parçacığı başına kayıt (YOLO/barkod sarmalayıcıları
        # buraya ekler), sayım başlangıcı, detay penceresi.
        self._tl = threading.local()
        self._cam_timing = {}
        self._t_run = None
        self._timing_window = None
        self._timing_text = None
        self._ocr_t0 = None
        # Barkod motoru hazır değilse SAYIM YAPILMAZ. Sebep: tekilleştirme
        # (CrateAggregator) kamera çiftleri arasındaki kaymayı ORTAK
        # BARKODLARDAN öğrenir. Barkod okunmazsa hiçbir kasa eşleşmez,
        # kameralar aynı kasayı ayrı ayrı sayar ve toplam gerçekte olduğundan
        # yüksek çıkar. Sessizce yanlış sayı vermek yerine hata gösterilir.
        self.barcode_ready = False
        self.barcode_error = "Barkod motoru henüz sınanmadı"
        self._barcode_probing = False
        self.cameras = []
        self.tiles: list[CameraTile] = []
        self.results = [None] * MAX_CAMERAS
        self.unread_records = []
        self.camera_images = []
        self.processing_lock = threading.Lock()
        # 1 Gbit hattı taşırmamak için eşzamanlı kare aktarımını sınırlar;
        # sıradaki kamera ancak bir öncekinin aktarımı bitince tetiklenir.
        self._capture_slots = threading.Semaphore(MAX_CONCURRENT_CAPTURES)
        self.preview_running = False
        self.preview_thread = None
        self.return_to_launcher = False
        self.camera_filter = 6
        self.camera_page_start = 0
        self.portrait_mode = False
        self.active_page = "session"
        # Sevkiyat süreci: bir tır = bir süreç. İrsaliye/plaka ile açılır,
        # tırdaki paletler tek tek okunup onaylandıkça sürece eklenir ve
        # operatör bitirene kadar açık kalır. None = süreç yok (deneme modu:
        # kameralar kontrol edilebilir, sayım yapılabilir ama kaydedilmez).
        self.session = None
        self._no_session_warned = False
        # Palet inceleme görünümü
        self.pallet_serial_options = ("6412", "6417", "6420", "6422")
        self.pallet_series_colors = (
            theme.GREEN, theme.CYAN, theme.BLUE, theme.AMBER, "#8B5CF6", "#C2557F",
        )
        self.pallet_crates = []
        self.last_aggregate = None
        self._pallet_cards = []
        self._pallet_photos = []
        self._pallet_width = 0
        self._modal_photo = None
        self.blink_on = True
        self._blink_job = None
        # Her yeni çekim/sayım bunu artırır; eski OCR iş parçacıklarının
        # gecikmiş sonuçları güncel sayıma karışmasın diye kontrol edilir.
        self._ocr_generation = 0
        self._ocr_warned = False  # "OCR devre dışı" uyarısı bir kez gösterilir
        self._discarded_count = 0  # "burada kasa yok" ile silinen yanlış tespit
        self._counting = False  # sayım sürüyor (doğrulama şeridi için)
        self._capture_busy = False
        self._capture_enabled = False
        # Sayım iptali: kamera iş parçacıkları bu bayrağı kontrol eder.
        self._capture_cancel = threading.Event()
        # Onaylanan palet, sürecin BU indeksindeki paletin üstüne yazılır
        # (yeniden say / revize). None ise yeni palet olarak eklenir.
        self._replace_target = None
        self._ocr_progress = None  # (işlenen, toplam) — OCR sürerken şeritte
        self._tile_resize_job = None
        self._tile_fitting = False
        self._toast_label = None
        self._toast_job = None
        self._modal_blocker = None
        self._modal_card = None
        self._viewer_photo = None
        self._viewer_path = None
        self._logo = None

        self.root = ctk.CTk()
        self.root.title("TTO · Trento Toplu Okuma · 64 Kasa · HIZLANDIRILMIŞ (test)")
        self._set_minsize(1380, 820)
        self.root.configure(fg_color=theme.BG)
        self.root.protocol("WM_DELETE_WINDOW", self._close)

        self._build_ui()
        make_fullscreen(self.root)
        if self.root.winfo_screenheight() > self.root.winfo_screenwidth():
            self._toggle_portrait_mode()
        self._blink_job = self.root.after(600, self._blink_tick)
        # OCR modelini arka planda şimdiden yükle + ısıt: ilk sayımda
        # okunamayan kasalar saniyeler içinde etiketlenir.
        threading.Thread(target=self._preload_ocr, daemon=True).start()
        self.log(f"Barkod motoru: {self.barcode_engine}")
        self.log("TTO 64 Kasa modu hazır.")
        # Dongle/SDK durumunu açılışta sına — operatör sayıma başlamadan görsün.
        self._start_barcode_probe(initial=True)

    def _set_minsize(self, width, height):
        """En küçük pencere boyutunu ekranı aşmayacak şekilde ayarlar.

        1200px genişlikteki dik panelde yatay moda geçilirse minsize(1380)
        pencereyi ekran dışına taşırıp sağdaki düğmeleri görünmez yapıyordu.
        """
        try:
            screen_w = self.root.winfo_screenwidth()
            screen_h = self.root.winfo_screenheight()
        except tk.TclError:
            screen_w, screen_h = width, height
        self.root.minsize(min(width, screen_w), min(height, screen_h))

    def _preload_ocr(self):
        if ocr_reader.availability_error():
            return
        ocr = ocr_reader.get_ocr(log_fn=self.log)
        if ocr is not None:
            self.log(f"OCR hazır bekliyor (cihaz={ocr_reader.active_device()}).")

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        self.sidebar = ctk.CTkFrame(
            self.root,
            width=220,
            corner_radius=0,
            fg_color=theme.BG_RAISED,
            border_width=0,
        )
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)
        self._build_sidebar()

        self.main = ctk.CTkFrame(self.root, fg_color=theme.BG, corner_radius=0)
        self.main.pack(side="left", fill="both", expand=True)

        self._build_toolbar()
        self._build_barcode_banner()
        self.page_container = ctk.CTkFrame(self.main, fg_color=theme.BG)
        self.page_container.pack(fill="both", expand=True, padx=18, pady=(0, 16))

        self.session_page = ctk.CTkFrame(self.page_container, fg_color=theme.BG)
        self.count_page = ctk.CTkFrame(self.page_container, fg_color=theme.BG)
        self.unread_page = ctk.CTkFrame(self.page_container, fg_color=theme.BG)
        self.ocr_page = ctk.CTkFrame(self.page_container, fg_color=theme.BG)
        self.pallet_page = ctk.CTkFrame(self.page_container, fg_color=theme.BG)
        self.summary_page = ctk.CTkFrame(self.page_container, fg_color=theme.BG)
        self._build_session_page()
        self._build_count_page()
        self._build_unread_page()
        self._build_ocr_page()
        self._build_pallet_page()
        self._build_summary_page()
        # Uygulama sevkiyat ekranıyla açılır: önce süreç, sonra sayım.
        self._show_page("session")

    def _build_sidebar(self):
        brand = ctk.CTkFrame(self.sidebar, fg_color="transparent", height=112)
        brand.pack(fill="x", padx=18, pady=(22, 16))
        brand.pack_propagate(False)

        logo_path = resolve_trento_logo_path()
        if logo_path:
            try:
                image = pil_trento_logo_to_rgb(Image.open(logo_path), theme.BG_RAISED)
                max_h = 34
                new_w = max(1, int(image.width * max_h / image.height))
                image = image.resize((new_w, max_h), Image.Resampling.LANCZOS)
                self._logo = ctk.CTkImage(image, image, size=(new_w, max_h))
                ctk.CTkLabel(brand, image=self._logo, text="").pack(anchor="w")
            except OSError:
                self._logo = None
        ctk.CTkLabel(
            brand,
            text="TRENTO",
            anchor="w",
            text_color=theme.GREEN,
            font=ctk.CTkFont(theme.FONT, 9, "bold"),
        ).pack(anchor="w", pady=(8, 0))
        ctk.CTkLabel(
            brand,
            text="TOPLU OKUMA",
            anchor="w",
            text_color=theme.TEXT,
            font=ctk.CTkFont(theme.FONT, 19, "bold"),
        ).pack(anchor="w")

        ctk.CTkLabel(
            self.sidebar,
            text="MENÜ",
            anchor="w",
            text_color=theme.TEXT_MUTED,
            font=ctk.CTkFont(theme.FONT, 9, "bold"),
        ).pack(fill="x", padx=20, pady=(12, 7))

        self.nav_session = self._nav_button(
            "▤", "Sevkiyat", lambda: self._show_page("session")
        )
        self.nav_count = self._nav_button("▦", "Sayım Ekranı", lambda: self._show_page("count"))
        self.nav_pallet = self._nav_button("▤", "Sayım Doğrulama", lambda: self._show_page("pallet"))
        self.nav_unread = self._nav_button("!", "Okunamayanlar", lambda: self._show_page("unread"))
        self.nav_ocr = self._nav_button("⚡", "Sistem Tespitleri", lambda: self._show_page("ocr"))

        ctk.CTkFrame(self.sidebar, height=1, fg_color=theme.BORDER).pack(fill="x", padx=18, pady=16)
        self._nav_button("←", "Ürün Seçimine Dön", self._back_to_launcher, subtle=True)

        status_wrap = ctk.CTkFrame(
            self.sidebar,
            fg_color=theme.SURFACE,
            corner_radius=13,
            border_width=1,
            border_color=theme.BORDER,
        )
        status_wrap.pack(side="bottom", fill="x", padx=14, pady=14)
        ctk.CTkLabel(
            status_wrap,
            text="64",
            width=46,
            height=46,
            corner_radius=11,
            fg_color=theme.GREEN_DARK,
            text_color=theme.GREEN,
            font=ctk.CTkFont(theme.FONT, 21, "bold"),
        ).pack(side="left", padx=10, pady=10)
        mode_text = ctk.CTkFrame(status_wrap, fg_color="transparent")
        mode_text.pack(side="left")
        ctk.CTkLabel(
            mode_text, text="AKTİF MOD", anchor="w", text_color=theme.TEXT_MUTED,
            font=ctk.CTkFont(theme.FONT, 8, "bold"),
        ).pack(anchor="w")
        ctk.CTkLabel(
            mode_text, text="64 Kasa", anchor="w", text_color=theme.TEXT,
            font=ctk.CTkFont(theme.FONT, 11, "bold"),
        ).pack(anchor="w")
        self.sidebar_session = ctk.CTkLabel(
            mode_text, text="", anchor="w", justify="left",
            text_color=theme.TEXT_MUTED,
            font=ctk.CTkFont(theme.MONO, 8),
        )
        self.sidebar_session.pack(anchor="w")

    def _nav_button(self, icon, text, command, subtle=False):
        button = ctk.CTkButton(
            self.sidebar,
            text=f"{icon}   {text}",
            anchor="w",
            height=42,
            corner_radius=10,
            fg_color="transparent",
            hover_color=theme.SURFACE,
            text_color=theme.TEXT_MUTED if subtle else theme.TEXT_SOFT,
            font=ctk.CTkFont(theme.FONT, 11, "bold"),
            command=command,
        )
        button.pack(fill="x", padx=12, pady=3)
        return button

    def _build_toolbar(self):
        self.toolbar = ctk.CTkFrame(
            self.main, height=84, fg_color=theme.BG, corner_radius=0
        )
        self.toolbar.pack(fill="x", padx=18, pady=(8, 0))
        self.toolbar.pack_propagate(False)

        self.toolbar_title = ctk.CTkFrame(self.toolbar, fg_color="transparent")
        self.toolbar_title.pack(side="left", pady=12)
        self.page_eyebrow = ctk.CTkLabel(
            self.toolbar_title,
            text="64 KASA MODU",
            anchor="w",
            text_color=theme.GREEN,
            font=ctk.CTkFont(theme.FONT, 9, "bold"),
        )
        self.page_eyebrow.pack(anchor="w")
        self.page_title = ctk.CTkLabel(
            self.toolbar_title,
            text="Kamera Kontrol Merkezi",
            anchor="w",
            text_color=theme.TEXT,
            font=ctk.CTkFont(theme.FONT, 24, "bold"),
        )
        self.page_title.pack(anchor="w")

        self.toolbar_actions = ctk.CTkFrame(
            self.toolbar, fg_color="transparent"
        )
        self.toolbar_actions.pack(side="right", pady=15)

        self.status_label = ctk.CTkLabel(
            self.toolbar_actions,
            text="●  Beklemede",
            width=130,
            height=36,
            corner_radius=10,
            fg_color=theme.SURFACE,
            text_color=theme.TEXT_MUTED,
            font=ctk.CTkFont(theme.FONT, 10, "bold"),
        )
        self.status_label.pack(side="left", padx=4)
        actions_parent = self.toolbar_actions

        self.preview_switch = ctk.CTkSwitch(
            self.toolbar_actions,
            text="Canlı Önizleme",
            width=120,
            progress_color=theme.GREEN,
            text_color=theme.TEXT_SOFT,
            font=ctk.CTkFont(theme.FONT, 10),
            command=self._toggle_preview,
        )
        self.preview_switch.pack(side="left", padx=10)

        self.portrait_btn = self._toolbar_button(
            "▯ Dik Ekran",
            self._toggle_portrait_mode,
            theme.SURFACE_LIGHT,
        )
        self.connect_btn = self._toolbar_button("⌁ Kameraları Tara", self._on_connect, theme.SURFACE_LIGHT)
        # Sayım düğmeleri: yatay için küçük, dik için tam genişlik takım.
        self.capture_group_h = ctk.CTkFrame(actions_parent, fg_color="transparent")
        self.capture_group_h.pack(side="left", padx=4)
        self.capture_buttons_h = self._build_capture_buttons(
            self.capture_group_h, big=False
        )

        # Dik ekranda kenar çubuğu gizlenir; gezinme araç çubuğuna taşınır.
        # KENDİ satırında ve tam genişlikte durur — eskiden diğer düğmelerle
        # aynı satıra sıkışıp "Kameraları Tara"/"Yatay Ekran" ekran dışına
        # taşıyordu.
        self.portrait_nav = ctk.CTkFrame(self.toolbar, fg_color="transparent")
        self.portrait_nav_buttons = {}
        for column, (text, page) in enumerate(
            (
                ("▤  Sevkiyat", "session"),
                ("▦  Sayım", "count"),
                ("▣  Doğrulama", "pallet"),
                ("!  Okunamayan", "unread"),
                ("⚡  Tespit", "ocr"),
            )
        ):
            button = ctk.CTkButton(
                self.portrait_nav,
                text=text,
                height=46,
                corner_radius=11,
                fg_color=theme.SURFACE_LIGHT,
                hover_color=theme.BORDER_ACTIVE,
                text_color=theme.TEXT,
                font=ctk.CTkFont(theme.FONT, 12, "bold"),
                command=lambda target=page: self._show_page(target),
            )
            button.grid(row=0, column=column, sticky="ew", padx=3)
            self.portrait_nav.grid_columnconfigure(
                column, weight=1, uniform="portrait_nav"
            )
            self.portrait_nav_buttons[page] = button
        self.portrait_session_btn = self.portrait_nav_buttons["session"]
        self.portrait_count_btn = self.portrait_nav_buttons["count"]
        self.portrait_pallet_btn = self.portrait_nav_buttons["pallet"]
        self.portrait_unread_btn = self.portrait_nav_buttons["unread"]
        self.portrait_ocr_btn = self.portrait_nav_buttons["ocr"]

        # Dik ekranda kenar çubuğu olmadığı için ürün seçimine dönüş yolu
        # kalmıyordu; bu düğme yalnız dik modda görünür.
        self.portrait_home_btn = self._toolbar_button(
            "←  Ürün Seçimi", self._back_to_launcher, theme.SURFACE_LIGHT
        )
        self.portrait_home_btn.pack_forget()

        # Dik ekranda araç çubuğuna sığmayan sayım düğmeleri için tam genişlik
        # takım — yalnızca dik modda görünür.
        self.capture_all_big = ctk.CTkFrame(self.toolbar, fg_color="transparent")
        self.capture_buttons_v = self._build_capture_buttons(
            self.capture_all_big, big=True
        )
        self._refresh_capture_buttons()

    def _set_capture_all_state(self, state, busy=False):
        self._capture_busy = bool(busy)
        self._capture_enabled = state == "normal"
        if not busy:
            for buttons in (self.capture_buttons_h, self.capture_buttons_v):
                buttons["cancel"].configure(state="normal", text=(
                    "⏹   SAYIMI İPTAL ET"
                    if buttons is self.capture_buttons_v
                    else "⏹ İPTAL"
                ))
        self._refresh_capture_buttons()

    # Sayım düğmesi takımı. Aynı takım hem yatayda (küçük) hem dikte (büyük)
    # kurulur; hangisinin görüneceğine _refresh_capture_buttons karar verir.
    #   yeni    : yeni palet olarak ekle
    #   again   : son eklenen paletin ÜSTÜNE yaz (yanlış sayımı düzeltmek için)
    #   revise  : sevkiyat listesinden seçilen paletin üstüne yaz
    #   cancel  : sürmekte olan sayımı iptal et
    def _build_capture_buttons(self, parent, big):
        height = 62 if big else 38
        radius = 14 if big else 10
        size = 17 if big else 10
        buttons = {}
        buttons["new"] = ctk.CTkButton(
            parent,
            text="▶   YENİ PALET OKU" if big else "◎ YENİ PALET",
            command=lambda: self._on_capture_all(replace_last=False),
            height=height, corner_radius=radius,
            fg_color=theme.GREEN, hover_color=theme.GREEN_HOVER,
            text_color="#FFFFFF", font=ctk.CTkFont(theme.FONT, size, "bold"),
        )
        buttons["again"] = ctk.CTkButton(
            parent,
            text="↻   PALETİ YENİDEN SAY" if big else "↻ YENİDEN SAY",
            command=lambda: self._on_capture_all(replace_last=True),
            height=height, corner_radius=radius,
            fg_color=theme.AMBER, hover_color="#A8730F",
            text_color="#FFFFFF", font=ctk.CTkFont(theme.FONT, size, "bold"),
        )
        buttons["revise"] = ctk.CTkButton(
            parent,
            text="✎   REVİZE ET",
            command=lambda: self._on_capture_all(replace_last=False),
            height=height, corner_radius=radius,
            fg_color=theme.AMBER, hover_color="#A8730F",
            text_color="#FFFFFF", font=ctk.CTkFont(theme.FONT, size, "bold"),
        )
        buttons["revise_cancel"] = ctk.CTkButton(
            parent,
            text="✕ Vazgeç",
            command=self._cancel_revise,
            width=110 if big else 92,
            height=height, corner_radius=radius,
            fg_color=theme.SURFACE_LIGHT, hover_color=theme.BORDER_ACTIVE,
            text_color=theme.TEXT, font=ctk.CTkFont(theme.FONT, size, "bold"),
        )
        buttons["cancel"] = ctk.CTkButton(
            parent,
            text="⏹   SAYIMI İPTAL ET" if big else "⏹ İPTAL",
            command=self._cancel_capture,
            height=height, corner_radius=radius,
            fg_color=theme.RED, hover_color="#B23A3A",
            text_color="#FFFFFF", font=ctk.CTkFont(theme.FONT, size, "bold"),
        )
        return buttons

    def _refresh_capture_buttons(self):
        """Duruma göre hangi sayım düğmelerinin görüneceğini belirler."""
        groups = (
            (self.capture_buttons_h, self.portrait_mode is False),
            (self.capture_buttons_v, self.portrait_mode is True),
        )
        session = self.session
        revising = self._replace_target is not None
        can_again = bool(
            session and session["pallets"] and session["finished_at"] is None
        )
        for buttons, active in groups:
            for button in buttons.values():
                button.pack_forget()
            if not active:
                continue
            if self._capture_busy:
                order = ["cancel"]
            elif revising:
                order = ["revise", "revise_cancel"]
            else:
                order = ["new"] + (["again"] if can_again else [])
            for key in order:
                button = buttons[key]
                if buttons is self.capture_buttons_v:
                    button.pack(side="left", fill="x", expand=True, padx=3)
                else:
                    button.pack(side="left", padx=3)
                if key in ("new", "again", "revise"):
                    button.configure(
                        state="normal" if self._capture_enabled else "disabled"
                    )
        if revising and not self._capture_busy:
            hedef = self._replace_target
            etiket = "?"
            if session and 0 <= hedef < len(session["pallets"]):
                etiket = str(session["pallets"][hedef]["no"])
            for buttons in (self.capture_buttons_h, self.capture_buttons_v):
                buttons["revise"].configure(text=f"✎   PALET {etiket} REVİZE ET")

    def _cancel_revise(self):
        self._replace_target = None
        self._refresh_capture_buttons()
        self._show_toast("Revize iptal edildi", theme.CYAN)

    def _cancel_capture(self):
        """Sürmekte olan sayımı iptal eder (yanlışlıkla başlatıldıysa)."""
        if not self._capture_busy:
            return
        self._capture_cancel.set()
        self.set_status("Sayım iptal ediliyor…", theme.AMBER)
        self.log("Sayım iptal edildi (operatör).")
        for buttons in (self.capture_buttons_h, self.capture_buttons_v):
            buttons["cancel"].configure(state="disabled", text="⏹ İPTAL EDİLİYOR…")

    def _toolbar_button(self, text, command, color, dark_text=False):
        actions = self.status_label.master
        button = ctk.CTkButton(
            actions,
            text=text,
            command=command,
            height=38,
            corner_radius=10,
            fg_color=color,
            hover_color=theme.GREEN_HOVER if color == theme.GREEN else theme.BORDER_ACTIVE,
            text_color=theme.BG if dark_text else theme.TEXT,
            font=ctk.CTkFont(theme.FONT, 10, "bold"),
        )
        button.pack(side="left", padx=4)
        return button

    # --------------------------------------------------------- barkod motoru
    def _build_barcode_banner(self):
        """Barkod SDK/dongle hatasında her sayfanın üstünde duran kırmızı şerit.

        Donanım hatası olduğu için tek bir sayfaya gizlenmez; sayım da
        engellenir (barkodsuz tekilleştirme yanlış sayı üretir).
        """
        self.barcode_banner = ctk.CTkFrame(
            self.main,
            height=76,
            corner_radius=12,
            fg_color=theme.RED,
        )
        self.barcode_banner.pack_propagate(False)
        text_wrap = ctk.CTkFrame(self.barcode_banner, fg_color="transparent")
        text_wrap.pack(side="left", fill="both", expand=True, padx=16, pady=10)
        self.barcode_banner_title = ctk.CTkLabel(
            text_wrap,
            text="⛔  BARKOD SDK HATASI — SAYIM YAPILAMAZ",
            anchor="w",
            text_color="#FFFFFF",
            font=ctk.CTkFont(theme.FONT, 14, "bold"),
        )
        self.barcode_banner_title.pack(anchor="w")
        self.barcode_banner_detail = ctk.CTkLabel(
            text_wrap,
            text="",
            anchor="w",
            justify="left",
            text_color="#FFE3E3",
            font=ctk.CTkFont(theme.FONT, 11),
        )
        self.barcode_banner_detail.pack(anchor="w")
        self.barcode_retry_btn = ctk.CTkButton(
            self.barcode_banner,
            text="↻  TEKRAR DENE",
            width=170,
            height=52,
            corner_radius=12,
            fg_color="#FFFFFF",
            hover_color="#FFE3E3",
            text_color=theme.RED,
            font=ctk.CTkFont(theme.FONT, 13, "bold"),
            command=lambda: self._start_barcode_probe(initial=False),
        )
        self.barcode_retry_btn.pack(side="right", padx=14, pady=12)

    def _start_barcode_probe(self, initial=False):
        """Barkod motorunu arka planda sınar (dongle sonradan takılmış olabilir)."""
        if self._barcode_probing:
            return
        self._barcode_probing = True
        self.barcode_retry_btn.configure(state="disabled", text="↻  DENENİYOR…")
        threading.Thread(
            target=self._barcode_probe_worker, args=(initial,), daemon=True
        ).start()

    def _barcode_probe_worker(self, initial):
        ready, error = self._probe_barcode_engine()
        self.root.after(0, lambda: self._apply_barcode_state(ready, error, initial))

    def _probe_barcode_engine(self):
        """(hazır, hata) döner. Aremak okuyucusunu gerçekten başlatmayı dener."""
        if not AREMAK_AVAILABLE or AremakDetectionProcessor is None:
            return False, (
                AREMAK_IMPORT_ERROR
                or "Aremak barkod SDK'sı bu ortamda yüklü değil"
            )
        try:
            # load_model hem YOLO'yu hem Aremak okuyucusunu (lazy) hazırlar;
            # tekrar çağrılabilir — başarısız okuyucu her çağrıda yeniden denenir.
            self.detector.load_model(log_fn=self.log)
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"
        if getattr(self.detector, "_reader", None) is None:
            return False, (
                getattr(self.detector, "_reader_init_error", None)
                or "Aremak okuyucu başlatılamadı — USB dongle takılı mı?"
            )
        self._hizli_hazirla()
        return True, None

    # ------------------------------------------------------- HIZLANDIRMA
    def _hizli_hazirla(self):
        """6 Aremak okuyucusunu kurar ve ısıtır (kesit + tam kare). Sayım
        süresine girmesin diye barkod sınamasında, arka planda yapılır."""
        if self.hizli_barkod is None or len(self.hizli_barkod.readers) >= 6:
            return
        try:
            self.hizli_barkod.hazirla(6, tam_kare_isit=True, ornek_frame=self._hizli_ornek_kare())
        except Exception as exc:
            self.log(f"[HIZLI] Okuyucu havuzu kurulamadı: {exc}")

    @staticmethod
    def _hizli_ornek_kare():
        """Isınma için son ham kare; yoksa kamera boyutunda yapay kare."""
        raws = sorted(list(CAPTURE_DIR.glob("*/raw_*.jpg"))
                      + list((REPO_DIR / "TTO" / "captures").glob("DB*/raw_*.jpg"))[-6:],
                      key=lambda p: p.stat().st_mtime)
        if raws:
            frame = cv2.imread(str(raws[-1]))
            if frame is not None:
                return frame
        rng = np.random.default_rng(0)
        return rng.integers(20, 60, (3648, 5472, 3), dtype=np.uint8)

    def _build_hizli_bar(self):
        bar = ctk.CTkFrame(
            self.count_page, height=48, fg_color=theme.SURFACE, corner_radius=12,
            border_width=1, border_color=theme.AMBER,
        )
        bar.pack(fill="x", pady=(0, 9))
        bar.pack_propagate(False)
        ctk.CTkLabel(
            bar, text="HIZLANDIRMA (test)", text_color=theme.AMBER,
            font=ctk.CTkFont(theme.FONT, 9, "bold"),
        ).pack(side="left", padx=15)
        for name in AYAR_ADLARI:
            var = tk.BooleanVar(value=self.hizli[name])
            self._hizli_vars[name] = var
            ctk.CTkCheckBox(
                bar, text=AYAR_ETIKET[name], variable=var,
                command=lambda n=name: self._on_hizli_toggle(n),
                text_color=theme.TEXT_SOFT, font=ctk.CTkFont(theme.FONT, 10, "bold"),
                checkbox_height=20, checkbox_width=20,
            ).pack(side="left", padx=8, pady=7)
        ctk.CTkButton(
            bar, text="⏱ Süre Detayı", width=120, height=32, corner_radius=8,
            fg_color=theme.AMBER, hover_color="#A8730F", text_color="#FFFFFF",
            font=ctk.CTkFont(theme.FONT, 10, "bold"), command=self._open_timing_window,
        ).pack(side="right", padx=(6, 12), pady=7)
        self.hizli_total_label = ctk.CTkLabel(
            bar, text="TOPLAM —", text_color=theme.GREEN,
            font=ctk.CTkFont(theme.MONO, 17, "bold"),
        )
        self.hizli_total_label.pack(side="right", padx=8)
        self.hizli_timing_label = ctk.CTkLabel(
            bar, text="", text_color=theme.TEXT_MUTED,
            font=ctk.CTkFont(theme.MONO, 10, "bold"),
        )
        self.hizli_timing_label.pack(side="right", padx=8)

    def _on_hizli_toggle(self, name):
        self.hizli.set(name, self._hizli_vars[name].get())
        for key, var in self._hizli_vars.items():   # dışlama/zorunluluk sonrası eşitle
            if var.get() != self.hizli[key]:
                var.set(self.hizli[key])
        self.log(f"[HIZLI] Ayar: {self.hizli.ozet()}")

    # ------------------------------------------------------ süre ölçümü
    def _tl_add(self, key, sec):
        rec = getattr(self._tl, "rec", None)
        if rec is not None:
            rec[key] = rec.get(key, 0.0) + sec

    def _install_timers(self):
        """YOLO tahmini ve Aremak taramasına süre sarmalayıcısı takar (davranış
        aynı; yalnız süre o anki kamera kaydına eklenir)."""
        det = self.detector
        if hasattr(det, "_scan_with_aremak") and not getattr(det, "_tto_timed", False):
            orig_scan = det._scan_with_aremak

            def timed_scan(*a, **k):
                t = time.perf_counter()
                try:
                    return orig_scan(*a, **k)
                finally:
                    self._tl_add("barkod", time.perf_counter() - t)

            det._scan_with_aremak = timed_scan
            det._tto_timed = True
        model = getattr(det, "model", None)
        if model is not None and not getattr(model, "_tto_timed", False):
            orig_predict = model.predict

            def timed_predict(*a, **k):
                t = time.perf_counter()
                try:
                    return orig_predict(*a, **k)
                finally:
                    self._tl_add("yolo", time.perf_counter() - t)

            model.predict = timed_predict
            model._tto_timed = True

    def _hizli_sure_yaz(self, stages, kasa, barkod, okunamayan):
        """stages: [(aşama, sn), ...] duvar saati; kamera kayıtları _cam_timing'de."""
        total = time.perf_counter() - self._t_run
        cams = [self._cam_timing[i] for i in sorted(self._cam_timing)]
        self._last_timing = {
            "zaman": datetime.now().strftime("%H:%M:%S"),
            "ayar": self.hizli.ozet(),
            "asamalar": stages,
            "toplam": total,
            "kameralar": cams,
            "kasa": kasa, "barkod": barkod, "okunamayan": okunamayan,
            "ocr": None, "ocr_kasa": 0, "toplam_ocr": None,
        }
        self._refresh_timing_views()
        parts = " · ".join(f"{name} {sec:.1f}" for name, sec in stages)
        self.log(f"[HIZLI] SÜRE {total:.2f} sn · {parts} · ayar: {self.hizli.ozet()}")
        self._write_timing_record()

    def _write_timing_record(self, extra=None):
        t = self._last_timing
        if not t:
            return
        try:
            with open(SURE_KAYIT_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "zaman": datetime.now().isoformat(timespec="seconds"),
                    "ayar": dict(self.hizli.values),
                    "toplam_sn": round(t["toplam"], 3),
                    "asamalar_sn": {n: round(v, 3) for n, v in t["asamalar"]},
                    "kameralar": [{k: (round(v, 3) if isinstance(v, float) else v)
                                   for k, v in c.items()} for c in t["kameralar"]],
                    "ocr_sn": None if t["ocr"] is None else round(t["ocr"], 3),
                    "toplam_ocr_dahil_sn": None if t["toplam_ocr"] is None else round(t["toplam_ocr"], 3),
                    "kasa": t["kasa"], "tekil_barkod": t["barkod"], "okunamayan": t["okunamayan"],
                    **(extra or {}),
                }, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _timing_report_text(self):
        t = self._last_timing
        if not t:
            return "Henüz sayım yapılmadı."
        lines = [f"Sayım {t['zaman']}  ·  {t['ayar']}",
                 f"Sonuç: {t['kasa']} kasa · {t['barkod']} tekil barkod · {t['okunamayan']} okunamayan",
                 "",
                 "AŞAMALAR (duvar saati)"]
        for name, sec in t["asamalar"]:
            share = 100 * sec / t["toplam"] if t["toplam"] else 0
            lines.append(f"  {name:<34}{sec:7.2f} sn   %{share:4.0f}")
        lines.append(f"  {'TOPLAM (sayım)':<34}{t['toplam']:7.2f} sn")
        if t["ocr"] is not None:
            lines.append(f"  {'OCR (' + str(t['ocr_kasa']) + ' kasa, arka planda)':<34}{t['ocr']:7.2f} sn")
            lines.append(f"  {'TOPLAM (OCR dahil)':<34}{t['toplam_ocr']:7.2f} sn")
        lines += ["", "KAMERALAR (sn; 'başla/bitiş' = sayım başından itibaren)"]
        cols = [("kamera", "Kam", "{:>4}"), ("başla", "Başla", "{:6.2f}"),
                ("sıra bekleme", "Sıra", "{:6.2f}"), ("çekim", "Çekim", "{:6.2f}"),
                ("ham kayıt", "HamKyt", "{:6.2f}"), ("yolo bekleme", "YOLOsıra", "{:8.2f}"),
                ("yolo", "YOLO", "{:6.2f}"), ("barkod", "Barkod", "{:6.2f}"),
                ("barkod bitti", "BrkBitti", "{:8.2f}"),
                ("kayıt+diğer", "Kayıt", "{:6.2f}"), ("bitiş", "Bitiş", "{:6.2f}"),
                ("kasa", "Kasa", "{:>5}")]
        lines.append("  " + " ".join(f"{h:>{max(6, len(h))}}" for _, h, _ in cols))
        for c in t["kameralar"]:
            row = []
            for key, head, fmt in cols:
                v = c.get(key)
                w = max(6, len(head))
                row.append(f"{'—':>{w}}" if v is None else f"{fmt.format(v):>{w}}")
            lines.append("  " + " ".join(row))
        lines += ["",
                  "Sıra: kameranın aktarım sırası beklemesi (aynı anda en fazla 2 kamera).",
                  "YOLOsıra: YOLO modeli tek; diğer kameranın YOLO'sunu bekleme.",
                  "Barkod: kutular kapalıyken kameranın kendi okuma süresi; havuz/tam kare",
                  "modunda barkod aşaması başından o kameranın son kasasının bittiği ana kadar.",
                  "Akış (çekerken oku) modunda aşamalar üst üste biner: her satır, bir önceki",
                  "aşamanın SON kamerası bittikten sonra kalan ek süredir; toplamları = sayım."]
        return "\n".join(lines)

    def _refresh_timing_views(self):
        t = self._last_timing

        def apply():
            if t:
                total = t["toplam_ocr"] if t["toplam_ocr"] is not None else t["toplam"]
                self.hizli_total_label.configure(
                    text=f"TOPLAM {total:.1f} sn" + (" (OCR dahil)" if t["toplam_ocr"] is not None else ""))
                self.hizli_timing_label.configure(
                    text=" · ".join(f"{n.split(' (')[0]} {v:.1f}" for n, v in t["asamalar"]))
            if self._timing_text is not None:
                try:
                    self._timing_text.configure(state="normal")
                    self._timing_text.delete("1.0", "end")
                    self._timing_text.insert("end", self._timing_report_text())
                    self._timing_text.configure(state="disabled")
                except tk.TclError:
                    self._timing_text = None

        self.root.after(0, apply)

    def _open_timing_window(self):
        if self._timing_window is not None:
            try:
                self._timing_window.deiconify()
                self._timing_window.lift()
                self._refresh_timing_views()
                return
            except tk.TclError:
                self._timing_window = None
        win = ctk.CTkToplevel(self.root)
        win.title("Süre Detayı · TTO Hızlandırılmış")
        win.geometry("980x560")
        win.attributes("-topmost", True)
        text = tk.Text(win, font=(theme.MONO, 11), bg=theme.TERMINAL_BG, fg=theme.TERMINAL_TEXT,
                       bd=0, padx=14, pady=12, wrap="none")
        text.pack(fill="both", expand=True)

        def closed():
            self._timing_window = None
            self._timing_text = None
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", closed)
        self._timing_window, self._timing_text = win, text
        self._refresh_timing_views()

    def _save_review_crops_hizli(self, frame, result, save_dir, timestamp):
        """_save_review_crops ile aynı dosyalar; yazma io havuzunda paralel."""
        review_dir = save_dir / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        height, width = frame.shape[:2]
        jobs = []
        for kasa in result.get("kasalar") or []:
            x1, y1, x2, y2 = (int(value) for value in kasa["bbox"])
            x1, x2 = max(0, x1), min(width, x2)
            y1, y2 = max(0, y1), min(height, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop_path = review_dir / f"{timestamp}_kasa_{kasa['kasa_no']}.jpg"
            kasa["tto_crop_path"] = str(crop_path)
            jobs.append((str(crop_path), frame[y1:y2, x1:x2]))
        list(self._io_pool.map(lambda job: cv2.imwrite(*job), jobs))

    def _capture_all_worker_hizli(self):
        """Sayım akışı, HIZLANDIRMA kutularına göre:
        1) çekim + ham kayıt + YOLO (kamera iş parçacıkları; YOLO kilit altında)
        2) barkod — kamera kamera / havuz / tam kare, paralel ya da tek okuyucu
        3) sonuç + filtre + kesit kayıtları, ardından tekilleştirme (TTO ile aynı)."""
        ayar = self.hizli
        self.set_status("Sayım yapılıyor", theme.AMBER)
        self.log(f"[HIZLI] Sayım başladı · {ayar.ozet()}")
        t_start = self._t_run
        self._tl.rec = None
        kalemler = [None] * len(self.cameras)
        fast_save = ayar["hizli_kayit"]

        def capture(index, camera):
            tile = self.tiles[index]
            if self._capture_cancel.is_set():
                self.root.after(0, lambda: tile.set_busy(False))
                return
            rec = {"kamera": index + 1, "başla": time.perf_counter() - t_start}
            self._cam_timing[index] = rec
            try:
                self.log(f"Kamera {index + 1}: kare yakalanıyor…")
                t_req = time.perf_counter()
                with self._capture_slots:
                    t_got = time.perf_counter()
                    frame = camera.capture(timeout_ms=5000, fresh=True)
                t_cap = time.perf_counter()
                rec.update({"sıra bekleme": t_got - t_req, "çekim": t_cap - t_got,
                            "çekim bitti": t_cap - t_start})
                self.root.after(0, lambda: tile.show_image(frame))
                folder = str(camera.serial or f"kamera_{index + 1}").replace(" ", "_")
                save_dir = CAPTURE_DIR / folder
                save_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                raw_path = save_dir / f"raw_{timestamp}.jpg"
                raw_future = None
                t = time.perf_counter()
                if fast_save:
                    raw_future = self._io_pool.submit(cv2.imwrite, str(raw_path), frame)
                else:
                    cv2.imwrite(str(raw_path), frame)
                rec["ham kayıt"] = time.perf_counter() - t
                t = time.perf_counter()
                with self.processing_lock:   # YOLO modeli paylaşımlı
                    t_lock = time.perf_counter()
                    boxes = self.hizli_barkod.tespit(frame)
                rec["yolo bekleme"] = t_lock - t
                rec["yolo"] = time.perf_counter() - t_lock
                rec["yolo bitti"] = time.perf_counter() - t_start
                rec["kasa"] = len(boxes)
                kalemler[index] = {
                    "index": index, "frame": frame, "save_dir": save_dir,
                    "timestamp": timestamp, "raw_path": str(raw_path),
                    "raw_future": raw_future, "tile": tile,
                    "crates": [{"kasa_no": i + 1, "bbox": b, "barkodlar": [], "barkod_okundu": False}
                               for i, b in enumerate(boxes)],
                }
                self.log(f"Kamera {index + 1}: {len(boxes)} kutu (YOLO).")
            except Exception as exc:
                self.log(f"Kamera {index + 1} HATA: {exc}")
                self.root.after(0, lambda: tile.set_busy(False))

        threads = [threading.Thread(target=capture, args=(i, c), daemon=True)
                   for i, c in enumerate(self.cameras)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        cams = [self._cam_timing[i] for i in self._cam_timing]
        t_cekim = max((c.get("çekim bitti", 0) for c in cams), default=0.0)
        t_yolo = time.perf_counter() - t_start
        stages = [("Fotoğraf çekimi (6 kamera)", t_cekim),
                  ("Ham kayıt + YOLO (çekimden sonra)", t_yolo - t_cekim)]

        if self._capture_cancel.is_set():
            self.root.after(0, self._finish_cancelled_capture)
            return
        items = [k for k in kalemler if k]

        t = time.perf_counter()
        try:
            self.hizli_barkod.barkod_oku(items, ayar)
        except Exception as exc:
            self.log(f"[HIZLI] Barkod HATA: {exc}")
        stages.append(("Barkod", time.perf_counter() - t))
        for item in items:
            self._cam_timing[item["index"]]["barkod"] = item.get("barkod_sn", 0.0)

        t = time.perf_counter()

        def finalize(item):
            t_f = time.perf_counter()
            result = self.hizli_barkod.sonuc_kur(item, item["save_dir"], fast_save, self._io_pool)
            self._filter_background_crates(item["index"], result, frame_height=item["frame"].shape[0])
            if fast_save:
                self._save_review_crops_hizli(item["frame"], result, item["save_dir"], item["timestamp"])
            else:
                self._save_review_crops(item["frame"], result, item["save_dir"], item["timestamp"])
            if item["raw_future"] is not None:
                item["raw_future"].result()
            item["result"] = result
            rec = self._cam_timing[item["index"]]
            rec["kayıt+diğer"] = time.perf_counter() - t_f
            rec["bitiş"] = time.perf_counter() - t_start
            rec["kasa"] = result["toplam_kasa"]

        if fast_save and len(items) > 1:
            with ThreadPoolExecutor(max_workers=len(items)) as pool:
                list(pool.map(finalize, items))
        else:
            for item in items:
                finalize(item)
        for item in items:   # UI/kayıt listeleri tek iş parçacığından
            index, tile, result = item["index"], item["tile"], item["result"]
            self.results[index] = result
            tile.last_result = result
            self.root.after(0, lambda tl=tile, r=result: self._apply_camera_result(tl, r))
            self._register_unread(index, result, item["raw_path"])
            self.log(
                f"Kamera {index + 1}: {result['toplam_kasa']} kasa, "
                f"{result['okunan_barkod']} barkod, "
                f"{result['okunamayan_barkod']} okunamayan · barkod {item.get('barkod_sn', 0):.2f} sn"
            )
            self.root.after(0, lambda tl=tile: tl.set_busy(False))
        stages.append(("Sonuç + filtre + kayıt", time.perf_counter() - t))

        health = self._barcode_health_after_capture()
        if health is not None:
            self.root.after(0, lambda: self._abort_count_barcode(health))
            return

        t = time.perf_counter()
        aggregate = self._aggregate()
        stages.append(("Tekilleştirme", time.perf_counter() - t))
        timing = {"toplam": time.perf_counter() - t_start}
        self._hizli_sure_yaz(
            stages,
            aggregate["unique_total"] if aggregate else 0,
            aggregate["unique_barcodes"] if aggregate else 0,
            sum(r["okunamayan_barkod"] for r in self.results if r),
        )

        def finish_count():
            self._counting = False
            self._set_capture_all_state("normal")
            if aggregate is not None:
                self._show_page("pallet")
            elif self.active_page == "pallet":
                self._refresh_pallet_view()

        self.root.after(0, finish_count)
        self.set_status(f"Sayım tamamlandı · {timing['toplam']:.1f} sn", theme.GREEN)
        self.log("Tüm kamera işlemleri tamamlandı.")

    def _apply_barcode_state(self, ready, error, initial=False):
        """Şeridi, durum etiketini ve sayım düğmelerini barkod durumuna göre kurar."""
        self._barcode_probing = False
        was_ready = self.barcode_ready
        self.barcode_ready = bool(ready)
        self.barcode_error = error
        self.barcode_retry_btn.configure(state="normal", text="↻  TEKRAR DENE")
        if ready:
            self.barcode_banner.pack_forget()
            self.log(f"✔ Barkod motoru hazır: {self.barcode_engine}")
            if not was_ready and not initial:
                self._show_toast("✓ Barkod okuyucu hazır — sayım yapabilirsiniz")
            if self.cameras:
                self.set_status(f"{len(self.cameras)} kamera bağlı", theme.GREEN)
        else:
            self.barcode_banner.pack(
                fill="x", padx=18, pady=(0, 8), before=self.page_container
            )
            detail = str(error or "").replace("\n", " ").strip()
            if len(detail) > 190:
                detail = detail[:190] + "…"
            self.barcode_banner_detail.configure(
                text=f"{detail}  ·  USB dongle takılı mı? Takıp TEKRAR DENE'ye basın."
            )
            self.set_status("BARKOD SDK HATASI", theme.RED)
            self.log(f"[HATA] Barkod motoru hazır değil: {error}")
        self._update_capture_enabled()

    def _update_capture_enabled(self):
        """Sayım düğmeleri yalnız kamera + barkod motoru hazırken açık olur."""
        can = bool(self.cameras) and self.barcode_ready
        self._set_capture_all_state("normal" if can else "disabled")
        for tile in self.tiles:
            if tile.camera is not None:
                tile.capture_btn.configure(
                    state="normal" if self.barcode_ready else "disabled"
                )

    def _block_without_barcode(self):
        """Barkod motoru hazır değilse işlemi durdurur ve nedenini gösterir."""
        if self.barcode_ready:
            return False
        self._show_toast(
            "⛔ Barkod SDK hatası — dongle takılı değil, sayım yapılamaz", theme.RED
        )
        self._apply_barcode_state(False, self.barcode_error)
        return True

    def _build_count_page(self):
        self.metrics_frame = ctk.CTkFrame(
            self.count_page, fg_color="transparent"
        )
        self.metrics_frame.pack(fill="x", pady=(4, 10))
        for column in range(3):
            self.metrics_frame.grid_columnconfigure(
                column, weight=1, uniform="metrics"
            )
        self.metric_total = MetricCard(self.metrics_frame, "TOPLAM KASA", theme.GREEN, "▣")
        self.metric_barcode = MetricCard(self.metrics_frame, "TEKİL BARKOD", theme.CYAN, "▥")
        self.metric_unread = MetricCard(self.metrics_frame, "OKUNAMAYAN", theme.RED, "!")
        self.metric_cards = (
            self.metric_total,
            self.metric_barcode,
            self.metric_unread,
        )
        for column, card in enumerate(self.metric_cards):
            card.grid(row=0, column=column, sticky="ew", padx=5)

        group_bar = ctk.CTkFrame(
            self.count_page,
            height=48,
            fg_color=theme.SURFACE,
            corner_radius=12,
            border_width=1,
            border_color=theme.BORDER,
        )
        group_bar.pack(fill="x", pady=(0, 9))
        group_bar.pack_propagate(False)
        ctk.CTkLabel(
            group_bar,
            text="GÖRÜNÜM",
            text_color=theme.TEXT_MUTED,
            font=ctk.CTkFont(theme.FONT, 9, "bold"),
        ).pack(side="left", padx=15)

        # Simge yerine yazı: "▫ / ▫▫▫▫▫▫" operatör için anlaşılmıyordu.
        self.filter_buttons = {}
        for count, symbol in ((1, "Tek Kamera"), (6, "6 Kamera")):
            button = ctk.CTkButton(
                group_bar,
                text=symbol,
                width={1: 112, 6: 104}[count],
                height=32,
                corner_radius=8,
                fg_color="transparent",
                hover_color=theme.SURFACE_LIGHT,
                text_color=theme.TEXT_SOFT,
                font=ctk.CTkFont(theme.FONT, 10, "bold"),
                command=lambda value=count: self._set_camera_filter(value),
            )
            button.pack(side="left", padx=3, pady=7)
            self.filter_buttons[count] = button

        self._build_hizli_bar()

        # Model seçici: Desktop\Models içindeki .pt dosyaları + varsayılan.
        # Farklı modellerin arka kasalara nasıl kutu attığını denemek için.
        self._model_paths = {}
        if MODELS_DIR.is_dir():
            for path in sorted(MODELS_DIR.glob("*.pt")):
                self._model_paths[path.name] = str(path)
        self._model_paths[DEFAULT_MODEL_LABEL] = None
        current = os.environ.get("ODAI_YOLO_MODEL", "")
        current_label = os.path.basename(current) if current else DEFAULT_MODEL_LABEL
        if current_label not in self._model_paths:
            current_label = DEFAULT_MODEL_LABEL
        ctk.CTkLabel(
            group_bar,
            text="MODEL",
            text_color=theme.TEXT_MUTED,
            font=ctk.CTkFont(theme.FONT, 9, "bold"),
        ).pack(side="left", padx=(16, 5))
        self.model_menu = ctk.CTkOptionMenu(
            group_bar,
            values=list(self._model_paths),
            command=self._on_model_selected,
            width=250,
            height=32,
            corner_radius=8,
            fg_color=theme.SURFACE_LIGHT,
            button_color=theme.GREEN_DARK,
            button_hover_color=theme.BORDER_ACTIVE,
            text_color=theme.TEXT,
            dropdown_fg_color=theme.SURFACE,
            dropdown_text_color=theme.TEXT,
            dropdown_hover_color=theme.SURFACE_LIGHT,
            font=ctk.CTkFont(theme.MONO, 9),
            dropdown_font=ctk.CTkFont(theme.MONO, 10),
        )
        self.model_menu.set(current_label)
        self.model_menu.pack(side="left", padx=3, pady=7)

        self.next_camera_btn = ctk.CTkButton(
            group_bar,
            text="›",
            width=36,
            height=32,
            corner_radius=8,
            fg_color=theme.SURFACE_LIGHT,
            hover_color=theme.BORDER_ACTIVE,
            text_color=theme.TEXT,
            font=ctk.CTkFont(theme.FONT, 18, "bold"),
            command=lambda: self._navigate_cameras(1),
        )
        self.next_camera_btn.pack(side="right", padx=(3, 12), pady=7)
        self.previous_camera_btn = ctk.CTkButton(
            group_bar,
            text="‹",
            width=36,
            height=32,
            corner_radius=8,
            fg_color=theme.SURFACE_LIGHT,
            hover_color=theme.BORDER_ACTIVE,
            text_color=theme.TEXT,
            font=ctk.CTkFont(theme.FONT, 18, "bold"),
            command=lambda: self._navigate_cameras(-1),
        )
        self.previous_camera_btn.pack(side="right", padx=3, pady=7)
        self.group_info = ctk.CTkLabel(
            group_bar,
            text="Tüm Kameralar 1—6",
            text_color=theme.TEXT_MUTED,
            font=ctk.CTkFont(theme.FONT, 9),
        )
        self.group_info.pack(side="right", padx=9)

        self.tiles_frame = ctk.CTkScrollableFrame(
            self.count_page, fg_color="transparent"
        )
        self.tiles_frame.pack(fill="both", expand=True)
        for column in range(3):
            self.tiles_frame.grid_columnconfigure(column, weight=1, uniform="tiles")
        for index in range(MAX_CAMERAS):
            tile = CameraTile(
                self.tiles_frame,
                index,
                self._on_single_capture,
                self._manual_upload_for_tile,
                self._open_single_camera,
                self._open_review_page,
            )
            tile.crate_click_command = self._on_image_crate_click
            tile.fullscreen_command = self._open_fullscreen_camera
            tile.cancel_command = self._cancel_capture
            tile.rename_command = self._open_camera_rename
            self.tiles.append(tile)
        canvas = getattr(self.tiles_frame, "_parent_canvas", None)
        if canvas is not None:
            # add="+" şart: CTk'nın kendi genişlik-uydurma bind'i korunmalı.
            canvas.bind(
                "<Configure>", lambda _event: self._schedule_tile_resize(), add="+"
            )
        self._apply_camera_filter()

    # ------------------------------------------------- kart görüntü boyutlama
    TILE_CHROME = 122  # başlık + pozlama satırı + iç boşluklar (px)

    def _schedule_tile_resize(self):
        if self._tile_resize_job is not None:
            try:
                self.root.after_cancel(self._tile_resize_job)
            except tk.TclError:
                pass
        self._tile_resize_job = self.root.after(120, self._update_tile_sizes)

    def _update_tile_sizes(self):
        """Görüntü alanlarını, karta tam genişlikte sığacak şekilde boyutlar.

        Dik ekranda görüntü yüksekliği en-boy oranından hesaplanır (beyaz
        boşluk kalmaz, gerekirse sayfa kaydırılır); yatayda kartlar ekrana
        tam sığacak şekilde bölünür.
        """
        self._tile_resize_job = None
        canvas = getattr(self.tiles_frame, "_parent_canvas", None)
        width = canvas.winfo_width() if canvas is not None else self.tiles_frame.winfo_width()
        height = canvas.winfo_height() if canvas is not None else self.tiles_frame.winfo_height()
        if width < 100 or height < 100:
            return
        if self.camera_filter == 1:
            columns, rows = 1, 1
        else:
            columns = 2 if self.portrait_mode else 3
            rows = 3 if self.portrait_mode else 2
        # Kart yüksekliği = görüntü + chrome (başlık/pozlama). Görüntü yüksekliği
        # satır sayısına bölünerek verilir; böylece 6 kamera dikeyde de yatayda da
        # KAYDIRMASIZ ekrana sığar. chrome ölçülerek alınır (sabit tahmin değil).
        chrome = max(
            (tile.chrome_height() for tile in self.tiles), default=self.TILE_CHROME
        )
        row_gap = 6  # tile.grid(pady=3) → üst + alt
        tile_width = width / columns - 10
        image_width = max(120, tile_width - 28)
        image_height = max(120, int(height / rows) - chrome - row_gap)
        self._apply_tile_height(image_height, height, rows)

    # Tahmini chrome her zaman birebir tutmaz (yazı tipi, DPI, tema). Bu yüzden
    # yükseklik uygulanıp GERÇEK içerik ölçülür ve kalan taşma satırlara bölünüp
    # bir kez düzeltilir. Böylece 6 kamera her ekranda kaydırmasız oturur.
    def _apply_tile_height(self, image_height, area_height, rows, _pass=0):
        if self._tile_fitting:
            return
        self._tile_fitting = True
        try:
            for tile in self.tiles:
                tile.set_image_height(image_height)
            self.tiles_frame.update_idletasks()
            overflow = self.tiles_frame.winfo_reqheight() - area_height
            if overflow > 2 and _pass < 2 and image_height > 120:
                corrected = max(120, image_height - (overflow // rows) - 1)
                if corrected != image_height:
                    self._tile_fitting = False
                    self._apply_tile_height(corrected, area_height, rows, _pass + 1)
                    return
        finally:
            self._tile_fitting = False
        self._sync_tiles_scrollbar()

    def _sync_tiles_scrollbar(self):
        """İçerik sığıyorsa kamera alanındaki kaydırma çubuğunu gizler."""
        scrollbar = getattr(self.tiles_frame, "_scrollbar", None)
        canvas = getattr(self.tiles_frame, "_parent_canvas", None)
        if scrollbar is None or canvas is None:
            return
        try:
            self.tiles_frame.update_idletasks()
            if self.tiles_frame.winfo_reqheight() <= canvas.winfo_height() + 2:
                scrollbar.grid_remove()
            else:
                scrollbar.grid()
        except tk.TclError:
            pass

    def _build_unread_page(self):
        body = ctk.CTkFrame(self.unread_page, fg_color="transparent")
        body.pack(fill="both", expand=True, pady=(4, 0))

        left = ctk.CTkFrame(
            body,
            width=350,
            fg_color=theme.SURFACE,
            corner_radius=14,
            border_width=1,
            border_color=theme.BORDER,
        )
        left.pack(side="left", fill="y", padx=(0, 10))
        left.pack_propagate(False)
        ctk.CTkLabel(
            left,
            text="OKUNAMAYAN BARKODLAR",
            anchor="w",
            text_color=theme.GREEN,
            font=ctk.CTkFont(theme.FONT, 9, "bold"),
        ).pack(fill="x", padx=14, pady=(16, 2))
        self.unread_count = ctk.CTkLabel(
            left,
            text="0 kayıt",
            anchor="w",
            text_color=theme.TEXT,
            font=ctk.CTkFont(theme.FONT, 18, "bold"),
        )
        self.unread_count.pack(fill="x", padx=14, pady=(0, 12))

        list_wrap = ctk.CTkFrame(left, fg_color=theme.BG_RAISED, corner_radius=10)
        list_wrap.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.unread_list = tk.Listbox(
            list_wrap,
            bg=theme.BG_RAISED,
            fg=theme.TEXT_SOFT,
            selectbackground=theme.GREEN_DARK,
            selectforeground=theme.GREEN,
            font=(theme.MONO, 10),
            activestyle="none",
            bd=0,
            highlightthickness=0,
        )
        scrollbar = tk.Scrollbar(list_wrap, command=self.unread_list.yview)
        self.unread_list.configure(yscrollcommand=scrollbar.set)
        self.unread_list.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=8)
        scrollbar.pack(side="right", fill="y", padx=(0, 8), pady=8)
        self.unread_list.bind("<<ListboxSelect>>", self._on_unread_select)

        right = ctk.CTkFrame(
            body,
            fg_color=theme.SURFACE,
            corner_radius=14,
            border_width=1,
            border_color=theme.BORDER,
        )
        right.pack(side="left", fill="both", expand=True)

        viewer_toolbar = ctk.CTkFrame(right, fg_color="transparent", height=55)
        viewer_toolbar.pack(fill="x", padx=13)
        self.viewer_title = ctk.CTkLabel(
            viewer_toolbar,
            text="İncelenecek kaydı seçin",
            text_color=theme.TEXT_SOFT,
            font=ctk.CTkFont(theme.FONT, 11, "bold"),
        )
        self.viewer_title.pack(side="left", pady=12)
        for text, kind in (("Crop", "crop"), ("Ham", "raw"), ("İşlenmiş", "annotated"), ("Barkod", "prep")):
            ctk.CTkButton(
                viewer_toolbar,
                text=text,
                width=74,
                height=30,
                corner_radius=8,
                fg_color=theme.SURFACE_LIGHT,
                hover_color=theme.BORDER_ACTIVE,
                font=ctk.CTkFont(theme.FONT, 9, "bold"),
                command=lambda selected=kind: self._show_unread_kind(selected),
            ).pack(side="right", padx=3, pady=10)

        self.viewer_frame = ctk.CTkFrame(
            right,
            fg_color=theme.IMAGE_BG,
            corner_radius=10,
            border_width=1,
            border_color=theme.BORDER,
        )
        self.viewer_frame.pack(fill="both", expand=True, padx=13, pady=(0, 13))
        self.viewer_label = tk.Label(
            self.viewer_frame,
            text="Okunamayan barkod kayıtları burada görüntülenir",
            bg=theme.IMAGE_BG,
            fg=theme.TEXT_MUTED,
            font=(theme.FONT, 11),
        )
        self.viewer_label.pack(fill="both", expand=True)
        self.viewer_frame.bind("<Configure>", lambda _event: self._render_viewer())

    # ------------------------------------------------------- OCR okumaları sayfası
    def _build_ocr_page(self):
        """Barkodu okunamayıp OCR'ın basılı numaradan çözdüğü kasaların listesi."""
        body = ctk.CTkFrame(self.ocr_page, fg_color="transparent")
        body.pack(fill="both", expand=True, pady=(4, 0))

        left = ctk.CTkFrame(
            body,
            width=350,
            fg_color=theme.SURFACE,
            corner_radius=14,
            border_width=1,
            border_color=theme.BORDER,
        )
        left.pack(side="left", fill="y", padx=(0, 10))
        left.pack_propagate(False)
        ctk.CTkLabel(
            left,
            text="⚡ SİSTEM TARAFINDAN TESPİT EDİLEN KASALAR",
            anchor="w",
            text_color=theme.CYAN,
            font=ctk.CTkFont(theme.FONT, 9, "bold"),
        ).pack(fill="x", padx=14, pady=(16, 2))
        self.ocr_count = ctk.CTkLabel(
            left,
            text="0 kasa",
            anchor="w",
            text_color=theme.TEXT,
            font=ctk.CTkFont(theme.FONT, 18, "bold"),
        )
        self.ocr_count.pack(fill="x", padx=14, pady=(0, 4))
        self.ocr_engine_status = ctk.CTkLabel(
            left,
            text="",
            anchor="w",
            justify="left",
            wraplength=310,
            text_color=theme.TEXT_MUTED,
            font=ctk.CTkFont(theme.FONT, 10),
        )
        self.ocr_engine_status.pack(fill="x", padx=14, pady=(0, 10))

        list_wrap = ctk.CTkFrame(left, fg_color=theme.BG_RAISED, corner_radius=10)
        list_wrap.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.ocr_list = tk.Listbox(
            list_wrap,
            bg=theme.BG_RAISED,
            fg=theme.TEXT_SOFT,
            selectbackground=theme.GREEN_DARK,
            selectforeground=theme.CYAN,
            font=(theme.MONO, 10),
            activestyle="none",
            bd=0,
            highlightthickness=0,
        )
        scrollbar = tk.Scrollbar(list_wrap, command=self.ocr_list.yview)
        self.ocr_list.configure(yscrollcommand=scrollbar.set)
        self.ocr_list.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=8)
        scrollbar.pack(side="right", fill="y", padx=(0, 8), pady=8)
        self.ocr_list.bind("<<ListboxSelect>>", self._on_ocr_select)

        right = ctk.CTkFrame(
            body,
            fg_color=theme.SURFACE,
            corner_radius=14,
            border_width=1,
            border_color=theme.BORDER,
        )
        right.pack(side="left", fill="both", expand=True)

        viewer_toolbar = ctk.CTkFrame(right, fg_color="transparent", height=55)
        viewer_toolbar.pack(fill="x", padx=13)
        self.ocr_viewer_title = ctk.CTkLabel(
            viewer_toolbar,
            text="İncelenecek kaydı seçin",
            text_color=theme.TEXT_SOFT,
            font=ctk.CTkFont(theme.FONT, 11, "bold"),
        )
        self.ocr_viewer_title.pack(side="left", pady=12)
        ctk.CTkButton(
            viewer_toolbar,
            text="✎ Düzelt",
            width=90,
            height=30,
            corner_radius=8,
            fg_color=theme.SURFACE_LIGHT,
            hover_color=theme.BORDER_ACTIVE,
            font=ctk.CTkFont(theme.FONT, 9, "bold"),
            command=self._correct_selected_ocr,
        ).pack(side="right", padx=3, pady=10)

        self.ocr_viewer_frame = ctk.CTkFrame(
            right,
            fg_color=theme.IMAGE_BG,
            corner_radius=10,
            border_width=1,
            border_color=theme.BORDER,
        )
        self.ocr_viewer_frame.pack(fill="both", expand=True, padx=13, pady=(0, 13))
        self.ocr_viewer_label = tk.Label(
            self.ocr_viewer_frame,
            text="Barkodu okunamayıp basılı numarasından sistemin (OCR) tespit\n"
            "ettiği kasalar burada listelenir; kaydı seçince görüntüsü açılır",
            bg=theme.IMAGE_BG,
            fg=theme.TEXT_MUTED,
            font=(theme.FONT, 11),
            justify="center",
        )
        self.ocr_viewer_label.pack(fill="both", expand=True)
        self._ocr_rows = []
        self._ocr_selected = None
        self._ocr_viewer_photo = None
        self.ocr_viewer_frame.bind(
            "<Configure>", lambda _event: self._render_ocr_viewer()
        )

    def _refresh_ocr_list(self):
        self.ocr_list.delete(0, "end")
        self._ocr_rows = []
        for index, crate in enumerate(self.pallet_crates):
            if not crate.get("ocr"):
                continue
            cam = crate.get("camera_index")
            cam_text = f"K{cam + 1}" if cam is not None else "K?"
            kasa_no = crate.get("kasa_no") or index + 1
            self.ocr_list.insert(
                "end",
                f"{cam_text}  ·  Kasa {kasa_no:>2}  →  {crate['serial']}",
            )
            self._ocr_rows.append(index)
        self.ocr_count.configure(text=f"{len(self._ocr_rows)} kasa")
        error = ocr_reader.availability_error()
        if error:
            self.ocr_engine_status.configure(
                text=f"⚠ OCR devre dışı: {error}", text_color=theme.RED
            )
        else:
            self.ocr_engine_status.configure(
                text="Tespit motoru hazır (PaddleOCR PP-OCRv5) — okunamayan kasalar otomatik denenir",
                text_color=theme.TEXT_MUTED,
            )
        if self._ocr_selected not in self._ocr_rows:
            self._ocr_selected = None
            self.ocr_viewer_title.configure(text="İncelenecek kaydı seçin")

    def _on_ocr_select(self, _event=None):
        selected = self.ocr_list.curselection()
        if not selected or int(selected[0]) >= len(self._ocr_rows):
            return
        self._ocr_selected = self._ocr_rows[int(selected[0])]
        crate = self.pallet_crates[self._ocr_selected]
        cam = crate.get("camera_index")
        cam_text = f"Kamera {cam + 1}" if cam is not None else "Kamera ?"
        self.ocr_viewer_title.configure(
            text=f"{cam_text} · Kasa {crate.get('kasa_no') or self._ocr_selected + 1}"
            f" → Seri {crate['serial']}  (sistem tespiti)"
        )
        self._render_ocr_viewer()

    def _render_ocr_viewer(self):
        if self._ocr_selected is None or self._ocr_selected >= len(self.pallet_crates):
            return
        crate = self.pallet_crates[self._ocr_selected]
        image = self._crate_context_image(crate)
        if image is None:
            self.ocr_viewer_label.configure(
                image="", text="Kasa görüntüsü bulunamadı"
            )
            return
        try:
            width = max(200, self.ocr_viewer_frame.winfo_width() - 8)
            height = max(200, self.ocr_viewer_frame.winfo_height() - 8)
            scale = min(width / image.width, height / image.height)
            image = image.resize(
                (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
                Image.Resampling.LANCZOS,
            )
            self._ocr_viewer_photo = ImageTk.PhotoImage(image)
            self.ocr_viewer_label.configure(image=self._ocr_viewer_photo, text="")
        except OSError:
            pass

    def _correct_selected_ocr(self):
        """Seçili OCR kaydı yanlışsa operatör seriyi elle düzeltir."""
        if self._ocr_selected is None or self._ocr_selected >= len(self.pallet_crates):
            self._show_toast("Önce listeden bir kayıt seçin", theme.AMBER)
            return
        self._open_serial_modal(self._ocr_selected)

    # ---------------------------------------------------------- UI davranışı
    def _show_page(self, page):
        self.active_page = page
        self._close_serial_modal()
        for frame in (
            self.session_page,
            self.count_page,
            self.unread_page,
            self.ocr_page,
            self.pallet_page,
            self.summary_page,
        ):
            frame.pack_forget()

        nav_map = {
            "session": (self.nav_session, self.portrait_session_btn),
            "count": (self.nav_count, self.portrait_count_btn),
            "pallet": (self.nav_pallet, self.portrait_pallet_btn),
            "unread": (self.nav_unread, self.portrait_unread_btn),
            "ocr": (self.nav_ocr, self.portrait_ocr_btn),
        }
        active_key = "pallet" if page == "summary" else page
        for key, (nav, portrait_btn) in nav_map.items():
            selected = key == active_key
            nav.configure(
                fg_color=theme.SURFACE if selected else "transparent",
                text_color=theme.GREEN if selected else theme.TEXT_SOFT,
            )
            portrait_btn.configure(
                fg_color=theme.GREEN_DARK if selected else theme.SURFACE_LIGHT,
                text_color=theme.GREEN if selected else theme.TEXT,
            )

        if page == "session":
            self.session_page.pack(fill="both", expand=True)
            self.page_eyebrow.configure(text="SEVKİYAT")
            self.page_title.configure(text="Tır / İrsaliye Süreci")
            self._refresh_session_page()
        elif page == "unread":
            self.unread_page.pack(fill="both", expand=True)
            self.page_eyebrow.configure(text="MANUEL KONTROL")
            self.page_title.configure(text="Okunamayan Barkodlar")
        elif page == "ocr":
            self.ocr_page.pack(fill="both", expand=True)
            self.page_eyebrow.configure(text="OTOMATİK TESPİT")
            self.page_title.configure(text="Sistem Tespitleri")
            self._refresh_ocr_list()
        elif page == "pallet":
            self.pallet_page.pack(fill="both", expand=True)
            self.page_eyebrow.configure(text="DOĞRULAMA")
            self.page_title.configure(text="Sayım Doğrulama")
            self._refresh_pallet_view()
        elif page == "summary":
            self._build_summary_content()
            self.summary_page.pack(fill="both", expand=True)
            self.page_eyebrow.configure(text="ÇIKTI")
            self.page_title.configure(text="Sayım Özeti")
        else:
            self.count_page.pack(fill="both", expand=True)
            self.page_eyebrow.configure(text="64 KASA MODU")
            self.page_title.configure(text="Kamera Kontrol Merkezi")
        self._sync_portrait_capture_button()

    # Dik ekranda araç çubuğu yükseklikleri: büyük TÜMÜNÜ OKU düğmesi
    # yalnız kamera/sevkiyat sayfalarında durur, diğerlerinde 68px yer açılır.
    PORTRAIT_TOOLBAR_TALL = 236
    PORTRAIT_TOOLBAR_SHORT = 168

    def _sync_portrait_capture_button(self):
        """Büyük çekim düğmesini yalnız anlamlı olduğu sayfalarda gösterir."""
        if not self.portrait_mode:
            return
        if self.active_page in ("count", "session"):
            if not self.capture_all_big.winfo_ismapped():
                self.capture_all_big.pack(side="top", fill="x", padx=4, pady=(2, 6))
            self.toolbar.configure(height=self.PORTRAIT_TOOLBAR_TALL)
        else:
            self.capture_all_big.pack_forget()
            self.toolbar.configure(height=self.PORTRAIT_TOOLBAR_SHORT)

    def _toggle_portrait_mode(self):
        self.portrait_mode = not self.portrait_mode
        self.toolbar_title.pack_forget()
        self.toolbar_actions.pack_forget()

        if self.portrait_mode:
            self._set_minsize(960, 820)
            self.sidebar.pack_forget()
            # 4 satır: başlık · yardımcı düğmeler · gezinme · TÜMÜNÜ OKU
            self.toolbar.configure(height=236)
            self.toolbar_title.pack(side="top", fill="x", padx=4, pady=(6, 0))
            self.toolbar_actions.pack(side="top", fill="x", padx=0, pady=(4, 2))
            self.portrait_nav.pack(side="top", fill="x", padx=1, pady=(2, 2))
            self.capture_group_h.pack_forget()
            self.capture_all_big.pack(side="top", fill="x", padx=4, pady=(2, 6))
            self.portrait_home_btn.pack(side="left", padx=4)
            self.portrait_btn.configure(text="▭ Yatay Ekran")
            self.page_container.pack_configure(padx=10, pady=(0, 8))
            self.camera_filter = 6
            self.camera_page_start = 0
        else:
            self._set_minsize(1380, 820)
            self.sidebar.pack(
                side="left", fill="y", before=self.main
            )
            self.toolbar.configure(height=84)
            self.toolbar_title.pack(side="left", pady=12)
            self.toolbar_actions.pack(side="right", pady=15)
            self.portrait_nav.pack_forget()
            self.capture_all_big.pack_forget()
            self.capture_group_h.pack(side="left", padx=4)
            self.portrait_home_btn.pack_forget()
            self.portrait_btn.configure(text="▯ Dik Ekran")
            self.page_container.pack_configure(padx=18, pady=(0, 16))
        # Metrik kartları her iki modda da TEK SATIRDA yan yana durur.
        for card in self.metric_cards:
            card.grid_forget()
        for column in range(3):
            self.metrics_frame.grid_columnconfigure(
                column, weight=1, uniform="metrics"
            )
        pad = 4 if self.portrait_mode else 5
        for column, card in enumerate(self.metric_cards):
            card.grid(
                row=0, column=column, columnspan=1, sticky="ew", padx=pad, pady=0
            )

        self._layout_review_cards()
        self._layout_session_cards()
        self._refresh_capture_buttons()
        self._sync_portrait_capture_button()
        self._apply_camera_filter()
        if self.active_page == "pallet":
            self._rebuild_pallet_cards()

    def _set_camera_filter(self, count):
        self.camera_filter = count
        self.camera_page_start = 0
        self._apply_camera_filter()

    def _navigate_cameras(self, direction):
        if self.camera_filter == 1:
            self.camera_page_start = (
                self.camera_page_start + direction
            ) % MAX_CAMERAS
        self._apply_camera_filter()

    def _open_single_camera(self, tile):
        if self.camera_filter != 6:
            return
        self.camera_filter = 1
        self.camera_page_start = tile.index
        self._apply_camera_filter()

    # Dik ekranda isimden yerleşim. Sütun: "Sol"→0, "Sağ"→1. Satır:
    # "Üst"→0, "Orta"→1, "Alt"→2. Altı kamera da tarafını+seviyesini
    # vermezse ya da iki kamera aynı hücreye düşerse None döner (kart sırası).
    _PORTRAIT_ROW = {"üst": 0, "ust": 0, "orta": 1, "alt": 2}

    def _portrait_named_positions(self):
        positions, kullanilan = {}, set()
        for index, tile in enumerate(self.tiles):
            ad = (tile.custom_name or "").lower()
            if "sol" in ad:
                column = 0
            elif "sağ" in ad or "sag" in ad:
                column = 1
            else:
                return None  # taraf bilgisi yok
            row = None
            for anahtar, deger in self._PORTRAIT_ROW.items():
                if anahtar in ad:
                    row = deger
                    break
            if row is None:
                return None  # seviye bilgisi yok
            if (row, column) in kullanilan:
                return None  # çakışma — güvenli tarafta kal
            kullanilan.add((row, column))
            positions[index] = (row, column)
        return positions

    def _apply_camera_filter(self):
        for tile in self.tiles:
            tile.grid_forget()

        if self.camera_filter == 1:
            indexes = [self.camera_page_start]
            self.tiles[indexes[0]].grid(
                row=0,
                column=0,
                columnspan=2 if self.portrait_mode else 3,
                sticky="nsew",
                padx=5,
                pady=3,
            )
            info = f"Kamera {indexes[0] + 1} / 6"
        else:
            indexes = list(range(MAX_CAMERAS))
            # Dik ekranda kameralar İSİMLERİNE göre dizilir: sol sütun / sağ
            # sütun, her sütunda Üst→Orta→Alt. Böylece palet iki yandan alt
            # alta görünür. İsimler bu düzene uymuyorsa kart sırasına düşülür.
            named = self._portrait_named_positions() if self.portrait_mode else None
            for index, tile in enumerate(self.tiles):
                if named is not None:
                    row, column = named[index]
                else:
                    columns = 2 if self.portrait_mode else 3
                    row, column = index // columns, index % columns
                tile.grid(
                    row=row,
                    column=column,
                    sticky="nsew",
                    padx=5,
                    pady=3,
                )
            info = "Tüm Kameralar 1—6"

        if self.portrait_mode:
            visible_columns = 2
            visible_rows = 1 if self.camera_filter == 1 else 3
        else:
            visible_columns = 3
            visible_rows = 2 if self.camera_filter == 6 else 1

        for column in range(3):
            self.tiles_frame.grid_columnconfigure(
                column,
                weight=1 if column < visible_columns else 0,
                uniform="tile_columns" if column < visible_columns else "",
            )
        for row in range(MAX_CAMERAS):
            self.tiles_frame.grid_rowconfigure(
                row,
                weight=1 if row < visible_rows else 0,
                uniform="tile_rows" if row < visible_rows else "",
            )

        for count, button in self.filter_buttons.items():
            selected = count == self.camera_filter
            button.configure(
                fg_color=theme.GREEN_DARK if selected else "transparent",
                text_color=theme.GREEN if selected else theme.TEXT_SOFT,
                border_width=1 if selected else 0,
                border_color=theme.BORDER_ACTIVE,
            )

        arrow_state = "disabled" if self.camera_filter == 6 else "normal"
        self.previous_camera_btn.configure(state=arrow_state)
        self.next_camera_btn.configure(state=arrow_state)
        connected = sum(1 for index in indexes if self.tiles[index].camera)
        self.group_info.configure(
            text=f"{info}  ·  {connected}/{len(indexes)} bağlı"
        )
        single = indexes[0] if self.camera_filter == 1 else None
        for tile in self.tiles:
            tile.set_interactive(tile.index == single)
        self._schedule_tile_resize()

    # ------------------------------------------------------------ model seçimi
    def _on_model_selected(self, label):
        self.model_menu.configure(state="disabled")
        threading.Thread(
            target=self._apply_model_worker, args=(label,), daemon=True
        ).start()

    def _apply_model_worker(self, label):
        """Seçilen YOLO modelini yükler; sonraki çekimler bu modelle yapılır."""
        path = self._model_paths.get(label)
        self.set_status("Model yükleniyor", theme.AMBER)
        self.log(f"YOLO modeli değiştiriliyor: {label}…")
        try:
            with self.processing_lock:
                if path:
                    os.environ["ODAI_YOLO_MODEL"] = path
                else:
                    os.environ.pop("ODAI_YOLO_MODEL", None)
                self.detector.model = None  # load_model yeni yolu çözsün
                self.detector.load_model(log_fn=self.log)
            self.set_status(f"Model hazır", theme.GREEN)
            self.root.after(0, lambda: self._show_toast(f"✓ Model aktif: {label}"))
        except Exception as exc:
            self.set_status("Model yüklenemedi", theme.RED)
            self.log(f"Model yükleme HATA: {exc}")
            self.log(traceback.format_exc())
            self.root.after(
                0,
                lambda: self._show_toast("⚠ Model yüklenemedi — log'a bakın", theme.RED),
            )
        finally:
            self.root.after(0, lambda: self.model_menu.configure(state="normal"))

    def _open_fullscreen_camera(self, tile):
        """Kamera görüntüsünü tam ekran açar (karta çift dokunuş).

        Kasa kutularını yakından incelemek için: görüntü pencereye sığacak
        şekilde büyütülür, boş alana ya da Kapat'a dokununca kapanır.
        """
        frame = tile.current_image()
        if frame is None:
            self._show_toast("Bu kamerada henüz görüntü yok", theme.AMBER)
            return
        self._close_serial_modal()
        self._modal_blocker = ctk.CTkFrame(
            self.main, fg_color=theme.OVERLAY_BG, corner_radius=0
        )
        self._modal_blocker.place(relx=0, rely=0, relwidth=1, relheight=1)
        self._modal_blocker.bind("<Button-1>", lambda _event: self._close_serial_modal())

        card = ctk.CTkFrame(self.main, fg_color=theme.OVERLAY_BG, corner_radius=0)
        card.place(relx=0.5, rely=0.5, anchor="center")
        self._modal_card = card

        header = ctk.CTkFrame(card, fg_color="transparent")
        header.pack(fill="x", padx=14, pady=(10, 4))
        result = self.results[tile.index] if tile.index < len(self.results) else None
        detail = ""
        if result:
            detail = (
                f"   ·   {result.get('toplam_kasa', 0)} tespit"
                f"   ·   {result.get('okunan_barkod', 0)} barkod"
            )
        ctk.CTkLabel(
            header,
            text=f"{self._camera_name(tile.index)}{detail}",
            anchor="w",
            text_color="#FFFFFF",
            font=ctk.CTkFont(theme.FONT, 15, "bold"),
        ).pack(side="left")
        ctk.CTkButton(
            header,
            text="✕  Kapat",
            width=120,
            height=40,
            corner_radius=10,
            fg_color="#FFFFFF",
            hover_color=theme.SURFACE_HOVER,
            text_color=theme.TEXT,
            font=ctk.CTkFont(theme.FONT, 12, "bold"),
            command=self._close_serial_modal,
        ).pack(side="right")

        # Başlık satırı (~56) + kart iç boşlukları (~40) + kenar payı düşülür,
        # yoksa büyük görüntü pencerenin altından taşıyor.
        max_width = max(320, self.main.winfo_width() - 60)
        max_height = max(240, self.main.winfo_height() - 170)
        height, width = frame.shape[:2]
        scale = min(max_width / width, max_height / height)
        display = cv2.resize(
            frame,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
        )
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        self._modal_photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        image_label = tk.Label(
            card, image=self._modal_photo, bg=theme.OVERLAY_BG, bd=0, cursor="hand2"
        )
        image_label.pack(padx=14, pady=(0, 14))
        image_label.bind("<Button-1>", lambda _event: self._close_serial_modal())

    def _open_review_page(self, _tile=None):
        # Kamera kartındaki OKUNAMAYAN rozeti: doğrulama ekranı zaten yalnız
        # okunamayan kasaları listeler.
        self._show_page("pallet")

    def _on_image_crate_click(self, tile, image_x, image_y):
        """Tekli görünümde görüntü üzerindeki kasa kutusuna tıklamayı işler."""
        result = self.results[tile.index] if tile.index < len(self.results) else None
        if not result:
            return
        hit = None
        hit_area = None
        for kasa in result.get("kasalar") or []:
            x1, y1, x2, y2 = kasa["bbox"]
            if x1 <= image_x <= x2 and y1 <= image_y <= y2:
                area = (x2 - x1) * (y2 - y1)
                if hit_area is None or area < hit_area:
                    hit, hit_area = kasa, area
        if hit is None:
            return
        key = (tile.index, hit.get("kasa_no"))
        for index, crate in enumerate(self.pallet_crates):
            if crate.get("source") != "real":
                continue
            members = crate.get("members") or [
                (crate.get("camera_index"), crate.get("kasa_no"))
            ]
            if key in members:
                self._open_serial_modal(index)
                return
        self._show_toast(
            "Bu kasanın kaydı bulunamadı — sayımı yenileyin", theme.AMBER
        )

    def set_status(self, text, color=theme.TEXT_MUTED):
        def update():
            self.status_label.configure(text=f"●  {text}", text_color=color)
        self.root.after(0, update)

    def log(self, message):
        print(f"[{datetime.now().strftime('%H:%M:%S')}]  {message}")

    # --------------------------------------------------------------- bağlantı
    def _on_connect(self):
        self.connect_btn.configure(state="disabled", text="Taranıyor…")
        threading.Thread(target=self._connect_worker, daemon=True).start()

    # ------------------------------------------------------ kamera adları
    def _load_camera_names(self):
        """kamera_isimleri.json → {seri: {"sira": int, "isim": str}}"""
        try:
            with open(CAMERA_NAMES_PATH, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            return {}
        return {
            serial: entry
            for serial, entry in data.items()
            if not serial.startswith("_") and isinstance(entry, dict)
        }

    def _save_camera_names(self, names):
        data = {
            "_aciklama": (
                "Kamera sırası ve isimleri seri numarasına bağlıdır; sira "
                "küçükten büyüğe kartlara dizilir. İsim vermek için uygulamada "
                "kamera başlığına çift dokunun."
            )
        }
        data.update(names)
        try:
            with open(CAMERA_NAMES_PATH, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=2)
        except OSError as exc:
            self.log(f"Kamera isim dosyası yazılamadı: {exc}")

    def _camera_name(self, index):
        """Kart başlığında ve tam ekranda kullanılan ad."""
        if index < len(self.tiles) and self.tiles[index].custom_name:
            return self.tiles[index].custom_name
        return f"Kamera {index + 1}"

    def _open_camera_rename(self, tile):
        """Kamera başlığına çift dokununca: konum adı ver/değiştir."""
        if tile.camera is None or not getattr(tile.camera, "serial", ""):
            self._show_toast(
                "⚠ İsim seri numarasına yazılır — önce kameraları bağlayın",
                theme.AMBER,
            )
            return
        serial = tile.camera.serial
        self._close_serial_modal()
        self._modal_blocker = ctk.CTkFrame(
            self.main, fg_color=theme.IMAGE_BG, corner_radius=0
        )
        self._modal_blocker.place(relx=0, rely=0, relwidth=1, relheight=1)
        self._modal_blocker.bind("<Button-1>", lambda _event: self._close_serial_modal())
        card = ctk.CTkFrame(
            self.main,
            fg_color=theme.SURFACE,
            corner_radius=18,
            border_width=2,
            border_color=theme.GREEN,
        )
        card.place(relx=0.5, rely=0.5, anchor="center")
        self._modal_card = card
        ctk.CTkLabel(
            card,
            text=f"KAMERA {tile.index + 1} · {serial}",
            text_color=theme.GREEN,
            font=ctk.CTkFont(theme.FONT, 12, "bold"),
        ).pack(padx=36, pady=(22, 2))
        ctk.CTkLabel(
            card,
            text="Bu kameraya konum adı verin — isim seri numarasına\n"
            "kaydedilir, kamera hangi sırayla bulunursa bulunsun korunur.",
            justify="center",
            text_color=theme.TEXT_SOFT,
            font=ctk.CTkFont(theme.FONT, 11),
        ).pack(padx=36, pady=(0, 12))
        entry = ctk.CTkEntry(
            card,
            width=320,
            height=48,
            placeholder_text="örn. Sağ Üst",
            fg_color=theme.BG_RAISED,
            border_color=theme.BORDER,
            font=ctk.CTkFont(theme.FONT, 16, "bold"),
            justify="center",
        )
        entry.pack(padx=28, pady=(0, 10))
        if tile.custom_name:
            entry.insert(0, tile.custom_name)
        # Dokunmatik için hazır konumlar
        grid = ctk.CTkFrame(card, fg_color="transparent")
        grid.pack(padx=28, pady=(0, 12))
        for pos, text in enumerate(
            ("Sol Üst", "Orta Üst", "Sağ Üst", "Sol Alt", "Orta Alt", "Sağ Alt")
        ):
            ctk.CTkButton(
                grid,
                text=text,
                width=104,
                height=40,
                corner_radius=9,
                fg_color=theme.SURFACE_LIGHT,
                hover_color=theme.BORDER_ACTIVE,
                text_color=theme.TEXT,
                font=ctk.CTkFont(theme.FONT, 11, "bold"),
                command=lambda t=text: (entry.delete(0, "end"), entry.insert(0, t)),
            ).grid(row=pos // 3, column=pos % 3, padx=3, pady=3)

        def kaydet():
            isim = entry.get().strip()
            names = self._load_camera_names()
            entry_cfg = names.setdefault(serial, {"sira": tile.index + 1})
            entry_cfg["isim"] = isim
            self._save_camera_names(names)
            tile.set_custom_name(isim)
            # İsim değişince dik ekranda kart hemen doğru konuma taşınsın
            # (Sol/Sağ sütun, Üst→Orta→Alt); yeniden bağlanmaya gerek kalmasın.
            self._apply_camera_filter()
            self._close_serial_modal()
            self._show_toast(
                f"✓ {serial} → {isim or f'Kamera {tile.index + 1}'}"
            )
            self.log(f"Kamera adı: {serial} = '{isim}' (kart {tile.index + 1})")

        entry.bind("<Return>", lambda _event: kaydet())
        satir = ctk.CTkFrame(card, fg_color="transparent")
        satir.pack(fill="x", padx=28, pady=(0, 20))
        ctk.CTkButton(
            satir,
            text="✕ Vazgeç",
            width=110,
            height=46,
            corner_radius=11,
            fg_color=theme.BG_RAISED,
            hover_color=theme.SURFACE_HOVER,
            text_color=theme.TEXT_MUTED,
            border_width=1,
            border_color=theme.BORDER,
            font=ctk.CTkFont(theme.FONT, 12, "bold"),
            command=self._close_serial_modal,
        ).pack(side="left")
        ctk.CTkButton(
            satir,
            text="KAYDET",
            width=190,
            height=46,
            corner_radius=11,
            fg_color=theme.GREEN,
            hover_color=theme.GREEN_HOVER,
            text_color="#FFFFFF",
            font=ctk.CTkFont(theme.FONT, 13, "bold"),
            command=kaydet,
        ).pack(side="right")
        entry.focus_set()

    def _connect_worker(self):
        try:
            self.set_status("Kameralar aranıyor", theme.AMBER)
            self.log("MVS SDK başlatılıyor…")
            self.manager.initialize_sdk()
            infos = self.manager.enum_devices()
            self.log(f"{len(infos)} kamera bulundu.")
            gige = [info for info in infos if info.get("ip")]
            adaylar = gige or infos
            # SDK'nın tarama sırası ağdan cevap gelme sırasıdır ve açılıştan
            # açılışa DEĞİŞEBİLİR. Kartlara dağıtım seri numarasına yazılı
            # "sira" değerine göre yapılır; dosyada olmayan kameralar sona
            # eklenir ve bir sonraki açılış için kaydedilir.
            names = self._load_camera_names()
            adaylar.sort(
                key=lambda info: (
                    names.get(info.get("serial") or "", {}).get("sira", 999),
                    info["index"],
                )
            )
            selected = adaylar[:MAX_CAMERAS]
            if not selected:
                raise RuntimeError("Bağlanılabilir kamera bulunamadı.")
            degisti = False
            for slot, info in enumerate(selected, 1):
                serial = info.get("serial") or ""
                if not serial:
                    continue
                kayit = names.setdefault(serial, {})
                if kayit.get("sira") != slot:
                    kayit["sira"] = slot
                    degisti = True
                kayit.setdefault("isim", "")
                isim = kayit["isim"]
                self.log(
                    f"K{slot} ← {info.get('ip', '?')} · {serial}"
                    + (f" · {isim}" if isim else "")
                )
            if degisti:
                self._save_camera_names(names)
            self.log(f"{len(selected)} kameraya bağlanılıyor…")
            self.manager.connect_selected(selected)
            self.cameras = list(self.manager.cameras)

            for index, camera in enumerate(self.cameras):
                ok, err = camera.set_exposure_us(DEFAULT_EXPOSURE_US)
                if ok:
                    self.log(
                        f"Kamera {index + 1}: pozlama {DEFAULT_EXPOSURE_US} µs (sabit)."
                    )
                else:
                    self.log(f"Kamera {index + 1}: pozlama ayarlanamadı ({err}).")

            def apply():
                for index, tile in enumerate(self.tiles):
                    if index < len(self.cameras):
                        tile.attach_camera(self.cameras[index])
                        serial = getattr(self.cameras[index], "serial", "") or ""
                        tile.set_custom_name(
                            names.get(serial, {}).get("isim", "")
                        )
                    else:
                        tile.detach_camera()
                self._update_capture_enabled()
                self.connect_btn.configure(state="normal", text="⌁ Kameraları Tara")
                self._apply_camera_filter()
            self.root.after(0, apply)
            self.set_status(f"{len(self.cameras)} kamera bağlı", theme.GREEN)
            self.log("Kamera bağlantıları hazır.")

            self.log("YOLO modeli hazırlanıyor…")
            self.detector.load_model(log_fn=self.log)
            self.log("YOLO modeli hazır.")
        except Exception as exc:
            self.log(f"HATA: {exc}")
            self.log(traceback.format_exc())
            self.set_status("Bağlantı hatası", theme.RED)
            self.root.after(
                0,
                lambda: self.connect_btn.configure(state="normal", text="⌁ Kameraları Tara"),
            )

    # -------------------------------------------------------------- önizleme
    def _toggle_preview(self):
        if self.preview_switch.get():
            if not self.cameras:
                self.preview_switch.deselect()
                self.log("Canlı önizleme için önce kameraları bağlayın.")
                return
            # Canlı görüntüye geçerken eski çekimin okunamayan uyarıları
            # anlamsız — yanıp sönen rozetleri kaldır.
            for tile in self.tiles:
                tile.set_unread_badge(0)
            self.preview_running = True
            self.preview_thread = threading.Thread(target=self._preview_loop, daemon=True)
            self.preview_thread.start()
            self.log("Canlı önizleme açıldı.")
        else:
            self.preview_running = False
            # Önizleme kapanınca son sayımın uyarıları (hâlâ geçerliyse) geri gelir.
            self._sync_unread_metrics()
            self.log("Canlı önizleme kapatıldı.")

    def _preview_loop(self):
        interval = 1.0 / PREVIEW_FPS
        while self.preview_running:
            started = time.time()
            for index, camera in enumerate(list(self.cameras)):
                if not self.preview_running:
                    break
                tile = self.tiles[index]
                if str(tile.capture_btn.cget("state")) == "disabled":
                    continue
                try:
                    frame = camera.capture(timeout_ms=700)
                    self.root.after(0, lambda target=tile, image=frame: target.show_image(image))
                except Exception:
                    pass
            elapsed = time.time() - started
            if elapsed < interval:
                time.sleep(interval - elapsed)

    # ----------------------------------------------------------------- çekim
    def _on_single_capture(self, tile):
        if not tile.camera:
            return
        if self._block_without_barcode():
            return
        # Yeni çekim başlarken bu kameranın eski uyarısı kalksın.
        tile.set_unread_badge(0)
        tile.set_busy(True)
        threading.Thread(
            target=self._single_capture_worker,
            args=(tile.index, tile.camera),
            daemon=True,
        ).start()

    def _single_capture_worker(self, index, camera):
        self._capture_camera(index, camera)
        self._aggregate()

    def _on_capture_all(self, replace_last=False):
        """Sayımı başlatır.

        replace_last=True: sonuç, sürece EKLENMEZ; en son eklenen paletin
        üstüne yazılır (yanlış sayılan paleti düzeltmek için).
        """
        if not self.cameras:
            self.log("Çekim için önce kameraları bağlayın.")
            return
        if self._block_without_barcode():
            return
        if replace_last:
            session = self.session
            if not session or not session["pallets"]:
                self._show_toast("⚠ Yeniden sayılacak palet yok", theme.AMBER)
                return
            self._replace_target = len(session["pallets"]) - 1
            hedef = session["pallets"][self._replace_target]["no"]
            self.log(f"Palet {hedef} yeniden sayılıyor (üstüne yazılacak).")
        self._capture_cancel.clear()
        if self.session is None and not self._no_session_warned:
            # Süreç yokken de sayım yapılabilir (kamera/pozlama kontrolü),
            # ama sonuç hiçbir sevkiyata yazılmaz; bir kez hatırlatılır.
            self._no_session_warned = True
            self._show_toast(
                "⚠ Deneme modu — sevkiyat açık değil, sayım kaydedilmez",
                theme.AMBER,
            )
        self._set_capture_all_state("disabled", busy=True)
        # Yeni sayım: önceki çekimin tüm sonuç ve uyarıları temizlenir;
        # yanıp sönen "OKUNAMAYAN" rozetleri yeni sonuca kadar görünmez.
        self._ocr_generation += 1
        self._counting = True
        self._ocr_progress = None
        self.results = [None] * MAX_CAMERAS
        self.unread_records = []
        self.camera_images = []
        self.pallet_crates = []
        self.last_aggregate = None
        self._refresh_unread_list()
        for tile in self.tiles:
            tile.last_result = None
            tile.set_unread_badge(0)
            tile.set_count_badge(None)
            if tile.camera:
                tile.set_busy(True)
        self.metric_total.set(0)
        self.metric_barcode.set(0)
        self.metric_unread.set(0)
        self._refresh_ocr_list()
        if self.active_page == "pallet":
            self._refresh_pallet_view()
        threading.Thread(target=self._capture_all_worker, daemon=True).start()

    def _capture_all_worker_akis(self):
        """ÇEKERKEN OKU: her kameranın fotoğrafı gelir gelmez barkod okuması
        başlar (diğer kameralar hâlâ çekilirken). Kamera okuması bitince o
        kameranın sonuç/kayıt işi ayrı havuzda hemen yapılır. Okuma yöntemi
        (kamera kamera / havuz / tam kare) ve okuyucu sayısı diğer kutulardan."""
        ayar = self.hizli
        self.set_status("Sayım yapılıyor (akış)", theme.AMBER)
        self.log(f"[HIZLI] Akışlı sayım başladı · {ayar.ozet()}")
        t_start = self._t_run
        self._tl.rec = None
        fast_save = ayar["hizli_kayit"]
        kalemler = [None] * len(self.cameras)
        fin_pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="sonuc")
        futures = {}

        def finalize(item):
            t_f = time.perf_counter()
            result = self.hizli_barkod.sonuc_kur(item, item["save_dir"], fast_save, self._io_pool)
            self._filter_background_crates(item["index"], result, frame_height=item["frame"].shape[0])
            if fast_save:
                self._save_review_crops_hizli(item["frame"], result, item["save_dir"], item["timestamp"])
            else:
                self._save_review_crops(item["frame"], result, item["save_dir"], item["timestamp"])
            if item["raw_future"] is not None:
                item["raw_future"].result()
            item["result"] = result
            rec = self._cam_timing[item["index"]]
            rec["barkod"] = item.get("barkod_sn", 0.0)
            rec["barkod bitti"] = item.get("barkod_bitti", 0.0)
            rec["kayıt+diğer"] = time.perf_counter() - t_f
            rec["bitiş"] = time.perf_counter() - t_start
            rec["kasa"] = result["toplam_kasa"]

        def on_done(item):
            futures[item["index"]] = fin_pool.submit(finalize, item)

        try:
            akis = BarkodAkisi(self.hizli_barkod, ayar, on_done, t_start)
        except Exception as exc:
            self.log(f"[HIZLI] Akış başlatılamadı ({exc}); normal hızlı yola geçiliyor.")
            fin_pool.shutdown(wait=False)
            return self._capture_all_worker_hizli()

        def capture(index, camera):
            tile = self.tiles[index]
            if self._capture_cancel.is_set():
                self.root.after(0, lambda: tile.set_busy(False))
                return
            rec = {"kamera": index + 1, "başla": time.perf_counter() - t_start}
            self._cam_timing[index] = rec
            try:
                t_req = time.perf_counter()
                with self._capture_slots:
                    t_got = time.perf_counter()
                    frame = camera.capture(timeout_ms=5000, fresh=True)
                t_cap = time.perf_counter()
                rec.update({"sıra bekleme": t_got - t_req, "çekim": t_cap - t_got,
                            "çekim bitti": t_cap - t_start})
                self.root.after(0, lambda: tile.show_image(frame))
                folder = str(camera.serial or f"kamera_{index + 1}").replace(" ", "_")
                save_dir = CAPTURE_DIR / folder
                save_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                raw_path = save_dir / f"raw_{timestamp}.jpg"
                item = {
                    "index": index, "frame": frame, "save_dir": save_dir,
                    "timestamp": timestamp, "raw_path": str(raw_path),
                    "raw_future": None, "tile": tile, "crates": [],
                }
                akis.kare_geldi(item)            # tam karede okuma BURADA başlar
                t = time.perf_counter()
                if fast_save:
                    item["raw_future"] = self._io_pool.submit(cv2.imwrite, str(raw_path), frame)
                else:
                    cv2.imwrite(str(raw_path), frame)
                rec["ham kayıt"] = time.perf_counter() - t
                t = time.perf_counter()
                with self.processing_lock:
                    t_lock = time.perf_counter()
                    boxes = self.hizli_barkod.tespit(frame)
                rec["yolo bekleme"] = t_lock - t
                rec["yolo"] = time.perf_counter() - t_lock
                rec["yolo bitti"] = time.perf_counter() - t_start
                item["crates"] = [{"kasa_no": i + 1, "bbox": b, "barkodlar": [], "barkod_okundu": False}
                                  for i, b in enumerate(boxes)]
                kalemler[index] = item
                akis.kutular_hazir(item)         # kesit / kamera okuması BURADA başlar
            except Exception as exc:
                self.log(f"Kamera {index + 1} HATA: {exc}")
                self.root.after(0, lambda: tile.set_busy(False))

        threads = [threading.Thread(target=capture, args=(i, c), daemon=True)
                   for i, c in enumerate(self.cameras)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        akis.bitir()
        items = [k for k in kalemler if k]
        for item in items:   # akışta sonuçlanmamış kamera kaldıysa (hata) yine de kur
            if item["index"] not in futures:
                for c in item["crates"]:
                    c.setdefault("proc", None)
                    c["barkod_okundu"] = bool(c.get("barkodlar"))
                futures[item["index"]] = fin_pool.submit(finalize, item)
        for fut in futures.values():
            try:
                fut.result()
            except Exception as exc:
                self.log(f"[HIZLI] Sonuç HATA: {exc}")
        fin_pool.shutdown(wait=False)

        if self._capture_cancel.is_set():
            self.root.after(0, self._finish_cancelled_capture)
            return
        for item in items:
            if "result" not in item:
                continue
            index, tile, result = item["index"], item["tile"], item["result"]
            self.results[index] = result
            tile.last_result = result
            self.root.after(0, lambda tl=tile, r=result: self._apply_camera_result(tl, r))
            self._register_unread(index, result, item["raw_path"])
            self.log(f"Kamera {index + 1}: {result['toplam_kasa']} kasa, "
                     f"{result['okunan_barkod']} barkod, {result['okunamayan_barkod']} okunamayan")
            self.root.after(0, lambda tl=tile: tl.set_busy(False))

        health = self._barcode_health_after_capture()
        if health is not None:
            self.root.after(0, lambda: self._abort_count_barcode(health))
            return

        cams = list(self._cam_timing.values())

        def last(key):
            return max((c.get(key, 0.0) for c in cams), default=0.0)

        m_foto, m_yolo, m_bark, m_end = last("çekim bitti"), last("yolo bitti"), last("barkod bitti"), last("bitiş")
        t = time.perf_counter()
        aggregate = self._aggregate()
        stages = [
            ("Fotoğraf çekimi (son kamera)", m_foto),
            ("YOLO (son çekimden sonra)", max(0.0, m_yolo - m_foto)),
            ("Barkod (son YOLO'dan sonra)", max(0.0, m_bark - m_yolo)),
            ("Kayıt (son barkoddan sonra)", max(0.0, m_end - max(m_bark, m_yolo))),
            ("Tekilleştirme", time.perf_counter() - t),
        ]
        self._hizli_sure_yaz(
            stages,
            aggregate["unique_total"] if aggregate else 0,
            aggregate["unique_barcodes"] if aggregate else 0,
            sum(r["okunamayan_barkod"] for r in self.results if r),
        )

        def finish_count():
            self._counting = False
            self._set_capture_all_state("normal")
            if aggregate is not None:
                self._show_page("pallet")
            elif self.active_page == "pallet":
                self._refresh_pallet_view()

        self.root.after(0, finish_count)
        self.set_status(f"Sayım tamamlandı · {time.perf_counter() - t_start:.1f} sn", theme.GREEN)
        self.log("Tüm kamera işlemleri tamamlandı (akış).")

    def _capture_all_worker(self):
        self._t_run = time.perf_counter()
        self._cam_timing = {}
        self._ocr_t0 = None
        self._install_timers()
        if (self.hizli.any_on and self.hizli_barkod is not None
                and getattr(self.detector, "_reader", None) is not None):
            if self.hizli["akis"]:
                return self._capture_all_worker_akis()
            return self._capture_all_worker_hizli()
        # ---- Bugünkü TTO yolu (birebir); yalnız süre ölçümü eklendi.
        t_start = self._t_run
        self.set_status("Sayım yapılıyor", theme.AMBER)
        self.log("Tüm kameralardan eşzamanlı çekim başladı.")
        threads = []
        for index, camera in enumerate(self.cameras):
            thread = threading.Thread(
                target=self._capture_camera,
                args=(index, camera),
                daemon=True,
            )
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()

        if self._capture_cancel.is_set():
            self.root.after(0, self._finish_cancelled_capture)
            return

        # Dongle çekim sırasında çıkarılmış olabilir: okuyucu düştüyse ya da
        # kasa bulunup HİÇ barkod okunmadıysa tekilleştirme güvenilmez.
        # Yanlış (şişmiş) sayı üretmektense hata verilir; OCR'a hiç girilmez.
        health = self._barcode_health_after_capture()
        if health is not None:
            self.root.after(0, lambda: self._abort_count_barcode(health))
            return

        t_islem = time.perf_counter() - t_start
        cams = list(self._cam_timing.values())
        t_foto = max((c.get("çekim bitti", 0) for c in cams), default=0.0)
        t = time.perf_counter()
        aggregate = self._aggregate()
        self._hizli_sure_yaz(
            [("Fotoğraf çekimi (6 kamera)", t_foto),
             ("İşleme (YOLO+barkod+kayıt, sırayla)", t_islem - t_foto),
             ("Tekilleştirme", time.perf_counter() - t)],
            aggregate["unique_total"] if aggregate else 0,
            aggregate["unique_barcodes"] if aggregate else 0,
            sum(r["okunamayan_barkod"] for r in self.results if r),
        )

        def finish_count():
            self._counting = False
            self._set_capture_all_state("normal")
            if aggregate is not None:
                # Sayım bitti: operatör "incele" demeden doğrudan doğrulama
                # ekranına alınır. OCR arka planda sürerken şerit ilerlemeyi
                # gösterir; çözülen kasalar listeden kendiliğinden düşer.
                self._show_page("pallet")
            elif self.active_page == "pallet":
                self._refresh_pallet_view()

        self.root.after(0, finish_count)
        self.set_status("Sayım tamamlandı", theme.GREEN)
        self.log("Tüm kamera işlemleri tamamlandı.")

    def _finish_cancelled_capture(self):
        """İptal edilen sayımı temizler: sonuç yok, toplama yok, OCR yok."""
        self._counting = False
        self._ocr_generation += 1
        self._ocr_progress = None
        self.results = [None] * MAX_CAMERAS
        self.pallet_crates = []
        self.last_aggregate = None
        self.unread_records = []
        self._replace_target = None
        for tile in self.tiles:
            tile.last_result = None
            tile.set_busy(False)
            tile.set_unread_badge(0)
            tile.set_count_badge(None)
        self.metric_total.set(0)
        self.metric_barcode.set(0)
        self.metric_unread.set(0)
        self._refresh_unread_list()
        self._refresh_ocr_list()
        self._set_capture_all_state("normal")
        self.set_status("Sayım iptal edildi", theme.AMBER)
        self._show_toast("⏹ Sayım iptal edildi — sonuç kaydedilmedi", theme.AMBER)
        self._show_page("count")

    def _barcode_health_after_capture(self):
        """Çekim sonrası barkod motoru sağlıklı mı? Sorun varsa mesaj döner."""
        if not AREMAK_AVAILABLE:
            return AREMAK_IMPORT_ERROR or "Aremak barkod SDK'sı yüklü değil"
        if getattr(self.detector, "_reader", None) is None:
            return (
                getattr(self.detector, "_reader_init_error", None)
                or "Aremak okuyucu düştü — USB dongle çıkarılmış olabilir"
            )
        crates = sum((r or {}).get("toplam_kasa", 0) for r in self.results)
        barcodes = sum((r or {}).get("okunan_barkod", 0) for r in self.results)
        if crates and not barcodes:
            return (
                f"{crates} kasa bulundu ama HİÇ barkod okunamadı — okuyucu "
                "çalışmıyor (dongle/lisans). Barkodsuz sayımda kameralar aynı "
                "kasayı ayrı ayrı sayar, toplam yanlış çıkar."
            )
        return None

    def _abort_count_barcode(self, message):
        """Barkod motoru bozukken sayımı iptal eder: sonuç yok, OCR yok."""
        self._counting = False
        self._ocr_generation += 1  # bekleyen OCR varsa geçersiz kıl
        self._ocr_progress = None
        self.results = [None] * MAX_CAMERAS
        self.pallet_crates = []
        self.last_aggregate = None
        self.unread_records = []
        self._refresh_unread_list()
        self._refresh_ocr_list()
        self.metric_total.set(0)
        self.metric_barcode.set(0)
        self.metric_unread.set(0)
        for tile in self.tiles:
            tile.last_result = None
            tile.set_unread_badge(0)
            tile.set_count_badge(None)
        self._set_capture_all_state("normal")
        self._apply_barcode_state(False, message)
        self.log("[HATA] Sayım iptal edildi — barkod motoru çalışmıyor.")
        self._open_confirm_modal(
            "⛔  BARKOD SDK HATASI — SAYIM İPTAL",
            f"{message}\n\nBarkod okunmadan tekilleştirme çalışmaz: kameralar "
            "aynı kasayı ayrı ayrı sayar ve toplam gerçekte olduğundan yüksek "
            "çıkar. Bu yüzden sayım kaydedilmedi ve OCR çalıştırılmadı.\n\n"
            "USB dongle'ı takıp TEKRAR DENE ile sürdürebilirsiniz.",
            "↻  TEKRAR DENE",
            lambda: self._start_barcode_probe(initial=False),
            theme.RED,
        )

    def _capture_camera(self, index, camera):
        tile = self.tiles[index]
        if self._capture_cancel.is_set():
            self.root.after(0, lambda: tile.set_busy(False))
            return
        t_run = self._t_run or time.perf_counter()
        rec = {"kamera": index + 1, "başla": time.perf_counter() - t_run}
        self._cam_timing[index] = rec
        self._tl.rec = rec
        try:
            self.log(f"Kamera {index + 1}: kare yakalanıyor…")
            # fresh=True: SDK tamponundaki eski kareler atılır; sayım her zaman
            # ŞU ANKİ görüntü üzerinden yapılır (eski çekim sayılmaz).
            # Semafor: tetikleme + aktarım aynı anda en fazla
            # MAX_CONCURRENT_CAPTURES kamerada çalışır (switch taşmasın).
            t_req = time.perf_counter()
            with self._capture_slots:
                t_got = time.perf_counter()
                frame = camera.capture(timeout_ms=5000, fresh=True)
            t_cap = time.perf_counter()
            rec.update({"sıra bekleme": t_got - t_req, "çekim": t_cap - t_got,
                        "çekim bitti": t_cap - t_run})
            self.root.after(0, lambda: tile.show_image(frame))
            result, raw_path = self._process_frame(index, frame, camera.serial or f"kamera_{index + 1}")
            rec["bitiş"] = time.perf_counter() - t_run
            rec["kasa"] = result.get("toplam_kasa", 0)
            self.results[index] = result
            tile.last_result = result
            self.root.after(
                0,
                lambda: self._apply_camera_result(tile, result),
            )
            self._register_unread(index, result, raw_path)
            self.log(
                f"Kamera {index + 1}: {result['toplam_kasa']} kasa, "
                f"{result['okunan_barkod']} barkod, "
                f"{result['okunamayan_barkod']} okunamayan."
            )
        except Exception as exc:
            self.log(f"Kamera {index + 1} HATA: {exc}")
        finally:
            self.root.after(0, lambda: tile.set_busy(False))

    def _process_frame(self, index, frame, folder_name):
        save_dir = CAPTURE_DIR / str(folder_name).replace(" ", "_")
        save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        raw_path = save_dir / f"raw_{timestamp}.jpg"
        rec = getattr(self._tl, "rec", None)
        t = time.perf_counter()
        cv2.imwrite(str(raw_path), frame)
        t_raw = time.perf_counter()
        with self.processing_lock:
            t_lock = time.perf_counter()
            self._install_timers()   # model değişmiş olabilir
            result = self.detector.process_image(
                frame,
                str(save_dir),
                log_fn=lambda message: self.log(f"K{index + 1} · {message}"),
            )
        t_proc = time.perf_counter()
        self._filter_background_crates(index, result, frame_height=frame.shape[0])
        self._save_review_crops(frame, result, save_dir, timestamp)
        if rec is not None:
            rec["ham kayıt"] = t_raw - t
            rec["yolo bekleme"] = t_lock - t_raw
            # process_image içi: YOLO + barkod sarmalayıcıdan; kalanı kesit/işaretli kayıt
            rest = (t_proc - t_lock) - rec.get("yolo", 0.0) - rec.get("barkod", 0.0)
            rec["kayıt+diğer"] = rest + (time.perf_counter() - t_proc)
        return result, str(raw_path)

    # Arka plandaki (aynı paletin arka sırası ya da komşu palet) kasaları
    # sayımdan ayıklamak için iki filtre birlikte çalışır:
    #  1) Genişlik: ort. genişliğin %85'inden dar kutular (yandan yarım
    #     görünen arka kasalar) elenir — ODAI'deki filtre.
    #  2) Sütun: kutular x-merkezine göre kümelenir; palet düzeninde her
    #     kamera 1-2 dolu sütun görür. Az üyeli kümeler (kenardan sızan
    #     arka sıra) elenir. 4x2 dizilimde arka sıra tam genişlikte bile
    #     görünse kendi sütununda az kutu olduğundan yakalanır.
    WIDTH_FILTER_RATIO = 0.85
    WIDTH_FILTER_MIN_CRATES = 6
    COLUMN_GAP_RATIO = 0.6      # yeni sütun sayılması için gereken cx boşluğu
    COLUMN_MIN_FRACTION = 0.15  # geçerli sütun için asgari tespit oranı
    COLUMN_MIN_CRATES = 4       # geçerli sütun için asgari kutu sayısı

    # Yükseklik (y) aykırı değer filtresi. Paletteki kasalar aynı ürün olduğu
    # için bir kutunun yüksekliği KOMŞULARININKİNE çok yakındır. Komşularından
    # ±%20'den fazla sapan kutu:
    #   · belirgin KISA  → arka sıradan sızmış / yarım görünen, kasa değil,
    #   · belirgin UZUN  → YOLO'nun iki kasayı tek kutuda birleştirmesi.
    #
    # Referans KÜRESEL medyan DEĞİL, aynı sütunda en yakın komşuların
    # medyanıdır. Sebep ölçüldü: perspektif yüzünden aynı paletin kasaları
    # kare boyunca 101–209px arasında değişiyor; küresel ±%20 eşiği 313
    # kutunun 34'ünü (gerçek kasaları) eliyordu. Komşu penceresi bu kademeli
    # kaymayı soğurur, yalnız tek başına sırıtan kutuyu yakalar (7 kutu).
    # Ölçülen gerçek davranış (6 kamera, 313 kutu):
    #   · Perspektif yüzünden boy sütun boyunca kademeli değişiyor
    #     (K4'te 124→252px). Küresel medyan eşiği 34 GERÇEK kasayı eliyordu.
    #   · Komşu MEDYANI da kenarlarda haksız: sütunun en üst/en alt kutusunun
    #     bütün komşuları tek yönde kaldığı için referans kayıyor
    #     (K1 #0: doğrusal tahminden -%2 sapıyor ama komşu medyanından -%13).
    # Bu yüzden referans, komşulara uydurulan DOĞRU üzerinde tahmin edilen
    # boydur; kenar kutularda eğilim ileri taşınır (ekstrapolasyon). Eğim
    # Theil-Sen (ikili eğimlerin medyanı) ile bulunur: penceredeki tek bozuk
    # komşu referansı bozamaz.
    # Eşikler ölçülerek seçildi (6 kamera, 295 kutu). Sapma dağılımı ASİMETRİK:
    #   kısa taraf : normal kasalar -%12.6'ya kadar iniyor, aykırılar -%25.7'den
    #                başlıyor  → -%20 tam boşluğun ortası
    #   uzun taraf : normal kasalar +%11.7'ye kadar çıkıyor, aykırılar
    #                +%53.2'den başlıyor → +%35 güvenli
    # Uzun tarafın bol bırakılması bilinçli: arka plandaki/yarım kasa KISA olur,
    # uzun sapma ise ancak YOLO iki kasayı birleştirince oluşur ve o çok büyük
    # (+%50 üstü) çıkar. Dar tutulursa perspektifte uzayan gerçek kasalar
    # haksız eleniyordu.
    HEIGHT_FILTER_SHORT_TOLERANCE = 0.20
    HEIGHT_FILTER_TALL_TOLERANCE = 0.35
    # Kare kenarına DEĞEN kutu (üstten/alttan kesilmiş) yarım görünür; arka
    # planda kalan kasalar çoğunlukla böyle. Bunlar için daha sıkı eşik
    # kullanılır — ama yalnız KISA tarafta: tam boy görünen, sadece kenarda
    # biten gerçek kasa elenmesin.
    HEIGHT_FILTER_EDGE_TOLERANCE = 0.08
    HEIGHT_FILTER_EDGE_MARGIN = 4  # px: kenara bu kadar yakınsa "değiyor"
    HEIGHT_FILTER_MIN_CRATES = 6
    HEIGHT_FILTER_WINDOW = 8       # karşılaştırmaya giren komşu sayısı
    HEIGHT_FILTER_MIN_COLUMN = 6   # bu kadar kutusu olmayan sütunda eleme yok
    # Bu orandan fazlasını eleyecekse referansın kendisi şüphelidir; dokunma.
    HEIGHT_FILTER_MAX_DROP = 0.30

    @staticmethod
    def _robust_line(xs, ys):
        """Theil-Sen doğrusu: eğim = ikili eğimlerin medyanı. (eğim, sabit)

        En küçük kareler penceredeki tek aykırı komşudan etkilenir; ikili
        eğimlerin medyanı etkilenmez.
        """
        slopes = []
        for i in range(len(xs)):
            for j in range(i + 1, len(xs)):
                dx = xs[j] - xs[i]
                if abs(dx) > 1e-6:
                    slopes.append((ys[j] - ys[i]) / dx)
        if not slopes:
            return 0.0, (sum(ys) / len(ys) if ys else 0.0)
        slope = sorted(slopes)[len(slopes) // 2]
        intercepts = sorted(y - slope * x for x, y in zip(xs, ys))
        return slope, intercepts[len(intercepts) // 2]

    def _column_clusters(self, kasalar):
        """Kutuları x-merkezine göre sütunlara ayırır (palet dizilimi)."""
        med_width = sorted(
            float(k["bbox"][2] - k["bbox"][0]) for k in kasalar
        )[len(kasalar) // 2]
        ordered = sorted(
            kasalar, key=lambda k: (k["bbox"][0] + k["bbox"][2]) / 2.0
        )
        clusters = [[ordered[0]]]
        for kasa in ordered[1:]:
            cx = (kasa["bbox"][0] + kasa["bbox"][2]) / 2.0
            prev = clusters[-1][-1]
            prev_cx = (prev["bbox"][0] + prev["bbox"][2]) / 2.0
            if cx - prev_cx > med_width * self.COLUMN_GAP_RATIO:
                clusters.append([kasa])
            else:
                clusters[-1].append(kasa)
        return clusters

    def _height_outliers(self, kasalar, frame_height=None):
        """Komşularının boyuna uymayan kutuları döner.

        [(kasa, boy, referans), ...] — çağıran nedeni görüntüye yazabilsin.
        frame_height verilirse karenin üst/alt kenarına değen kutulara daha
        sıkı eşik uygulanır (yarım görünen arka plan kasaları).
        """
        if len(kasalar) < self.HEIGHT_FILTER_MIN_CRATES:
            return []
        margin = self.HEIGHT_FILTER_EDGE_MARGIN
        window = self.HEIGHT_FILTER_WINDOW
        outliers = []
        for column in self._column_clusters(kasalar):
            if len(column) < self.HEIGHT_FILTER_MIN_COLUMN:
                continue  # kısa sütunda güvenilir referans yok
            ordered = sorted(
                column, key=lambda k: (k["bbox"][1] + k["bbox"][3]) / 2.0
            )
            heights = [float(k["bbox"][3] - k["bbox"][1]) for k in ordered]
            centers = [(k["bbox"][1] + k["bbox"][3]) / 2.0 for k in ordered]
            for position, kasa in enumerate(ordered):
                end = min(len(ordered), max(0, position - window // 2) + window + 1)
                start = max(0, end - window - 1)
                nx = [
                    centers[offset]
                    for offset in range(start, end)
                    if offset != position
                ]
                ny = [
                    heights[offset]
                    for offset in range(start, end)
                    if offset != position
                ]
                if len(ny) < 4:
                    continue
                slope, intercept = self._robust_line(nx, ny)
                reference = slope * centers[position] + intercept
                if reference <= 0:
                    continue
                height = heights[position]
                short_tol = self.HEIGHT_FILTER_SHORT_TOLERANCE
                # Karenin üst/alt kenarında biten kutu yarım görünüyor demektir;
                # KISA tarafta daha sıkı davran (uzun tarafa dokunma).
                if frame_height:
                    y1, y2 = kasa["bbox"][1], kasa["bbox"][3]
                    if y1 <= margin or y2 >= frame_height - margin:
                        short_tol = self.HEIGHT_FILTER_EDGE_TOLERANCE
                low = reference * (1.0 - short_tol)
                high = reference * (1.0 + self.HEIGHT_FILTER_TALL_TOLERANCE)
                if not (low < height < high):
                    outliers.append((kasa, height, reference))
        if len(outliers) > len(kasalar) * self.HEIGHT_FILTER_MAX_DROP:
            return []  # emniyet: referans şüpheli, eleme yapma
        return outliers

    def _column_outliers(self, kasalar):
        """Ana sütunlara oturmayan (arka sıradan sızan) kutuları döner."""
        clusters = self._column_clusters(kasalar)
        min_members = max(
            self.COLUMN_MIN_CRATES, int(len(kasalar) * self.COLUMN_MIN_FRACTION)
        )
        if not any(len(c) >= min_members for c in clusters):
            return []  # hiç güçlü sütun yoksa eleme yapma (emniyet)
        outliers = []
        for cluster in clusters:
            if len(cluster) < min_members:
                outliers.extend(cluster)
        return outliers

    def _filter_background_crates(self, index, result, frame_height=None):
        """Arka plan kutularını sonuçtan çıkarır; sayaçları düzeltir."""
        kasalar = result.get("kasalar") or []
        if frame_height is None:
            annotated = result.get("annotated_image")
            if annotated is not None:
                frame_height = annotated.shape[0]
        if len(kasalar) < self.WIDTH_FILTER_MIN_CRATES:
            return
        widths = [float(k["bbox"][2] - k["bbox"][0]) for k in kasalar]
        avg_width = sum(widths) / len(widths)
        threshold = avg_width * self.WIDTH_FILTER_RATIO
        kept, dropped = [], []
        reasons = {}  # id(kasa) -> görüntüye yazılacak eleme nedeni (ASCII)
        for kasa in kasalar:
            width = float(kasa["bbox"][2] - kasa["bbox"][0])
            if width >= threshold:
                kept.append(kasa)
            else:
                dropped.append(kasa)
                reasons[id(kasa)] = "ARKA PLAN"
        # Yükseklik aykırıları: medyandan ±%20 sapan kutular
        if len(kept) >= self.HEIGHT_FILTER_MIN_CRATES:
            height_dropped = self._height_outliers(kept, frame_height)
            if height_dropped:
                for kasa, height, reference in height_dropped:
                    reasons[id(kasa)] = "AYKIRI BOY %d/%d" % (
                        int(height), int(reference)
                    )
                dropped.extend(kasa for kasa, _, _ in height_dropped)
                dropped_ids = {id(kasa) for kasa, _, _ in height_dropped}
                kept = [k for k in kept if id(k) not in dropped_ids]
        if len(kept) >= self.WIDTH_FILTER_MIN_CRATES:
            column_dropped = self._column_outliers(kept)
            if column_dropped:
                dropped.extend(column_dropped)
                for kasa in column_dropped:
                    reasons[id(kasa)] = "SUTUN DISI"
                dropped_ids = {id(kasa) for kasa in column_dropped}
                kept = [k for k in kept if id(k) not in dropped_ids]
        if not dropped:
            return
        result["kasalar"] = kept
        result["toplam_kasa"] = len(kept)
        dropped_unread_nos = {
            kasa["kasa_no"] for kasa in dropped if not kasa.get("barkodlar")
        }
        result["okunan_barkod"] = max(
            0,
            result.get("okunan_barkod", 0)
            - sum(len(kasa.get("barkodlar") or []) for kasa in dropped),
        )
        result["okunamayan_barkod"] = max(
            0, result.get("okunamayan_barkod", 0) - len(dropped_unread_nos)
        )

        def is_dropped(path):
            base = os.path.basename(str(path))
            return any(f"_kasa_{no}." in base for no in dropped_unread_nos)

        result["barkod_bulunamayan_yollar"] = [
            p for p in result.get("barkod_bulunamayan_yollar") or [] if not is_dropped(p)
        ]
        result["barkod_bulunamayan_isimler"] = [
            n for n in result.get("barkod_bulunamayan_isimler") or [] if not is_dropped(n)
        ]
        annotated = result.get("annotated_image")
        if annotated is not None:
            for kasa in dropped:
                x1, y1, x2, y2 = (int(v) for v in kasa["bbox"])
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (140, 140, 140), 4)
                cv2.putText(
                    annotated,
                    reasons.get(id(kasa), "ARKA PLAN"),
                    (x1 + 6, max(24, y1 + 26)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (140, 140, 140),
                    2,
                    cv2.LINE_AA,
                )
        self.log(
            f"K{index + 1}: {len(dropped)} kutu elendi "
            f"(dar eşik {threshold:.0f}px · komşu boyundan -%"
            f"{int(self.HEIGHT_FILTER_SHORT_TOLERANCE * 100)}/+%"
            f"{int(self.HEIGHT_FILTER_TALL_TOLERANCE * 100)} sapma · sütun dışı) → "
            + ", ".join(
                f"#{k['kasa_no']}[{reasons.get(id(k), 'ARKA PLAN')}]"
                for k in dropped
            )
        )

    def _save_review_crops(self, frame, result, save_dir, timestamp):
        """Palet inceleme ekranı için her kasanın kesitini ayrı dosyaya yazar."""
        review_dir = save_dir / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        height, width = frame.shape[:2]
        for kasa in result.get("kasalar") or []:
            x1, y1, x2, y2 = (int(value) for value in kasa["bbox"])
            x1, x2 = max(0, x1), min(width, x2)
            y1, y2 = max(0, y1), min(height, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop_path = review_dir / f"{timestamp}_kasa_{kasa['kasa_no']}.jpg"
            cv2.imwrite(str(crop_path), frame[y1:y2, x1:x2])
            kasa["tto_crop_path"] = str(crop_path)

    def _apply_camera_result(self, tile, result):
        # Tekilleştirme henüz yapılmadı: kamera şimdilik HAM tespitini yazar,
        # _mark_shared_crates sonrası bu rozet kendi payına dönüşür.
        tile.set_count_badge(raw=result.get("toplam_kasa", 0), pending=True)
        tile.set_stats(
            result.get("toplam_kasa", 0),
            result.get("okunan_barkod", 0),
            result.get("okunamayan_barkod", 0),
        )
        tile.set_unread_badge(result.get("okunamayan_barkod", 0))
        annotated = result.get("annotated_image")
        if annotated is not None:
            tile.show_image(annotated)
        self._schedule_tile_resize()

    def _aggregate(self):
        """Kamera sonuçlarını tekilleştirir; sonuç yoksa None döner."""
        if not any(self.results):
            return None
        aggregate = self.aggregator.aggregate(self.results)
        self.last_aggregate = aggregate
        self.root.after(0, lambda: self._apply_summary(aggregate))
        self.root.after(0, self._build_pallet_from_results)
        self.root.after(0, lambda: self._mark_shared_crates(aggregate))
        self.log(
            f"PALET ÖZETİ · ham={aggregate['raw_total']} · "
            f"çakışan={aggregate['duplicates']} · gerçek={aggregate['unique_total']}"
        )
        return aggregate

    # Ortak kasa çerçeveleri: aynı fiziksel kasayı gören kameralardan yalnız
    # BİRİ onu sayar (sahip = gruptaki EN KÜÇÜK kamera numarası). Sarı = bu
    # kamera sayıyor, gri = başka kameraya verildi.
    SHARED_KEPT_COLOR = (0, 215, 255)      # BGR sarı — ortak, BENDE
    SHARED_GIVEN_COLOR = (165, 165, 165)   # BGR gri  — ortak, başka kamerada

    def _mark_shared_crates(self, aggregate):
        """Ortak kasaları işaretler ve her kameranın sayım payını rozete yazar.

        Tekilleştirme sonrası her kasa grubu tek bir fiziksel kasadır ve gruba
        dahil kameralardan en küçük numaralısına yazılır (CrateAggregator).
        Operatör burada hangi kasanın kime yazıldığını görüp toplamı elle
        doğrulayabilir. Çizim işlenmiş görüntünün KOPYASINA yapılır; her yeni
        toplamada eski işaretler kendiliğinden silinir.
        """
        crates = aggregate.get("crates") or []
        groups = aggregate.get("groups") or []
        unique_per_cam = aggregate.get("unique_per_camera") or []
        boxes_per_cam = {}
        kept_per_cam = {}       # cam -> kaç ORTAK kasayı kendi sayıyor
        given_per_cam = {}      # cam -> kaç kasası başkasına yazıldı
        given_to_per_cam = {}   # cam -> {sahip_cam: adet}
        for members in groups:
            cams = {crates[m]["cam"] for m in members}
            if len(cams) < 2:
                continue
            owner = min(cams)
            for m in members:
                crate = crates[m]
                cam = crate["cam"]
                partners = sorted(cams - {cam})
                partner_text = "+".join(f"K{p + 1}" for p in partners)
                # NOT: cv2.putText yalnız ASCII çizer ("·"/"→" → "??").
                if cam == owner:
                    label = f"ORTAK {partner_text} = BENDE"
                    kept_per_cam[cam] = kept_per_cam.get(cam, 0) + 1
                else:
                    label = f"ORTAK {partner_text} -> K{owner + 1}"
                    given_per_cam[cam] = given_per_cam.get(cam, 0) + 1
                    hedef = given_to_per_cam.setdefault(cam, {})
                    hedef[owner] = hedef.get(owner, 0) + 1
                boxes_per_cam.setdefault(cam, []).append(
                    (crate["bbox"], label, cam == owner)
                )
        shared_total = sum(len(v) for v in boxes_per_cam.values())
        for cam in range(min(MAX_CAMERAS, len(self.tiles))):
            result = self.results[cam] if cam < len(self.results) else None
            annotated = (result or {}).get("annotated_image")
            boxes = boxes_per_cam.get(cam)
            if annotated is not None:
                if not boxes:
                    display = annotated  # işaret yok → temiz görüntüye dön
                else:
                    display = annotated.copy()
                    for (x1, y1, x2, y2), label, kept in boxes:
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        color = (
                            self.SHARED_KEPT_COLOR if kept else self.SHARED_GIVEN_COLOR
                        )
                        thickness = max(4, (y2 - y1) // 16)
                        cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)
                        scale = max(0.9, min(2.2, (y2 - y1) / 55.0))
                        cv2.putText(
                            display,
                            label,
                            (x1 + 12, y2 - max(8, (y2 - y1) // 5)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            scale,
                            color,
                            max(2, int(scale * 2)),
                            cv2.LINE_AA,
                        )
                if not self.preview_running:
                    self.tiles[cam].show_image(display)
            # Kartın köşesindeki "ben bu kadar saydım" rozeti
            if result is None:
                continue
            own = unique_per_cam[cam] if cam < len(unique_per_cam) else 0
            given = given_per_cam.get(cam, 0)
            targets = given_to_per_cam.get(cam, {})
            # Adet zaten "12→" kısmında yazdığı için hedefte yalnız kamera adı
            # gösterilir; rozet dar tutulup görüntüyü kapatmaz.
            given_text = ",".join(
                f"K{owner + 1}"
                for owner, count in sorted(targets.items(), key=lambda kv: -kv[1])
            )
            self.tiles[cam].set_count_badge(
                own=own,
                raw=result.get("toplam_kasa", 0),
                kept=kept_per_cam.get(cam, 0),
                given=given,
                given_text=given_text,
            )
        if shared_total:
            self.log(
                f"Ortak kasa: {shared_total} kutu işaretlendi "
                f"({len(boxes_per_cam)} kamerada) · kamera payları="
                + ", ".join(
                    f"K{cam + 1}:{value}" for cam, value in enumerate(unique_per_cam)
                )
            )

    def _apply_summary(self, aggregate):
        self.metric_total.set(aggregate.get("unique_total", 0))
        self.metric_barcode.set(aggregate.get("unique_barcodes", 0))
        unread = sum(item.get("okunamayan", 0) for item in aggregate.get("per_camera", []))
        self.metric_unread.set(unread)

    # ---------------------------------------------------------- manuel yükleme
    def _manual_upload_for_tile(self, tile):
        path = filedialog.askopenfilename(
            parent=self.root,
            title=f"Kamera {tile.index + 1} için görüntü seçin",
            filetypes=[
                ("Görüntüler", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                ("Tüm dosyalar", "*.*"),
            ],
        )
        if not path:
            return
        self.unread_records = [
            record
            for record in self.unread_records
            if record["camera_index"] != tile.index
        ]
        self.camera_images = [
            image
            for image in self.camera_images
            if image["camera_index"] != tile.index
        ]
        self._refresh_unread_list()
        tile.set_busy(True)
        threading.Thread(
            target=self._manual_upload_tile_worker,
            args=(tile.index, path),
            daemon=True,
        ).start()

    def _manual_upload_tile_worker(self, index, path):
        tile = self.tiles[index]
        self.set_status(f"Kamera {index + 1} manuel işleniyor", theme.CYAN)
        self.log(f"Kamera {index + 1}: manuel görüntü seçildi.")
        try:
            frame = cv2.imread(path)
            if frame is None:
                raise RuntimeError(f"Dosya okunamadı: {path}")
            self.root.after(0, lambda: tile.show_image(frame))
            result, raw_path = self._process_frame(
                index, frame, f"manuel_kamera_{index + 1}"
            )
            self.results[index] = result
            tile.last_result = result
            self.root.after(
                0, lambda: self._apply_camera_result(tile, result)
            )
            self._register_unread(index, result, raw_path)
            self._aggregate()
            self.set_status(f"Kamera {index + 1} manuel tamamlandı", theme.GREEN)
            self.log(f"Kamera {index + 1}: manuel görüntü işlendi.")
        except Exception as exc:
            self.set_status("Manuel yükleme hatası", theme.RED)
            self.log(f"Kamera {index + 1} manuel yükleme HATA: {exc}")
            self.log(traceback.format_exc())
        finally:
            self.root.after(0, lambda: tile.set_busy(False))

    # ---------------------------------------------------------- okunamayanlar
    def _register_unread(self, index, result, raw_path):
        camera_image = {
            "camera_index": index,
            "raw_path": raw_path,
            "annotated_path": result.get("annotated_path"),
        }
        self.camera_images = [
            item for item in self.camera_images if item["camera_index"] != index
        ]
        self.camera_images.append(camera_image)
        # Aynı kameranın önceki çekiminden kalan kayıtlar birikmesin.
        self.unread_records = [
            record
            for record in self.unread_records
            if record["camera_index"] != index
        ]
        paths = result.get("barkod_bulunamayan_yollar", []) or []
        for order, crop_path in enumerate(paths, 1):
            self.unread_records.append(
                {
                    "camera_index": index,
                    "crate": order,
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "crop_path": crop_path,
                    "raw_path": raw_path,
                    "annotated_path": result.get("annotated_path"),
                    "prep_path": result.get("barcode_preprocess_path"),
                }
            )
        self.root.after(0, self._refresh_unread_list)

    def _refresh_unread_list(self):
        self.unread_list.delete(0, "end")
        for record in self.unread_records:
            self.unread_list.insert(
                "end",
                f"K{record['camera_index'] + 1}  ·  Kasa {record['crate']:02d}  ·  {record['time']}",
            )
        self.unread_count.configure(text=f"{len(self.unread_records)} kayıt")

    def _selected_unread(self):
        selected = self.unread_list.curselection()
        if not selected:
            return None
        index = int(selected[0])
        return self.unread_records[index] if index < len(self.unread_records) else None

    def _on_unread_select(self, _event=None):
        self._show_unread_kind("crop")

    def _show_unread_kind(self, kind):
        record = self._selected_unread()
        if not record:
            return
        key = {
            "crop": "crop_path",
            "raw": "raw_path",
            "annotated": "annotated_path",
            "prep": "prep_path",
        }.get(kind, "crop_path")
        path = record.get(key)
        if not path or not os.path.exists(path):
            self.viewer_title.configure(text="Görüntü bulunamadı")
            return
        self._viewer_path = path
        self.viewer_title.configure(
            text=f"Kamera {record['camera_index'] + 1} · Kasa {record['crate']} · {kind}"
        )
        self._render_viewer()

    def _render_viewer(self):
        if not self._viewer_path or not os.path.exists(self._viewer_path):
            return
        try:
            image = Image.open(self._viewer_path).convert("RGB")
            width = max(200, self.viewer_frame.winfo_width() - 8)
            height = max(200, self.viewer_frame.winfo_height() - 8)
            scale = min(width / image.width, height / image.height)
            image = image.resize(
                (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
                Image.Resampling.LANCZOS,
            )
            self._viewer_photo = ImageTk.PhotoImage(image)
            self.viewer_label.configure(image=self._viewer_photo, text="")
        except OSError:
            pass

    # ======================================================= sevkiyat süreci
    # Bir tır = bir sevkiyat süreci. Süreç irsaliye no / plaka / irsaliyedeki
    # kasa adediyle açılır; tırdaki paletler tek tek okunur, her onaylanan palet
    # sürece eklenir ve toplam irsaliye adedine doğru ilerler. Operatör süreci
    # bitirene kadar açık kalır. (Süreç kapanışında irsaliye adedi ve okunan
    # toplam ileride SQL'e yazılacak; şimdilik masaüstüne txt.)
    def _build_session_page(self):
        self.session_start_frame = ctk.CTkFrame(
            self.session_page, fg_color="transparent"
        )
        self.session_active_frame = ctk.CTkFrame(
            self.session_page, fg_color="transparent"
        )
        self._build_session_start(self.session_start_frame)
        self._build_session_active(self.session_active_frame)

    def _build_session_start(self, parent):
        """Süreç yokken görünen başlatma formu."""
        card = ctk.CTkScrollableFrame(
            parent,
            fg_color=theme.SURFACE,
            corner_radius=16,
            border_width=1,
            border_color=theme.BORDER,
        )
        card.pack(fill="both", expand=True)
        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(pady=(26, 20))

        ctk.CTkLabel(
            inner,
            text="YENİ SEVKİYAT",
            text_color=theme.GREEN,
            font=ctk.CTkFont(theme.FONT, 10, "bold"),
        ).pack()
        ctk.CTkLabel(
            inner,
            text="Tırı Karşıla ve Süreci Başlat",
            text_color=theme.TEXT,
            font=ctk.CTkFont(theme.FONT, 26, "bold"),
        ).pack(pady=(2, 4))
        ctk.CTkLabel(
            inner,
            text="Tırdaki paletler tek tek okunur; onayladığınız her palet bu sürece\n"
            "eklenir ve irsaliyedeki adede doğru ilerler.",
            justify="center",
            text_color=theme.TEXT_SOFT,
            font=ctk.CTkFont(theme.FONT, 12),
        ).pack(pady=(0, 16))

        # İleride mini kamera irsaliyeyi okuyup alanları dolduracak.
        self.session_scan_btn = ctk.CTkButton(
            inner,
            text="📷    İRSALİYEYİ OKUT    ·    YAKINDA",
            width=520,
            height=54,
            corner_radius=13,
            state="disabled",
            fg_color=theme.BG_RAISED,
            text_color=theme.TEXT_MUTED,
            border_width=1,
            border_color=theme.BORDER,
            font=ctk.CTkFont(theme.FONT, 13, "bold"),
        )
        self.session_scan_btn.pack()
        ctk.CTkLabel(
            inner,
            text="— ya da bilgileri elle girin —",
            text_color=theme.TEXT_MUTED,
            font=ctk.CTkFont(theme.FONT, 10, "bold"),
        ).pack(pady=(12, 8))

        form = ctk.CTkFrame(inner, fg_color="transparent")
        form.pack()
        form.grid_columnconfigure(0, weight=1, uniform="form")
        form.grid_columnconfigure(1, weight=1, uniform="form")

        def field(row, column, label_text, placeholder, mono=True, span=1):
            wrap = ctk.CTkFrame(form, fg_color="transparent")
            wrap.grid(row=row, column=column, columnspan=span, sticky="ew", padx=6, pady=6)
            ctk.CTkLabel(
                wrap,
                text=label_text,
                anchor="w",
                text_color=theme.TEXT_MUTED,
                font=ctk.CTkFont(theme.FONT, 9, "bold"),
            ).pack(fill="x", pady=(0, 3))
            entry = ctk.CTkEntry(
                wrap,
                width=250,
                height=48,
                placeholder_text=placeholder,
                fg_color=theme.BG_RAISED,
                border_color=theme.BORDER,
                font=ctk.CTkFont(theme.MONO if mono else theme.FONT, 16, "bold"),
            )
            entry.pack(fill="x")
            return entry

        self.start_no_entry = field(0, 0, "İRSALİYE NO", "örn. IRS-2026-0917")
        self.start_plate_entry = field(0, 1, "PLAKA", "örn. 34 ABC 123")
        self.start_qty_entry = field(1, 0, "İRSALİYEDEKİ KASA ADEDİ", "örn. 8074")
        self.start_pallets_entry = field(
            1, 1, "PALET SAYISI  (opsiyonel)", "örn. 33"
        )
        for entry in (
            self.start_no_entry,
            self.start_plate_entry,
            self.start_qty_entry,
            self.start_pallets_entry,
        ):
            entry.bind("<Return>", lambda _event: self._start_session())

        ctk.CTkButton(
            inner,
            text="▶    SÜRECİ BAŞLAT",
            width=520,
            height=62,
            corner_radius=14,
            fg_color=theme.GREEN,
            hover_color=theme.GREEN_HOVER,
            text_color="#FFFFFF",
            font=ctk.CTkFont(theme.FONT, 18, "bold"),
            command=self._start_session,
        ).pack(pady=(16, 6))
        ctk.CTkButton(
            inner,
            text="Sevkiyat açmadan kameraları kontrol et  →",
            width=520,
            height=42,
            corner_radius=12,
            fg_color="transparent",
            hover_color=theme.SURFACE_HOVER,
            text_color=theme.TEXT_MUTED,
            font=ctk.CTkFont(theme.FONT, 11, "bold"),
            command=lambda: self._show_page("count"),
        ).pack()
        ctk.CTkLabel(
            inner,
            text="Sevkiyat açmadan da sayım yapılabilir (bağlantı, pozlama denemesi);\n"
            "ancak o sayımlar hiçbir sürece eklenmez.",
            justify="center",
            text_color=theme.TEXT_MUTED,
            font=ctk.CTkFont(theme.FONT, 10),
        ).pack(pady=(4, 0))

    def _build_session_active(self, parent):
        """Süreç açıkken (ve bittiğinde) görünen durum ekranı."""
        self.session_banner = ctk.CTkLabel(
            parent,
            text="",
            height=54,
            corner_radius=12,
            fg_color=theme.GREEN,
            text_color="#FFFFFF",
            font=ctk.CTkFont(theme.FONT, 15, "bold"),
        )
        self.session_banner.pack(fill="x", pady=(4, 8))

        self.session_cards_frame = ctk.CTkFrame(parent, fg_color="transparent")
        self.session_cards_frame.pack(fill="x", pady=(0, 8))
        self.scard_expected = InfoCard(
            self.session_cards_frame, "İRSALİYE ADEDİ", theme.BLUE, "▤"
        )
        self.scard_counted = InfoCard(
            self.session_cards_frame, "OKUNAN TOPLAM", theme.GREEN, "▣"
        )
        self.scard_remaining = InfoCard(
            self.session_cards_frame, "KALAN", theme.AMBER, "◷"
        )
        self.scard_pallets = InfoCard(
            self.session_cards_frame, "PALET", theme.CYAN, "▥"
        )
        self.session_cards = (
            self.scard_expected,
            self.scard_counted,
            self.scard_remaining,
            self.scard_pallets,
        )
        self._layout_session_cards()

        # İlerleme çubuğunun üstünde yüzde ve kalan: operatör tek bakışta
        # sevkiyatın neresinde olduğunu görür.
        progress_row = ctk.CTkFrame(parent, fg_color="transparent")
        progress_row.pack(fill="x", pady=(0, 3))
        self.session_progress_percent = ctk.CTkLabel(
            progress_row,
            text="",
            anchor="w",
            text_color=theme.GREEN,
            font=ctk.CTkFont(theme.FONT, 12, "bold"),
        )
        self.session_progress_percent.pack(side="left", padx=2)
        self.session_progress_detail = ctk.CTkLabel(
            progress_row,
            text="",
            anchor="e",
            text_color=theme.TEXT_SOFT,
            font=ctk.CTkFont(theme.MONO, 11, "bold"),
        )
        self.session_progress_detail.pack(side="right", padx=2)
        self.session_progress = ctk.CTkProgressBar(
            parent,
            height=16,
            corner_radius=8,
            fg_color=theme.SURFACE_LIGHT,
            progress_color=theme.GREEN,
        )
        self.session_progress.set(0)
        self.session_progress.pack(fill="x", pady=(0, 10))

        head = ctk.CTkFrame(parent, fg_color="transparent")
        head.pack(fill="x", pady=(0, 4))
        self.session_list_title = ctk.CTkLabel(
            head,
            text="OKUNAN PALETLER",
            anchor="w",
            text_color=theme.TEXT_MUTED,
            font=ctk.CTkFont(theme.FONT, 10, "bold"),
        )
        self.session_list_title.pack(side="left", padx=6)
        self.session_list_hint = ctk.CTkLabel(
            head,
            text="",
            anchor="e",
            text_color=theme.TEXT_MUTED,
            font=ctk.CTkFont(theme.FONT, 10),
        )
        self.session_list_hint.pack(side="right", padx=6)

        self.session_scroll = ctk.CTkScrollableFrame(
            parent,
            fg_color=theme.SURFACE,
            corner_radius=14,
            border_width=1,
            border_color=theme.BORDER,
        )
        self.session_scroll.pack(fill="both", expand=True)

        bar = ctk.CTkFrame(
            parent,
            height=84,
            fg_color=theme.SURFACE,
            corner_radius=14,
            border_width=1,
            border_color=theme.BORDER,
        )
        bar.pack(fill="x", pady=(8, 0))
        bar.pack_propagate(False)
        self.session_save_btn = ctk.CTkButton(
            bar,
            text="TXT KAYDET",
            width=140,
            height=52,
            corner_radius=12,
            fg_color=theme.SURFACE_LIGHT,
            hover_color=theme.BORDER_ACTIVE,
            text_color=theme.TEXT,
            font=ctk.CTkFont(theme.FONT, 12, "bold"),
            command=self._on_save_session,
        )
        self.session_save_btn.pack(side="left", padx=12, pady=14)
        self.session_status_label = ctk.CTkLabel(
            bar,
            text="",
            anchor="w",
            text_color=theme.TEXT_SOFT,
            font=ctk.CTkFont(theme.FONT, 13, "bold"),
        )
        self.session_status_label.pack(side="left", padx=6)
        self.session_primary_btn = ctk.CTkButton(
            bar,
            text="▶  YENİ PALET OKU",
            width=250,
            height=56,
            corner_radius=14,
            fg_color=theme.GREEN,
            hover_color=theme.GREEN_HOVER,
            text_color="#FFFFFF",
            font=ctk.CTkFont(theme.FONT, 16, "bold"),
            command=self._session_primary_action,
        )
        self.session_primary_btn.pack(side="right", padx=12, pady=14)
        self.session_finish_btn = ctk.CTkButton(
            bar,
            text="SÜRECİ BİTİR",
            width=170,
            height=56,
            corner_radius=14,
            fg_color=theme.SURFACE_LIGHT,
            hover_color=theme.BORDER_ACTIVE,
            text_color=theme.TEXT,
            border_width=1,
            border_color=theme.BORDER_ACTIVE,
            font=ctk.CTkFont(theme.FONT, 13, "bold"),
            command=self._ask_finish_session,
        )
        self.session_finish_btn.pack(side="right", pady=14)

    def _layout_session_cards(self):
        """Sevkiyat kartları: yatayda 4 sütun, dik ekranda 2 sütun."""
        cards = getattr(self, "session_cards", None)
        if not cards:
            return
        columns = 2 if self.portrait_mode else 4
        for card in cards:
            card.grid_forget()
        for column in range(4):
            active = column < columns
            self.session_cards_frame.grid_columnconfigure(
                column,
                weight=1 if active else 0,
                uniform="session" if active else "",
            )
        for index, card in enumerate(cards):
            card.grid(
                row=index // columns,
                column=index % columns,
                sticky="nsew",
                padx=4,
                pady=4,
            )

    # ------------------------------------------------------- süreç yönetimi
    def _start_session(self):
        """Formdaki bilgilerle yeni sevkiyat sürecini açar."""
        waybill_no = self.start_no_entry.get().strip()
        plate = self.start_plate_entry.get().strip().upper()
        qty_text = self.start_qty_entry.get().strip()
        pallets_text = self.start_pallets_entry.get().strip()
        if not qty_text.isdigit() or int(qty_text) <= 0:
            self._show_toast(
                "⚠ İrsaliyedeki kasa adedini girin (örn. 8074)", theme.AMBER
            )
            return
        if not waybill_no and not plate:
            self._show_toast("⚠ İrsaliye no ya da plaka girin", theme.AMBER)
            return
        self.session = {
            "waybill_no": waybill_no,
            "plate": plate,
            "expected": int(qty_text),
            "pallet_hint": int(pallets_text) if pallets_text.isdigit() else None,
            "started_at": datetime.now(),
            "finished_at": None,
            "pallets": [],
        }
        self.log(
            f"SEVKİYAT AÇILDI · irsaliye={waybill_no or '-'} · plaka={plate or '-'} · "
            f"beklenen={self.session['expected']} kasa"
        )
        self._refresh_session_page()
        self._refresh_capture_buttons()
        self._show_page("count")
        if not self.cameras:
            self._show_toast(
                "▶ Sevkiyat açıldı — önce ⌁ Kameraları Tara ile bağlanın", theme.AMBER
            )
        else:
            self._show_toast(
                f"▶ Sevkiyat açıldı — irsaliye {self.session['expected']} kasa"
            )

    def _session_primary_action(self):
        """Açık süreçte yeni palet okumaya, bitmiş süreçte yeni sevkiyata gider."""
        if self.session is not None and self.session["finished_at"] is not None:
            self.session = None
            self._clear_pallet_state()
            self._refresh_session_page()
            self._show_toast("Yeni sevkiyat için bilgileri girin", theme.CYAN)
            return
        self._show_page("count")

    def _add_pallet_to_session(self, finish=False):
        """Onaylanan paleti sürece ekler; finish=True ise süreci de bitirir."""
        if self.session is None:
            self._show_toast("⚠ Önce sevkiyat başlatın", theme.AMBER)
            self._show_page("session")
            return
        if not self.pallet_crates:
            self._show_toast("⚠ Eklenecek palet yok", theme.AMBER)
            return
        unread = self._unread_crate_count()
        if unread:
            self._show_toast(
                f"⚠ Önce {unread} okunamayan kasayı tamamlayın", theme.AMBER
            )
            self._show_page("pallet")
            return
        stats = self._count_stats()
        entry = {
            "no": 0,  # aşağıda belirlenir (yeni ekleme mi, üstüne yazma mı)
            "time": datetime.now().strftime("%H:%M:%S"),
            "total": stats["total"],
            "barcode": stats["barcode"],
            "ocr": stats["ocr"],
            "manual": stats["manual"],
            "discarded": self._discarded_count,
            "series": self._series_counts(),
        }
        pallets = self.session["pallets"]
        target = self._replace_target
        if target is not None and 0 <= target < len(pallets):
            eski = pallets[target]
            entry["no"] = eski["no"]  # numara korunur, eski kayıt silinir
            pallets[target] = entry
            fiil = f"güncellendi (eski {eski['total']} kasa silindi)"
        else:
            entry["no"] = len(pallets) + 1
            pallets.append(entry)
            fiil = "eklendi"
        self._replace_target = None
        counted = self._session_counted()
        self.log(
            f"SEVKİYAT · palet {entry['no']} {fiil} ({entry['total']} kasa) · "
            f"toplam {counted}/{self.session['expected'] or '?'}"
        )
        self._clear_pallet_state()
        self._refresh_capture_buttons()
        if finish:
            self._finish_session()
            return
        remaining = self._session_remaining()
        if remaining is None:
            tail = f"toplam {counted} kasa"
        elif remaining > 0:
            tail = f"{counted}/{self.session['expected']} · {remaining} kasa kaldı"
        elif remaining == 0:
            tail = "irsaliye adedine ulaşıldı — süreci bitirebilirsiniz"
        else:
            tail = f"⚠ irsaliye {abs(remaining)} kasa aşıldı"
        self._show_toast(f"✓ Palet {entry['no']} {fiil.split(' (')[0]} — {tail}")
        self._refresh_session_page()
        if remaining is not None and remaining <= 0:
            self._show_page("session")  # bitiş kararı operatörde
        else:
            self._show_page("count")

    def _start_revise(self, index):
        """Seçilen paleti yeniden saymaya gider; sonuç eskisinin üstüne yazılır."""
        session = self.session
        if not session or not (0 <= index < len(session["pallets"])):
            return
        if session["finished_at"] is not None:
            self._show_toast("⚠ Kapanmış sevkiyatta revize yapılamaz", theme.AMBER)
            return
        self._replace_target = index
        pallet = session["pallets"][index]
        self._refresh_capture_buttons()
        self._show_page("count")
        self._show_toast(
            f"✎ Palet {pallet['no']} revize ediliyor — sayımı yapıp "
            f"REVİZE ET'e basın",
            theme.AMBER,
        )
        self.log(f"Palet {pallet['no']} revize moduna alındı.")

    def _ask_delete_pallet(self, index):
        session = self.session
        if not session or not (0 <= index < len(session["pallets"])):
            return
        pallet = session["pallets"][index]
        self._open_confirm_modal(
            "PALETİ SİL",
            f"Palet {pallet['no']} ({pallet['total']} kasa) sevkiyattan "
            "silinecek.\n\nOkunan toplam bu kadar azalır ve sonraki paletler "
            "yeniden numaralanır.",
            "🗑  EVET, SİL",
            lambda: self._delete_pallet(index),
            theme.RED,
        )

    def _delete_pallet(self, index):
        session = self.session
        if not session or not (0 <= index < len(session["pallets"])):
            return
        pallet = session["pallets"].pop(index)
        # Numaralar sıralı kalsın
        for sira, kalan in enumerate(session["pallets"], 1):
            kalan["no"] = sira
        if self._replace_target is not None:
            self._replace_target = None  # hedef kaydı değişti, revizeyi iptal et
        self.log(
            f"SEVKİYAT · palet {pallet['no']} silindi ({pallet['total']} kasa) · "
            f"toplam {self._session_counted()}"
        )
        self._show_toast(f"🗑 Palet {pallet['no']} silindi ({pallet['total']} kasa)")
        self._refresh_capture_buttons()
        self._refresh_session_page()

    def _clear_pallet_state(self):
        """Palet sürece yazıldı; ekran bir sonraki palete hazırlanır."""
        # Bekleyen OCR sonuçları yeni palete karışmasın diye kuşak artırılır.
        self._ocr_generation += 1
        self._ocr_progress = None
        self.pallet_crates = []
        self.last_aggregate = None
        self.results = [None] * MAX_CAMERAS
        self.unread_records = []
        self.camera_images = []
        self._discarded_count = 0
        self._ocr_selected = None
        for tile in self.tiles:
            tile.last_result = None
            tile.set_unread_badge(0)
            tile.set_count_badge(None)
        self.metric_total.set(0)
        self.metric_barcode.set(0)
        self.metric_unread.set(0)
        self._refresh_unread_list()
        self._refresh_ocr_list()
        self._refresh_pallet_view()

    def _ask_finish_session(self):
        """Süreci bitirmeden önce eksik/fazla varsa operatöre sorar."""
        if self.session is None:
            return
        if self.session["finished_at"] is not None:
            return
        remaining = self._session_remaining()
        pending = len(self.pallet_crates)
        warnings = []
        if pending:
            warnings.append(
                f"Ekranda eklenmemiş {pending} kasalık bir palet var; "
                "süreci bitirirseniz bu palet sayılmaz."
            )
        if remaining is not None and remaining > 0:
            warnings.append(
                f"İrsaliyeye göre {remaining} kasa eksik "
                f"({self._session_counted()}/{self.session['expected']})."
            )
        elif remaining is not None and remaining < 0:
            warnings.append(
                f"İrsaliye {abs(remaining)} kasa aşıldı "
                f"({self._session_counted()}/{self.session['expected']})."
            )
        if not warnings:
            self._finish_session()
            return
        self._open_confirm_modal(
            "SÜRECİ BİTİR",
            "\n\n".join(warnings) + "\n\nYine de bitirilsin mi?",
            "EVET, BİTİR",
            self._finish_session,
            theme.AMBER,
        )

    def _finish_session(self):
        """Süreci kapatır, raporu txt olarak yazar ve sonuç ekranını gösterir."""
        self._close_serial_modal()
        if self.session is None:
            return
        self.session["finished_at"] = datetime.now()
        counted = self._session_counted()
        expected = self.session["expected"]
        self.log(
            f"SEVKİYAT KAPANDI · {len(self.session['pallets'])} palet · "
            f"{counted}/{expected or '?'} kasa"
        )
        path = self._write_session_report()
        if path is not None:
            self._show_toast(f"✓ Sevkiyat tamamlandı — {path.name}")
        else:
            self._show_toast("✓ Sevkiyat tamamlandı", theme.GREEN)
        self._refresh_session_page()
        self._show_page("session")

    def _open_confirm_modal(self, title, message, ok_text, on_ok, color=theme.AMBER):
        """Basit onay penceresi (dokunmatik için büyük düğmeler)."""
        self._close_serial_modal()
        self._modal_blocker = ctk.CTkFrame(
            self.main, fg_color=theme.IMAGE_BG, corner_radius=0
        )
        self._modal_blocker.place(relx=0, rely=0, relwidth=1, relheight=1)
        self._modal_blocker.bind("<Button-1>", lambda _event: None)
        card = ctk.CTkFrame(
            self.main,
            fg_color=theme.SURFACE,
            corner_radius=18,
            border_width=2,
            border_color=color,
        )
        card.place(relx=0.5, rely=0.5, anchor="center")
        self._modal_card = card
        ctk.CTkLabel(
            card,
            text=title,
            text_color=color,
            font=ctk.CTkFont(theme.FONT, 15, "bold"),
        ).pack(padx=40, pady=(24, 6))
        ctk.CTkLabel(
            card,
            text=message,
            justify="center",
            wraplength=520,
            text_color=theme.TEXT,
            font=ctk.CTkFont(theme.FONT, 13),
        ).pack(padx=40, pady=(0, 16))
        ctk.CTkButton(
            card,
            text=ok_text,
            height=52,
            corner_radius=12,
            fg_color=color,
            hover_color=theme.GREEN_HOVER,
            text_color="#FFFFFF",
            font=ctk.CTkFont(theme.FONT, 14, "bold"),
            command=lambda: (self._close_serial_modal(), on_ok()),
        ).pack(fill="x", padx=28, pady=(0, 8))
        ctk.CTkButton(
            card,
            text="✕  Vazgeç",
            height=44,
            corner_radius=12,
            fg_color=theme.BG_RAISED,
            hover_color=theme.SURFACE_HOVER,
            text_color=theme.TEXT_MUTED,
            border_width=1,
            border_color=theme.BORDER,
            font=ctk.CTkFont(theme.FONT, 12, "bold"),
            command=self._close_serial_modal,
        ).pack(fill="x", padx=28, pady=(0, 20))

    # --------------------------------------------------------- sevkiyat görünümü
    def _session_series_totals(self):
        totals = {}
        for pallet in (self.session or {}).get("pallets", []):
            for serial, count in pallet["series"].items():
                totals[serial] = totals.get(serial, 0) + count
        return dict(sorted(totals.items()))

    def _refresh_session_page(self):
        """Sevkiyat sayfasını (form / açık süreç / bitmiş süreç) yeniler."""
        self._refresh_sidebar_session()
        session = self.session
        if session is None:
            self.session_active_frame.pack_forget()
            self.session_start_frame.pack(fill="both", expand=True)
            return
        self.session_start_frame.pack_forget()
        self.session_active_frame.pack(fill="both", expand=True)

        counted = self._session_counted()
        expected = session["expected"]
        pallets = len(session["pallets"])
        finished = session["finished_at"] is not None
        remaining = expected - counted if expected else None

        head = f"İrsaliye {session['waybill_no'] or '-'}"
        if session["plate"]:
            head += f"  ·  {session['plate']}"
        if finished:
            if remaining == 0:
                text, color = f"✓  SEVKİYAT TAMAMLANDI  ·  {head}", theme.GREEN
            else:
                text, color = f"■  SEVKİYAT KAPANDI  ·  {head}", theme.TEXT_SOFT
        elif remaining is None:
            text, color = f"●  SEVKİYAT AÇIK  ·  {head}", theme.GREEN
        elif remaining > 0:
            text, color = (
                f"●  SEVKİYAT AÇIK  ·  {head}  ·  {remaining} kasa bekleniyor",
                theme.GREEN,
            )
        elif remaining == 0:
            text, color = (
                f"✓  İRSALİYE ADEDİNE ULAŞILDI  ·  {head}  ·  süreci bitirebilirsiniz",
                theme.GREEN,
            )
        else:
            text, color = (
                f"⚠  İRSALİYE {abs(remaining)} KASA AŞILDI  ·  {head}",
                theme.RED,
            )
        self.session_banner.configure(text=text, fg_color=color)

        started = session["started_at"].strftime("%d.%m.%Y %H:%M")
        self.scard_expected.set(
            expected or "—",
            sub=f"başlangıç {started}"
            + (f" · {session['pallet_hint']} palet bekleniyor" if session["pallet_hint"] else ""),
        )
        self.scard_counted.set(
            counted,
            sub=f"%{counted * 100 / expected:.0f} tamamlandı" if expected else "irsaliye adedi girilmedi",
        )
        if remaining is None:
            self.scard_remaining.set("—", sub="irsaliye adedi yok")
        elif remaining >= 0:
            self.scard_remaining.set(
                remaining,
                sub="tamamlandı ✓" if remaining == 0 else "okunacak kasa",
                sub_color=theme.GREEN if remaining == 0 else None,
            )
        else:
            self.scard_remaining.set(
                f"+{abs(remaining)}", sub="irsaliye aşıldı", sub_color=theme.RED
            )
        hint = session["pallet_hint"]
        self.scard_pallets.set(
            pallets,
            sub=f"{hint} palet bekleniyor" if hint else "okunan palet sayısı",
        )
        self.session_progress.set(
            min(1.0, counted / expected) if expected else (1.0 if finished else 0.0)
        )
        if expected:
            yuzde = min(100.0, counted * 100.0 / expected)
            self.session_progress_percent.configure(
                text=f"%{yuzde:.0f} tamamlandı",
                text_color=theme.RED if remaining is not None and remaining < 0 else theme.GREEN,
            )
            detail = f"{counted} / {expected} kasa"
            if remaining is not None and remaining > 0:
                detail += f"   ·   {remaining} kalan"
            elif remaining == 0:
                detail += "   ·   tamamlandı"
            else:
                detail += f"   ·   {abs(remaining)} fazla"
            self.session_progress_detail.configure(text=detail)
        else:
            self.session_progress_percent.configure(
                text="İrsaliye adedi girilmedi", text_color=theme.TEXT_MUTED
            )
            self.session_progress_detail.configure(text=f"{counted} kasa okundu")
        self.session_progress.configure(
            progress_color=theme.RED
            if (remaining is not None and remaining < 0)
            else theme.GREEN
        )

        totals = self._session_series_totals()
        self.session_list_title.configure(
            text=f"OKUNAN PALETLER · {pallets}",
            text_color=theme.TEXT_MUTED,
        )
        self.session_list_hint.configure(
            text="  ".join(f"{serial} × {count}" for serial, count in totals.items())
        )
        self._rebuild_session_rows()

        if finished:
            self.session_status_label.configure(
                text=f"Kapandı {session['finished_at'].strftime('%H:%M')}  ·  "
                f"{pallets} palet · {counted} kasa"
            )
            self.session_primary_btn.configure(text="＋  YENİ SEVKİYAT")
            self.session_finish_btn.configure(state="disabled")
        else:
            self.session_status_label.configure(
                text=f"{counted}/{expected} kasa  ·  {pallets} palet okundu"
                if expected
                else f"{counted} kasa  ·  {pallets} palet okundu"
            )
            self.session_primary_btn.configure(text="▶  YENİ PALET OKU")
            self.session_finish_btn.configure(state="normal")

    def _rebuild_session_rows(self):
        """Sürece eklenen paletlerin listesini kurar (en yeni üstte)."""
        for child in self.session_scroll.winfo_children():
            child.destroy()
        session = self.session
        pallets = list((session or {}).get("pallets", []))
        if not pallets:
            empty = ctk.CTkFrame(self.session_scroll, fg_color="transparent")
            empty.pack(pady=54)
            ctk.CTkLabel(
                empty,
                text="Henüz palet eklenmedi",
                text_color=theme.TEXT,
                font=ctk.CTkFont(theme.FONT, 17, "bold"),
            ).pack()
            ctk.CTkLabel(
                empty,
                text="YENİ PALET OKU ile sayım ekranına geçin; onayladığınız her\n"
                "palet burada sırayla listelenir.",
                justify="center",
                text_color=theme.TEXT_MUTED,
                font=ctk.CTkFont(theme.FONT, 12),
            ).pack(pady=(6, 0))
            return
        for pallet in reversed(pallets):
            row = ctk.CTkFrame(
                self.session_scroll,
                height=62,
                fg_color=theme.BG_RAISED,
                corner_radius=11,
                border_width=1,
                border_color=theme.BORDER,
            )
            row.pack(fill="x", padx=8, pady=4)
            row.pack_propagate(False)
            ctk.CTkLabel(
                row,
                text=f"{pallet['no']:02d}",
                width=48,
                height=40,
                corner_radius=10,
                fg_color=theme.GREEN_DARK,
                text_color=theme.GREEN,
                font=ctk.CTkFont(theme.FONT, 17, "bold"),
            ).pack(side="left", padx=(10, 12), pady=10)
            info = ctk.CTkFrame(row, fg_color="transparent")
            info.pack(side="left", fill="y", pady=10)
            series = "  ".join(
                f"{serial}×{count}" for serial, count in pallet["series"].items()
            )
            ctk.CTkLabel(
                info,
                text=f"{pallet['total']} kasa      {series or '—'}",
                anchor="w",
                text_color=theme.TEXT,
                font=ctk.CTkFont(theme.MONO, 13, "bold"),
            ).pack(anchor="w")
            detail = (
                f"{pallet['time']}  ·  {pallet['barcode']} barkod · "
                f"{pallet['ocr']} sistem · {pallet['manual']} elle"
            )
            if pallet["discarded"]:
                detail += f" · {pallet['discarded']} yanlış tespit silindi"
            ctk.CTkLabel(
                info,
                text=detail,
                anchor="w",
                text_color=theme.TEXT_MUTED,
                font=ctk.CTkFont(theme.FONT, 10),
            ).pack(anchor="w")
            if session and session["finished_at"] is None:
                sira = pallets.index(pallet)
                ctk.CTkButton(
                    row,
                    text="🗑 Sil",
                    width=74,
                    height=36,
                    corner_radius=9,
                    fg_color=theme.BG_RAISED,
                    hover_color=theme.SURFACE_HOVER,
                    text_color=theme.RED,
                    border_width=1,
                    border_color=theme.RED,
                    font=ctk.CTkFont(theme.FONT, 11, "bold"),
                    command=lambda i=sira: self._ask_delete_pallet(i),
                ).pack(side="right", padx=(6, 12))
                ctk.CTkButton(
                    row,
                    text="✎ Revize",
                    width=96,
                    height=36,
                    corner_radius=9,
                    fg_color=theme.AMBER,
                    hover_color="#A8730F",
                    text_color="#FFFFFF",
                    font=ctk.CTkFont(theme.FONT, 11, "bold"),
                    command=lambda i=sira: self._start_revise(i),
                ).pack(side="right")

    def _refresh_sidebar_session(self):
        """Kenar çubuğundaki küçük sevkiyat durumu satırı."""
        label = getattr(self, "sidebar_session", None)
        if label is None:
            return
        session = self.session
        if session is None:
            label.configure(text="sevkiyat yok", text_color=theme.TEXT_MUTED)
            return
        counted = self._session_counted()
        expected = session["expected"]
        state = "kapandı" if session["finished_at"] else "açık"
        label.configure(
            text=f"{state} · {counted}/{expected}" if expected else f"{state} · {counted}",
            text_color=theme.TEXT_MUTED if session["finished_at"] else theme.GREEN,
        )

    # ------------------------------------------------------------ sevkiyat raporu
    def _session_report_lines(self):
        session = self.session
        counted = self._session_counted()
        expected = session["expected"]
        finished = session["finished_at"] or datetime.now()
        lines = [
            "TTO · Trento Toplu Okuma — SEVKİYAT KAYDI",
            f"İrsaliye no    : {session['waybill_no'] or '-'}",
            f"Plaka          : {session['plate'] or '-'}",
            f"Başlangıç      : {session['started_at'].strftime('%d.%m.%Y %H:%M:%S')}",
            f"Bitiş          : {finished.strftime('%d.%m.%Y %H:%M:%S')}",
            "-" * 52,
            f"İRSALİYE ADEDİ : {expected or '-'}",
            f"OKUNAN TOPLAM  : {counted}",
        ]
        if expected:
            diff = counted - expected
            lines.append(
                f"FARK           : {diff:+d}   "
                + ("UYUMLU" if diff == 0 else "FARK VAR")
            )
        lines.append(f"PALET SAYISI   : {len(session['pallets'])}")
        if session["pallet_hint"]:
            lines.append(f"Beklenen palet : {session['pallet_hint']}")
        lines += ["-" * 52, "Paletler:"]
        for pallet in session["pallets"]:
            series = ", ".join(
                f"{serial}×{count}" for serial, count in pallet["series"].items()
            )
            lines.append(
                f"  Palet {pallet['no']:>2} · {pallet['time']} · "
                f"{pallet['total']:>4} kasa · {series or '-'}"
            )
            lines.append(
                f"            {pallet['barcode']} barkod · {pallet['ocr']} sistem "
                f"tespiti · {pallet['manual']} elle"
                + (
                    f" · {pallet['discarded']} yanlış tespit silindi"
                    if pallet["discarded"]
                    else ""
                )
            )
        totals = self._session_series_totals()
        if totals:
            lines += ["-" * 52, "Seri dağılımı (sevkiyat toplamı):"]
            for serial, count in totals.items():
                lines.append(f"  Seri {serial:<6}: {count:>5} adet")
        lines.append("")
        return lines

    def _write_session_report(self):
        session = self.session
        if session is None or not session["pallets"]:
            return None
        stamp = (session["finished_at"] or datetime.now()).strftime("%Y%m%d_%H%M%S")
        tag = "".join(
            ch if ch.isalnum() or ch in "-_" else "_"
            for ch in (session["waybill_no"] or session["plate"] or "sevkiyat")
        )[:30]
        file_path = self._desktop_dir() / f"TTO_sevkiyat_{tag}_{stamp}.txt"
        try:
            file_path.write_text("\n".join(self._session_report_lines()), encoding="utf-8")
        except OSError as exc:
            self.log(f"Sevkiyat kaydı HATASI: {exc}")
            self._show_toast("⚠ Sevkiyat kaydı yazılamadı", theme.RED)
            return None
        self.log(f"Sevkiyat kaydı yazıldı: {file_path}")
        return file_path

    def _on_save_session(self):
        if self.session is None or not self.session["pallets"]:
            self._show_toast("⚠ Kaydedilecek palet yok", theme.AMBER)
            return
        path = self._write_session_report()
        if path is not None:
            self._show_toast(f"✓ Kaydedildi: {path.name}")

    # ------------------------------------------------- sayım doğrulama (palet)
    def _build_pallet_page(self):
        """Sayım bitince otomatik açılan doğrulama ekranı.

        Yapı: durum şeridi → özet kartları (toplam, barkodlu/barkodsuz,
        sistem tespiti, okunamayan, sevkiyat) → seri ve kamera detay satırı →
        yalnız dikkat gerektiren kasaların (okunamayan + elle işaretlenen)
        kesitleri. Okunan kasaların görselleri burada gösterilmez; sistem
        tespitleri kartına dokununca kendi sayfasında incelenir. Onaylanan
        palet açık sevkiyat sürecine eklenir (ileride SQL'e yazılacak).
        """
        self.pallet_banner = ctk.CTkLabel(
            self.pallet_page,
            text="",
            height=54,
            corner_radius=12,
            fg_color=theme.AMBER,
            text_color="#FFFFFF",
            font=ctk.CTkFont(theme.FONT, 15, "bold"),
        )
        self.pallet_banner.pack(fill="x", pady=(4, 8))

        # Özet kartları: yatayda tek sıra, dik ekranda 3 sütun
        # (bkz. _layout_review_cards).
        self.review_cards_frame = ctk.CTkFrame(self.pallet_page, fg_color="transparent")
        self.review_cards_frame.pack(fill="x", pady=(0, 8))
        self.card_total = InfoCard(self.review_cards_frame, "TOPLAM KASA", theme.TEXT, "▣")
        self.card_barcode = InfoCard(
            self.review_cards_frame,
            "BARKOD",
            theme.GREEN,
            "▥",
            rows=("Barkodlu", "Barkodsuz"),
        )
        self.card_system = InfoCard(
            self.review_cards_frame,
            "SİSTEM TARAFINDAN\nTESPİT EDİLEN",
            theme.CYAN,
            "⚡",
            command=lambda: self._show_page("ocr"),
        )
        self.card_unread = InfoCard(self.review_cards_frame, "OKUNAMAYAN", theme.RED, "!")
        self.card_session = InfoCard(
            self.review_cards_frame, "SEVKİYAT", theme.BLUE, "▤", custom=True
        )
        self._build_session_card(self.card_session)
        self.review_cards = (
            self.card_total,
            self.card_barcode,
            self.card_system,
            self.card_unread,
            self.card_session,
        )
        self._layout_review_cards()

        # Seri dağılımı (sol) + kamera detayı (sağ)
        detail = ctk.CTkFrame(
            self.pallet_page,
            height=46,
            fg_color=theme.SURFACE,
            corner_radius=12,
            border_width=1,
            border_color=theme.BORDER,
        )
        detail.pack(fill="x", pady=(0, 8))
        detail.pack_propagate(False)
        self.review_series_frame = ctk.CTkFrame(detail, fg_color="transparent")
        self.review_series_frame.pack(side="left", padx=12, pady=6)
        self.review_camera_label = ctk.CTkLabel(
            detail,
            text="",
            anchor="e",
            text_color=theme.TEXT_MUTED,
            font=ctk.CTkFont(theme.MONO, 10),
        )
        self.review_camera_label.pack(side="right", padx=14)

        grid_head = ctk.CTkFrame(self.pallet_page, fg_color="transparent")
        grid_head.pack(fill="x", pady=(0, 4))
        self.review_grid_title = ctk.CTkLabel(
            grid_head,
            text="OKUNAMAYAN KASALAR",
            anchor="w",
            text_color=theme.RED,
            font=ctk.CTkFont(theme.FONT, 10, "bold"),
        )
        self.review_grid_title.pack(side="left", padx=6)
        self.review_grid_hint = ctk.CTkLabel(
            grid_head,
            text="",
            anchor="e",
            text_color=theme.TEXT_MUTED,
            font=ctk.CTkFont(theme.FONT, 10),
        )
        self.review_grid_hint.pack(side="right", padx=6)

        self.pallet_scroll = ctk.CTkScrollableFrame(
            self.pallet_page,
            fg_color=theme.SURFACE,
            corner_radius=14,
            border_width=1,
            border_color=theme.BORDER,
        )
        self.pallet_scroll.pack(fill="both", expand=True)
        canvas = getattr(self.pallet_scroll, "_parent_canvas", None)
        if canvas is not None:
            # add="+" şart: CTk'nın kendi genişlik-uydurma bind'i korunmalı.
            canvas.bind(
                "<Configure>", lambda _event: self._on_pallet_resize(), add="+"
            )

        action_bar = ctk.CTkFrame(
            self.pallet_page,
            height=84,
            fg_color=theme.SURFACE,
            corner_radius=14,
            border_width=1,
            border_color=theme.BORDER,
        )
        action_bar.pack(fill="x", pady=(8, 0))
        action_bar.pack_propagate(False)
        # Sayım bitince bu ekran kendiliğinden açıldığı için kamera ekranına
        # dönüş yolu burada da olmalı (yeniden çekim, pozlama kontrolü vb.).
        ctk.CTkButton(
            action_bar,
            text="‹  Sayım Ekranı",
            width=150,
            height=52,
            corner_radius=12,
            fg_color=theme.SURFACE_LIGHT,
            hover_color=theme.BORDER_ACTIVE,
            text_color=theme.TEXT,
            font=ctk.CTkFont(theme.FONT, 12, "bold"),
            command=lambda: self._show_page("count"),
        ).pack(side="left", padx=(12, 4), pady=14)
        ctk.CTkButton(
            action_bar,
            text="↻ Sıfırla",
            width=96,
            height=52,
            corner_radius=12,
            fg_color=theme.SURFACE_LIGHT,
            hover_color=theme.BORDER_ACTIVE,
            text_color=theme.TEXT,
            font=ctk.CTkFont(theme.FONT, 12, "bold"),
            command=self._reset_pallet,
        ).pack(side="left", padx=4, pady=14)
        self.pallet_progress = ctk.CTkLabel(
            action_bar,
            text="",
            anchor="w",
            text_color=theme.TEXT_SOFT,
            font=ctk.CTkFont(theme.FONT, 13, "bold"),
        )
        self.pallet_progress.pack(side="left", padx=6)
        self.pallet_confirm_btn = ctk.CTkButton(
            action_bar,
            text="ÖZETİ İNCELE  ›",
            width=250,
            height=56,
            corner_radius=14,
            fg_color=theme.GREEN,
            hover_color=theme.GREEN_HOVER,
            text_color="#FFFFFF",
            font=ctk.CTkFont(theme.FONT, 16, "bold"),
            command=self._confirm_pallet,
        )
        self.pallet_confirm_btn.pack(side="right", padx=12, pady=14)

    def _build_session_card(self, card):
        """Doğrulama ekranındaki sevkiyat kartı: bu palet sürecin neresinde.

        Değerler salt okunur — irsaliye no/plaka/adet sürecin başında bir kez
        girilir (Sevkiyat sayfası). Süreç açık değilse kart deneme modunu
        söyler ve süreci başlatmaya götürür.
        """
        self.session_card_title = ctk.CTkLabel(
            card.body,
            text="",
            anchor="w",
            justify="left",
            wraplength=170,
            text_color=theme.TEXT_MUTED,
            font=ctk.CTkFont(theme.MONO, 9),
        )
        self.session_card_title.pack(anchor="w", fill="x", pady=(0, 3))
        self.session_card_rows = {}
        for key, label_text in (
            ("expected", "İrsaliye"),
            ("pallet", "Bu palet"),
            ("remaining", "Kalan"),
        ):
            row = ctk.CTkFrame(card.body, fg_color="transparent")
            row.pack(fill="x")
            ctk.CTkLabel(
                row,
                text=label_text,
                anchor="w",
                text_color=theme.TEXT_SOFT,
                font=ctk.CTkFont(theme.FONT, 10, "bold"),
            ).pack(side="left")
            value = ctk.CTkLabel(
                row,
                text="—",
                anchor="e",
                text_color=theme.BLUE,
                font=ctk.CTkFont(theme.MONO, 14, "bold"),
            )
            value.pack(side="right")
            self.session_card_rows[key] = value
        self.session_card_btn = ctk.CTkButton(
            card.body,
            text="SEVKİYAT BAŞLAT ›",
            height=28,
            corner_radius=8,
            fg_color=theme.SURFACE_LIGHT,
            hover_color=theme.BORDER_ACTIVE,
            text_color=theme.TEXT,
            font=ctk.CTkFont(theme.FONT, 10, "bold"),
            command=lambda: self._show_page("session"),
        )
        self.session_card_btn.pack(fill="x", pady=(6, 0))

    def _refresh_session_card(self):
        """Sevkiyat kartını ve durum şeridini güncel palete göre yeniler."""
        total = len(self.pallet_crates)
        session = self.session
        if session is None:
            self.session_card_title.configure(
                text="Deneme modu — sevkiyat açık değil", text_color=theme.AMBER
            )
            self.session_card_rows["expected"].configure(text="—")
            self.session_card_rows["pallet"].configure(text=str(total))
            self.session_card_rows["remaining"].configure(text="—")
            self.session_card_btn.configure(text="SEVKİYAT BAŞLAT ›")
        else:
            plate = f" · {session['plate']}" if session["plate"] else ""
            self.session_card_title.configure(
                text=f"{session['waybill_no'] or 'no yok'}{plate}\n"
                f"Palet {len(session['pallets']) + 1} · şu ana kadar "
                f"{self._session_counted()} kasa",
                text_color=theme.TEXT_MUTED,
            )
            self.session_card_rows["expected"].configure(
                text=str(session["expected"]) if session["expected"] else "—"
            )
            self.session_card_rows["pallet"].configure(text=str(total))
            remaining = self._session_remaining(include_current=True)
            if remaining is None:
                text, color = "—", theme.BLUE
            elif remaining < 0:
                text, color = f"+{abs(remaining)}", theme.RED
            else:
                text, color = str(remaining), theme.GREEN if remaining == 0 else theme.BLUE
            self.session_card_rows["remaining"].configure(text=text, text_color=color)
            self.session_card_btn.configure(text="SEVKİYAT ÖZETİ ›")
        self._refresh_pallet_banner()

    def _layout_review_cards(self):
        """Özet kartlarını yatayda tek sıraya, dik ekranda 3 sütuna dizer."""
        count = len(self.review_cards)
        columns = 3 if self.portrait_mode else count
        for card in self.review_cards:
            card.grid_forget()
        for column in range(count):
            active = column < columns
            self.review_cards_frame.grid_columnconfigure(
                column,
                weight=1 if active else 0,
                uniform="review" if active else "",
            )
        for index, card in enumerate(self.review_cards):
            row, column = divmod(index, columns)
            last = index == count - 1
            card.grid(
                row=row,
                column=column,
                columnspan=columns - column if last else 1,
                sticky="nsew",
                padx=4,
                pady=4,
            )

    def _load_pallet_demo(self):
        """Gerçek çekim yokken arayüzü denemek için demo verisi üretir."""
        unread_slots = {5, 9, 16, 22, 27}
        ocr_slots = {3, 11, 19}  # sistem (OCR) tespiti
        manual_slots = {30}  # operatör seçimi
        self.pallet_crates = []
        for index in range(32):
            serial = (
                None
                if index in unread_slots
                else self.pallet_serial_options[(index * 13 + 7) % 4]
            )
            self.pallet_crates.append(
                {
                    "serial": serial,
                    "manual": index in manual_slots,
                    "ocr": index in ocr_slots,
                    "crop_path": None,
                    "camera_index": None,
                    "kasa_no": None,
                    "barcodes": [],
                    "source": "demo",
                }
            )
        self.last_aggregate = None

    def _reset_pallet(self):
        """Elle yapılan işaretlemeleri atar, son çekim sonuçlarına döner."""
        self._close_serial_modal()
        if self.last_aggregate is not None:
            self._build_pallet_from_results()
            self._show_toast("↻ Son çekim sonuçlarına dönüldü", theme.CYAN)
        elif self.pallet_crates:
            self._load_pallet_demo()
            self._show_toast("↻ Demo palet yeniden yüklendi", theme.CYAN)
        self._refresh_pallet_view()

    def _unread_crate_count(self):
        return sum(1 for crate in self.pallet_crates if crate["serial"] is None)

    @staticmethod
    def _serial_from_barcodes(barcodes):
        """Barkod listesinden 4 haneli kasa serisini (6412 vb.) çıkarır."""
        for barcode in barcodes or []:
            text = str(barcode).strip()
            if len(text) >= 4 and text[:4].isdigit() and text[:2] in ("64", "43"):
                return text[:4]
        for barcode in barcodes or []:
            text = str(barcode).strip()
            if text:
                return text[:8]  # seri çözülemedi; barkodun başını göster
        return None

    def _serial_options(self):
        """Manuel seçim düğmeleri: bu çekimde görülen seriler + varsayılanlar."""
        seen = sorted(
            {
                crate["serial"]
                for crate in self.pallet_crates
                if crate["serial"] and len(str(crate["serial"])) == 4
            }
        )
        options = list(seen)
        for serial in self.pallet_serial_options:
            if serial not in options:
                options.append(serial)
        return options[:6]

    def _build_pallet_from_results(self):
        """Toplama sonucundaki tekil kasa gruplarından inceleme listesi kurar."""
        aggregate = self.last_aggregate
        if not aggregate:
            return
        crates = aggregate.get("crates") or []
        groups = aggregate.get("groups") or []
        entries = []
        for members in groups:
            rep = None
            for member in members:
                if crates[member]["barkodlar"]:
                    rep = crates[member]
                    break
            if rep is None:
                rep = crates[members[0]]
            cam = rep["cam"]
            kasa_no = rep.get("kasa_no")
            crop_path = None
            result = self.results[cam] if cam < len(self.results) else None
            image_width = (result or {}).get("image_width") or 1
            if result:
                for kasa in result.get("kasalar") or []:
                    if kasa.get("kasa_no") == kasa_no:
                        crop_path = kasa.get("tto_crop_path")
                        break
            raw_path = next(
                (
                    item["raw_path"]
                    for item in self.camera_images
                    if item["camera_index"] == cam
                ),
                None,
            )
            entries.append(
                {
                    "serial": self._serial_from_barcodes(rep["barkodlar"]),
                    "manual": False,
                    "ocr": False,
                    "crop_path": crop_path,
                    "camera_index": cam,
                    "kasa_no": kasa_no,
                    "bbox": tuple(rep["bbox"]),
                    "raw_path": raw_path,
                    "members": [
                        (crates[m]["cam"], crates[m].get("kasa_no"))
                        for m in members
                    ],
                    "barcodes": list(rep["barkodlar"]),
                    "source": "real",
                    "_sort": (cam, int(rep["cx"] > image_width / 2), rep["cy"]),
                }
            )
        entries.sort(key=lambda entry: entry.pop("_sort"))
        self.pallet_crates = entries
        self._discarded_count = 0  # yeni toplama; eski silmeler geçersiz
        self._sync_unread_metrics()
        self._refresh_ocr_list()
        if self.active_page == "pallet":
            self._refresh_pallet_view()
        self._start_unread_ocr()

    # ------------------------------------------------- okunamayanlara OCR
    def _start_unread_ocr(self):
        """Okunamayan kasaların basılı numarasını arka planda OCR ile okur.

        Kasa üstünde ör. "12" yazıyorsa seri otomatik "6412" olarak
        etiketlenir; OCR da okuyamazsa kasa kırmızı kalır ve operatöre
        sorulur (mevcut manuel seçim akışı).
        """
        if not self.barcode_ready:
            # Barkod motoru bozukken sayım zaten güvenilmez; OCR ile üstünü
            # örtmek yanlış sonucu gizler.
            self.log("OCR atlandı: barkod motoru hazır değil.")
            return
        self._ocr_generation += 1
        generation = self._ocr_generation
        targets = []
        for index, crate in enumerate(self.pallet_crates):
            if crate.get("source") != "real" or crate["serial"] is not None:
                continue
            paths = self._member_crop_paths(crate)
            if paths:
                targets.append((index, paths))
            else:
                self.log(f"OCR: kasa {index + 1} için kesit bulunamadı, atlandı.")
        if not targets:
            return
        error = ocr_reader.availability_error()
        if error:
            self.log(f"OCR devre dışı: {error}")
            if not self._ocr_warned:
                self._ocr_warned = True
                self._show_toast(f"⚠ OCR devre dışı — {error}", theme.RED)
            return
        self.log(f"OCR: {len(targets)} okunamayan kasa numara için taranacak…")
        self._ocr_t0 = time.perf_counter()
        self._ocr_n = len(targets)
        self._ocr_progress = (0, len(targets))
        self._refresh_pallet_banner()
        threading.Thread(
            target=self._unread_ocr_worker,
            args=(generation, targets),
            daemon=True,
        ).start()

    def _member_crop_paths(self, crate):
        """Grubun tüm üyelerinin kesit yollarını döner (temsilci önce).

        Aynı fiziksel kasa birden çok kamerada görünmüş olabilir; temsilci
        kesitte parlama varsa diğer kameranın kesiti OCR'a şans verir.
        """
        paths = []
        if crate.get("crop_path"):
            paths.append(crate["crop_path"])
        for cam, kasa_no in crate.get("members") or []:
            result = self.results[cam] if cam is not None and cam < len(self.results) else None
            for kasa in (result or {}).get("kasalar") or []:
                if kasa.get("kasa_no") == kasa_no:
                    path = kasa.get("tto_crop_path")
                    if path and path not in paths:
                        paths.append(path)
                    break
        return [p for p in paths if p and os.path.exists(p)]

    def _unread_ocr_worker(self, generation, targets):
        resolved = 0
        for done, (index, crop_paths) in enumerate(targets, 1):
            self.set_status(f"OCR okuyor ({done}/{len(targets)})", theme.CYAN)
            self.root.after(
                0,
                lambda d=done, n=len(targets), g=generation: self._set_ocr_progress(d, n, g),
            )
            serial = None
            for attempt, crop_path in enumerate(crop_paths, 1):
                if generation != self._ocr_generation:
                    self.log("OCR: yeni çekim başladı, eski tarama iptal edildi.")
                    return  # yeni çekim başladı; eski sonuçları uygulama
                started = time.time()
                serial = ocr_reader.read_crate_serial(
                    crop_path,
                    prefix="64",
                    log_fn=self.log,
                    tag=f"kasa{index + 1}.{attempt}",
                )
                self.log(
                    f"OCR kasa{index + 1}.{attempt}: {time.time() - started:.1f} sn"
                    f" → {serial or 'okunamadı'}"
                )
                if serial:
                    break
            if generation != self._ocr_generation:
                return
            if serial:
                resolved += 1
                self.root.after(
                    0,
                    lambda i=index, s=serial, g=generation: self._apply_ocr_serial(i, s, g),
                )
        def finish():
            if generation != self._ocr_generation:
                return
            self._ocr_progress = None
            t = self._last_timing
            if t is not None and self._ocr_t0 is not None and self._t_run is not None:
                t["ocr"] = time.perf_counter() - self._ocr_t0
                t["ocr_kasa"] = getattr(self, "_ocr_n", 0)
                t["toplam_ocr"] = time.perf_counter() - self._t_run
                self._ocr_t0 = None
                self._refresh_timing_views()
                self._write_timing_record({"tip": "ocr_sonrasi"})
                self.log(f"[HIZLI] OCR {t['ocr']:.2f} sn · toplam (OCR dahil) {t['toplam_ocr']:.2f} sn")
            remaining = self._unread_crate_count()
            # Bilgilendirme penceresi yok: operatör zaten doğrulama ekranında,
            # şerit ve kartlar sonucu gösterir; kısa bir bildirim yeter.
            if remaining:
                self.set_status(f"{remaining} kasa elle seçilmeli", theme.AMBER)
                self._show_toast(
                    f"⚠ Sistem {resolved} kasayı tespit etti — "
                    f"{remaining} kasa elle seçilmeli",
                    theme.AMBER,
                )
            else:
                self.set_status("Sayım tamamlandı", theme.GREEN)
                if resolved:
                    self._show_toast(
                        f"✓ Sistem {resolved} kasayı tespit etti — tüm kasalar okundu"
                    )
            if self.active_page == "pallet":
                self._refresh_pallet_view()

        self.root.after(0, finish)

    def _set_ocr_progress(self, done, total, generation):
        if generation != self._ocr_generation:
            return
        self._ocr_progress = (done, total)
        self._refresh_pallet_banner()

    def _apply_ocr_serial(self, index, serial, generation):
        if generation != self._ocr_generation or index >= len(self.pallet_crates):
            return
        crate = self.pallet_crates[index]
        if crate["serial"] is not None:
            return  # operatör bu arada elle seçmiş
        crate["serial"] = str(serial)
        crate["manual"] = False
        crate["ocr"] = True
        self.log(f"OCR: kasa {index + 1} → {serial} olarak etiketlendi.")
        self._sync_unread_metrics()
        self._refresh_ocr_list()
        if self.active_page == "pallet":
            self._refresh_pallet_view()

    def _sync_unread_metrics(self):
        """Okunamayan sayacını ve kamera rozetlerini inceleme listesiyle eşitler."""
        if not any(crate.get("source") == "real" for crate in self.pallet_crates):
            return
        per_camera = {}
        for crate in self.pallet_crates:
            if crate["serial"] is None and crate["camera_index"] is not None:
                per_camera[crate["camera_index"]] = (
                    per_camera.get(crate["camera_index"], 0) + 1
                )
        for tile in self.tiles:
            # Canlı önizlemede rozet gösterilmez; eski sayımın uyarısı canlı
            # görüntünün üstünde yanıp sönmesin.
            tile.set_unread_badge(
                0 if self.preview_running else per_camera.get(tile.index, 0)
            )
        self.metric_unread.set(sum(per_camera.values()))

    def _count_stats(self):
        """Doğrulama kartlarının sayıları: kasa hangi yolla okundu?"""
        total = len(self.pallet_crates)
        unread = self._unread_crate_count()
        ocr = sum(1 for crate in self.pallet_crates if crate.get("ocr"))
        manual = sum(1 for crate in self.pallet_crates if crate.get("manual"))
        barcode = max(0, total - unread - ocr - manual)
        return {
            "total": total,
            "unread": unread,
            "ocr": ocr,
            "manual": manual,
            "barcode": barcode,
            "nonbarcode": total - barcode,
        }

    # ------------------------------------------------------------ sevkiyat hesapları
    def _session_counted(self):
        """Süreçte şimdiye kadar ONAYLANAN paletlerin kasa toplamı."""
        session = self.session
        return sum(item["total"] for item in session["pallets"]) if session else 0

    def _session_remaining(self, include_current=False):
        """İrsaliyeye göre kalan kasa; süreç/adet yoksa None.

        include_current=True ise ekranda duran (henüz onaylanmamış) palet de
        düşülür — "bu paleti eklersem ne kalır" sorusunun cevabı.
        """
        session = self.session
        if not session or not session["expected"]:
            return None
        counted = self._session_counted()
        if include_current:
            counted += len(self.pallet_crates)
        return session["expected"] - counted

    def _session_summary_text(self):
        """(metin, renk) — özet sayfası ve sevkiyat şeridi ortak kullanır."""
        session = self.session
        if session is None:
            return "Deneme modu — sevkiyat açık değil, bu sayım kaydedilmez", theme.AMBER
        counted = self._session_counted()
        expected = session["expected"]
        pallets = len(session["pallets"])
        head = f"İrsaliye {session['waybill_no'] or '-'}"
        if session["plate"]:
            head += f" · {session['plate']}"
        if not expected:
            return f"{head} · {pallets} palet · {counted} kasa okundu", theme.TEXT_MUTED
        diff = counted - expected
        if diff == 0 and pallets:
            return f"{head} · ✓ irsaliye tamamlandı ({counted}/{expected})", theme.GREEN
        if diff > 0:
            return f"{head} · ⚠ irsaliye {diff} kasa AŞILDI ({counted}/{expected})", theme.RED
        return (
            f"{head} · {counted}/{expected} kasa · {abs(diff)} kalan · {pallets} palet",
            theme.TEXT_MUTED,
        )

    # ------------------------------------------------------------ şerit / kartlar
    def _banner_state(self):
        """Doğrulama şeridinin metni ve rengi; öncelik sırası önemlidir."""
        total = len(self.pallet_crates)
        unread = self._unread_crate_count()
        if not total:
            if self._counting:
                return "⏳  Sayım yapılıyor — kameralar okunuyor…", theme.AMBER
            return (
                "Henüz sayım yok — Sayım Ekranından TÜMÜNÜ OKU ile başlayın",
                theme.AMBER,
            )
        if self._ocr_progress:
            done, count = self._ocr_progress
            return (
                f"⏳  Sistem okunamayan kasaları tespit ediyor ({done}/{count})"
                f" — {unread} kasa bekliyor",
                theme.CYAN,
            )
        if unread:
            return (
                f"⚠  {unread} kasa okunamadı — kırmızı yanıp sönen kasaya dokunup seriyi seçin",
                theme.RED,
            )
        if self.session is None:
            return (
                "✓  Tüm kasalar okundu — deneme modu; palet eklemek için sevkiyat başlatın",
                theme.AMBER,
            )
        remaining = self._session_remaining(include_current=True)
        if remaining is None:
            return "✓  Tüm kasalar okundu — paleti sevkiyata ekleyebilirsiniz", theme.GREEN
        if remaining < 0:
            return (
                f"⚠  Tüm kasalar okundu — bu palet irsaliyeyi {abs(remaining)} kasa AŞIYOR",
                theme.RED,
            )
        if remaining == 0:
            return (
                "✓  Tüm kasalar okundu — bu palet irsaliyeyi tamamlıyor, sevkiyat bitirilebilir",
                theme.GREEN,
            )
        return (
            f"✓  Tüm kasalar okundu — paleti ekleyin, {remaining} kasa kalacak",
            theme.GREEN,
        )
    def _refresh_pallet_banner(self):
        text, color = self._banner_state()
        self.pallet_banner.configure(text=text, fg_color=color)

    def _refresh_series_chips(self):
        for child in self.review_series_frame.winfo_children():
            child.destroy()
        counts = self._series_counts()
        if not counts:
            ctk.CTkLabel(
                self.review_series_frame,
                text="Seri dağılımı sayımdan sonra burada görünür",
                text_color=theme.TEXT_MUTED,
                font=ctk.CTkFont(theme.FONT, 10),
            ).pack(side="left")
            return
        ctk.CTkLabel(
            self.review_series_frame,
            text="SERİ",
            text_color=theme.TEXT_MUTED,
            font=ctk.CTkFont(theme.FONT, 9, "bold"),
        ).pack(side="left", padx=(0, 10))
        for position, (serial, count) in enumerate(counts.items()):
            color = self.pallet_series_colors[position % len(self.pallet_series_colors)]
            ctk.CTkLabel(
                self.review_series_frame,
                text=f"● {serial} × {count}",
                text_color=color,
                font=ctk.CTkFont(theme.MONO, 12, "bold"),
            ).pack(side="left", padx=(0, 16))

    def _camera_detail_text(self):
        aggregate = self.last_aggregate
        if not aggregate:
            return ""
        parts = [
            f"K{cam['cam'] + 1} {cam['kasa']}"
            for cam in aggregate.get("per_camera", [])
            if cam.get("kasa")
        ]
        raw = aggregate.get("raw_total", 0)
        duplicates = aggregate.get("duplicates", 0)
        tail = f"ham {raw} → tekil {len(self.pallet_crates)}"
        if duplicates:
            tail += f" ({duplicates} ortak)"
        return f"{' · '.join(parts)}     {tail}" if parts else tail

    def _total_sub_text(self, stats):
        if not stats["total"]:
            return "sayım bekleniyor"
        cameras = [
            cam
            for cam in (self.last_aggregate or {}).get("per_camera", [])
            if cam.get("kasa")
        ]
        parts = [f"{len(cameras)} kameradan tekilleştirildi" if cameras else "demo veri"]
        if self._discarded_count:
            parts.append(f"{self._discarded_count} yanlış tespit silindi")
        return " · ".join(parts)

    def _refresh_pallet_view(self):
        """Şerit, özet kartları, detay satırı, alt çubuk ve kasa kesitlerini yeniler."""
        stats = self._count_stats()
        total, unread = stats["total"], stats["unread"]

        self.card_total.set(total, sub=self._total_sub_text(stats))
        self.card_barcode.set(
            rows=(stats["barcode"], stats["nonbarcode"]),
            sub=(
                f"barkodsuz: {stats['ocr']} sistem · {stats['manual']} elle · {unread} okunamayan"
                if total
                else "barkod okuma sonucu"
            ),
        )
        self.card_system.set(
            stats["ocr"],
            sub=(
                "dokunun → kasa görüntüleri ve tespit çıktısı"
                if stats["ocr"]
                else "basılı numaradan tespit edilen kasa yok"
            ),
        )
        if stats["manual"]:
            unread_sub, unread_color = f"✎ {stats['manual']} kasa elle işaretlendi", theme.BLUE
        elif total and not unread:
            unread_sub, unread_color = "hepsi çözüldü ✓", theme.GREEN
        else:
            unread_sub, unread_color = "operatör seçimi bekleyen kasa", None
        self.card_unread.set(unread, sub=unread_sub, sub_color=unread_color)
        self._refresh_session_card()  # şeridi de yeniler
        self._refresh_series_chips()
        self.review_camera_label.configure(text=self._camera_detail_text())

        if total:
            self.pallet_progress.configure(
                text=(
                    f"{total - unread}/{total} okundu   ·   "
                    f"{stats['barcode']} barkod · {stats['ocr']} sistem · {stats['manual']} elle"
                )
            )
            self.pallet_confirm_btn.configure(
                state="normal",
                text=f"ÖZETİ İNCELE  ({unread} eksik)"
                if unread
                else "ÖZETİ İNCELE  ›",
            )
        else:
            self.pallet_progress.configure(text="")
            self.pallet_confirm_btn.configure(
                state="disabled", text="ÖZETİ İNCELE  ›"
            )

        title = f"OKUNAMAYAN KASALAR · {unread}"
        if stats["manual"]:
            title += f"      ELLE İŞARETLENEN · {stats['manual']}"
        self.review_grid_title.configure(
            text=title, text_color=theme.RED if unread else theme.TEXT_MUTED
        )
        self.review_grid_hint.configure(
            text=(
                "kasaya dokunarak seriyi seçin ya da düzeltin"
                if unread or stats["manual"]
                else ""
            )
        )
        self._rebuild_pallet_cards()

    def _on_pallet_resize(self):
        if self.active_page != "pallet" or not self._pallet_cards:
            return
        width = self._pallet_grid_width()
        if abs(width - self._pallet_width) > 60:
            self._rebuild_pallet_cards()

    def _pallet_grid_width(self):
        canvas = getattr(self.pallet_scroll, "_parent_canvas", None)
        width = canvas.winfo_width() if canvas is not None else 0
        if width < 100:
            width = self.root.winfo_width() - (40 if self.portrait_mode else 300)
        return max(400, width)

    def _rebuild_pallet_cards(self):
        """Dikkat gerektiren kasaların (okunamayan + elle) kesitlerini dizer.

        Barkodla ya da sistemle okunan kasalar burada listelenmez; elle
        işaretlenenler operatörün yanlış dokunuşu geri alabilmesi için kalır.
        """
        for child in self.pallet_scroll.winfo_children():
            child.destroy()
        self._pallet_cards = []
        self._pallet_photos = []

        columns = 2 if self.portrait_mode else 3
        for column in range(columns):
            self.pallet_scroll.grid_columnconfigure(column, weight=1, uniform="pallet")

        if not self.pallet_crates:
            empty = ctk.CTkFrame(self.pallet_scroll, fg_color="transparent")
            empty.grid(row=0, column=0, columnspan=columns, pady=70)
            ctk.CTkLabel(
                empty,
                text="Henüz sayım yok",
                text_color=theme.TEXT,
                font=ctk.CTkFont(theme.FONT, 18, "bold"),
            ).pack()
            ctk.CTkLabel(
                empty,
                text="TÜMÜNÜ OKU bitince bu ekran kendiliğinden açılır; okunamayan\n"
                "kasalar burada kesitleriyle listelenir, okunanlar gösterilmez.",
                text_color=theme.TEXT_MUTED,
                justify="center",
                font=ctk.CTkFont(theme.FONT, 12),
            ).pack(pady=(6, 14))
            ctk.CTkButton(
                empty,
                text="Demo Yükle",
                width=150,
                height=40,
                corner_radius=10,
                fg_color=theme.SURFACE_LIGHT,
                hover_color=theme.BORDER_ACTIVE,
                text_color=theme.TEXT,
                font=ctk.CTkFont(theme.FONT, 11, "bold"),
                command=lambda: (self._load_pallet_demo(), self._refresh_pallet_view()),
            ).pack()
            return

        # Okunamayanlar önce (kırmızı, yanıp söner), sonra elle işaretlenenler.
        visible = [
            (index, crate)
            for index, crate in enumerate(self.pallet_crates)
            if crate["serial"] is None
        ] + [
            (index, crate)
            for index, crate in enumerate(self.pallet_crates)
            if crate["serial"] is not None and crate.get("manual")
        ]
        if not visible:
            empty = ctk.CTkFrame(self.pallet_scroll, fg_color="transparent")
            empty.grid(row=0, column=0, columnspan=columns, pady=60)
            ctk.CTkLabel(
                empty,
                text="✓  Dikkat gerektiren kasa yok",
                text_color=theme.GREEN,
                font=ctk.CTkFont(theme.FONT, 18, "bold"),
            ).pack()
            ctk.CTkLabel(
                empty,
                text="Tüm kasalar barkod ya da sistem tespitiyle okundu.",
                text_color=theme.TEXT_MUTED,
                font=ctk.CTkFont(theme.FONT, 12),
            ).pack(pady=(6, 0))
            return

        width = self._pallet_grid_width()
        self._pallet_width = width
        image_width = max(160, int(width / columns) - 42)

        # Kartlar hafif tk widget'larıyla kurulur; index kasanın gerçek sırası
        # olarak korunur ki tıklama/blink doğru kasaya gitsin.
        for position, (index, crate) in enumerate(visible):
            unread = crate["serial"] is None
            border = theme.RED if unread else theme.BLUE
            card = tk.Frame(
                self.pallet_scroll,
                bg=theme.SURFACE,
                highlightthickness=3,
                highlightbackground=border,
                highlightcolor=border,
            )
            card.grid(
                row=position // columns,
                column=position % columns,
                sticky="ew",
                padx=6,
                pady=6,
            )
            card.grid_columnconfigure(0, weight=1)

            image_label = tk.Label(card, bg=theme.IMAGE_BG, cursor="hand2")
            photo = None
            crop_path = crate.get("crop_path")
            if crop_path and os.path.exists(crop_path):
                try:
                    image = Image.open(crop_path)
                    image.draft("RGB", (image_width, image_width))
                    image = image.convert("RGB")
                    scale = image_width / image.width
                    image = image.resize(
                        (image_width, max(24, int(image.height * scale))),
                        Image.Resampling.BILINEAR,
                    )
                    photo = ImageTk.PhotoImage(image)
                except OSError:
                    photo = None
            if photo is not None:
                image_label.configure(image=photo)
                self._pallet_photos.append(photo)
            else:
                image_label.configure(
                    text=crate["serial"] or "?",
                    fg=theme.RED if unread else theme.TEXT_SOFT,
                    font=(theme.MONO, 22, "bold"),
                    height=2,
                )
            image_label.grid(row=0, column=0, padx=9, pady=(9, 3))

            footer = tk.Frame(card, bg=theme.SURFACE)
            footer.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 8))
            # Kasayı gören TÜM kameralar yazılır ("K1+K4"): operatör hangi
            # kameradan bakacağını bilir. Tek kamera görmüşse eskisi gibi.
            cams = sorted(
                {cam for cam, _ in (crate.get("members") or []) if cam is not None}
            )
            if not cams and crate["camera_index"] is not None:
                cams = [crate["camera_index"]]
            if cams:
                position_text = "+".join(f"K{cam + 1}" for cam in cams)
                position_text += f" · {crate['kasa_no'] or index + 1}"
            else:
                position_text = f"Kasa {index + 1}"
            tk.Label(
                footer,
                text=position_text,
                bg=theme.SURFACE,
                fg=theme.TEXT_MUTED,
                font=(theme.MONO, 9),
            ).pack(side="left")
            if len(cams) > 1:
                tk.Label(
                    footer,
                    text=f" {len(cams)} kamerada ",
                    bg=theme.GREEN_DARK,
                    fg=theme.CYAN,
                    font=(theme.MONO, 8, "bold"),
                ).pack(side="left", padx=6)
            if unread:
                serial_text, serial_color = "? ? ? ?", theme.RED
            else:
                serial_text, serial_color = f"{crate['serial']}  ✎", theme.BLUE
            serial_label = tk.Label(
                footer,
                text=serial_text,
                bg=theme.SURFACE,
                fg=serial_color,
                font=(theme.MONO, 13, "bold"),
            )
            serial_label.pack(side="right")

            for widget in (card, image_label, footer, serial_label):
                widget.bind(
                    "<Button-1>",
                    lambda _event, i=index: self._on_crate_click(i),
                )
            self._pallet_cards.append(
                {"index": index, "frame": card, "serial_label": serial_label}
            )

    # ------------------------------------------------------------ yanıp sönme
    def _blink_tick(self):
        """Uygulama boyunca çalışan tek blink döngüsü: rozetler + kartlar."""
        self._blink_job = None
        self.blink_on = not self.blink_on
        for tile in self.tiles:
            tile.blink_badge(self.blink_on)
        if self.active_page == "pallet":
            for card in self._pallet_cards:
                crate = self.pallet_crates[card["index"]] if card["index"] < len(self.pallet_crates) else None
                if not crate or crate["serial"] is not None:
                    continue
                blink_color = theme.RED if self.blink_on else "#F3B6B6"
                try:
                    card["frame"].configure(
                        highlightbackground=blink_color, highlightcolor=blink_color
                    )
                    card["serial_label"].configure(
                        fg=theme.RED if self.blink_on else "#E8A5A5"
                    )
                except tk.TclError:
                    pass
        self._blink_job = self.root.after(450, self._blink_tick)

    # ------------------------------------------------------- manuel seri seçimi
    def _crate_context_image(self, crate):
        """Kasayı ham kareden çevresiyle birlikte keser, kırmızı çerçeveler.

        YOLO kesiti bazen kaymış oluyor; ham kareden komşu kasalarla birlikte
        gösterildiğinde barkod/numara çok daha rahat seçiliyor. Ham kare yoksa
        dar kesite düşülür.
        """
        raw_path = crate.get("raw_path")
        bbox = crate.get("bbox")
        if raw_path and bbox and os.path.exists(raw_path):
            try:
                full = Image.open(raw_path).convert("RGB")
                x1, y1, x2, y2 = (int(value) for value in bbox)
                margin_x = int((x2 - x1) * 0.30)
                margin_y = int((y2 - y1) * 1.6)
                cx1 = max(0, x1 - margin_x)
                cy1 = max(0, y1 - margin_y)
                cx2 = min(full.width, x2 + margin_x)
                cy2 = min(full.height, y2 + margin_y)
                context = full.crop((cx1, cy1, cx2, cy2))
                draw = ImageDraw.Draw(context)
                line = max(3, (y2 - y1) // 14)
                draw.rectangle(
                    (x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1),
                    outline=(214, 77, 77),
                    width=line,
                )
                return context
            except OSError:
                pass
        crop_path = crate.get("crop_path")
        if crop_path and os.path.exists(crop_path):
            try:
                return Image.open(crop_path).convert("RGB")
            except OSError:
                pass
        return None

    def _on_crate_click(self, index):
        self._open_serial_modal(index)

    def _open_serial_modal(self, index):
        self._close_serial_modal()
        crate = self.pallet_crates[index]
        unread = crate["serial"] is None

        # Modal ana çerçeveye yerleşir; böylece sayım ekranındaki görüntüden
        # kasa seçildiğinde de aynı pencere açılabilir.
        self._modal_blocker = ctk.CTkFrame(
            self.main, fg_color=theme.IMAGE_BG, corner_radius=0
        )
        self._modal_blocker.place(relx=0, rely=0, relwidth=1, relheight=1)
        self._modal_blocker.bind("<Button-1>", lambda _event: None)

        card = ctk.CTkFrame(
            self.main,
            fg_color=theme.SURFACE,
            corner_radius=18,
            border_width=2,
            border_color=theme.RED if unread else theme.BORDER_ACTIVE,
        )
        card.place(relx=0.5, rely=0.5, anchor="center")
        self._modal_card = card

        ctk.CTkLabel(
            card,
            text="OKUNAMAYAN KASA — SERİYİ SEÇİN" if unread else "KASA İNCELE",
            text_color=theme.RED if unread else theme.GREEN,
            font=ctk.CTkFont(theme.FONT, 12, "bold"),
        ).pack(padx=28, pady=(18, 0))
        if crate["camera_index"] is not None:
            position_text = f"Kamera {crate['camera_index'] + 1} · Kasa {crate['kasa_no'] or index + 1}"
        else:
            position_text = f"Kasa {index + 1}"
        if not unread:
            suffix = ""
            if crate["manual"]:
                suffix = "  (elle)"
            elif crate.get("ocr"):
                suffix = "  (sistem tespiti)"
            position_text += f"   ·   Seri {crate['serial']}{suffix}"
        ctk.CTkLabel(
            card,
            text=position_text,
            text_color=theme.TEXT,
            font=ctk.CTkFont(theme.FONT, 19, "bold"),
        ).pack(padx=28, pady=(2, 8))

        # Kasanın görüntüsü: tam kareden, komşu kasalar da görünecek şekilde
        # çevresiyle kesilir ve ilgili kasa kırmızı çerçeveyle işaretlenir.
        # YOLO kesiti kötüyse bile operatör kasayı bağlamında görür.
        self._modal_photo = None
        image = self._crate_context_image(crate)
        if image is not None:
            try:
                max_width = min(760, int(self.root.winfo_width() * 0.74))
                max_height = int(self.root.winfo_height() * 0.36)
                scale = min(max_width / image.width, max_height / image.height)
                image = image.resize(
                    (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
                    Image.Resampling.LANCZOS,
                )
                self._modal_photo = ImageTk.PhotoImage(image)
            except OSError:
                self._modal_photo = None
        if self._modal_photo is not None:
            tk.Label(
                card, image=self._modal_photo, bg=theme.IMAGE_BG, bd=0
            ).pack(padx=24, pady=(0, 12))
        else:
            ctk.CTkLabel(
                card,
                text="Kasa görüntüsü yok",
                height=64,
                corner_radius=10,
                fg_color=theme.IMAGE_BG,
                text_color=theme.TEXT_MUTED,
                font=ctk.CTkFont(theme.FONT, 11),
            ).pack(fill="x", padx=24, pady=(0, 12))

        ctk.CTkLabel(
            card,
            text=(
                "Barkod parlama vb. nedenle okunamadıysa doğru seriyi seçin"
                if unread
                else "Seri yanlışsa aşağıdan düzeltebilirsiniz"
            ),
            text_color=theme.TEXT_SOFT,
            font=ctk.CTkFont(theme.FONT, 11),
        ).pack(padx=28, pady=(0, 6))

        grid = ctk.CTkFrame(card, fg_color="transparent")
        grid.pack(padx=24, pady=(0, 6))
        for position, serial in enumerate(self._serial_options()):
            ctk.CTkButton(
                grid,
                text=serial,
                width=150,
                height=70,
                corner_radius=14,
                fg_color=theme.SURFACE_LIGHT,
                hover_color=theme.BORDER_ACTIVE,
                text_color=theme.TEXT,
                border_width=2,
                border_color=theme.BORDER,
                font=ctk.CTkFont(theme.MONO, 26, "bold"),
                command=lambda value=serial: self._apply_manual_serial(index, value),
            ).grid(row=position // 3, column=position % 3, padx=6, pady=6)

        entry_row = ctk.CTkFrame(card, fg_color="transparent")
        entry_row.pack(fill="x", padx=24, pady=(2, 6))
        manual_entry = ctk.CTkEntry(
            entry_row,
            height=48,
            placeholder_text="Elle seri girin (örn. 6412)",
            fg_color=theme.BG_RAISED,
            border_color=theme.BORDER,
            font=ctk.CTkFont(theme.MONO, 18, "bold"),
        )
        manual_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))

        def apply_entry():
            value = manual_entry.get().strip()
            if len(value) == 4 and value.isdigit():
                self._apply_manual_serial(index, value)
            else:
                self._show_toast("⚠ Seri 4 haneli sayı olmalı (örn. 6412)", theme.AMBER)

        ctk.CTkButton(
            entry_row,
            text="Ekle",
            width=92,
            height=48,
            corner_radius=12,
            fg_color=theme.GREEN,
            hover_color=theme.GREEN_HOVER,
            text_color="#FFFFFF",
            font=ctk.CTkFont(theme.FONT, 14, "bold"),
            command=apply_entry,
        ).pack(side="right")
        manual_entry.bind("<Return>", lambda _event: apply_entry())

        if unread:
            # YOLO yanlış tespiti (parlama, forklift vb. kasa sanması) için:
            # kaydı sayımdan tamamen çıkarır, toplam kasa 1 azalır.
            ctk.CTkButton(
                card,
                text="🗑  Burada kasa yok — yanlış tespit",
                height=44,
                corner_radius=12,
                fg_color=theme.BG_RAISED,
                hover_color=theme.SURFACE_HOVER,
                text_color=theme.RED,
                border_width=1,
                border_color=theme.RED,
                font=ctk.CTkFont(theme.FONT, 12, "bold"),
                command=lambda: self._discard_crate(index),
            ).pack(fill="x", padx=24, pady=(2, 4))
        ctk.CTkButton(
            card,
            text="✕  Vazgeç" if unread else "✕  Kapat",
            height=48,
            corner_radius=12,
            fg_color=theme.BG_RAISED,
            hover_color=theme.SURFACE_HOVER,
            text_color=theme.TEXT_MUTED,
            border_width=1,
            border_color=theme.BORDER,
            font=ctk.CTkFont(theme.FONT, 14, "bold"),
            command=self._close_serial_modal,
        ).pack(fill="x", padx=24, pady=(2, 18))

    def _discard_crate(self, index):
        """YOLO'nun yanlış tespitini sayımdan çıkarır ("burada kasa yok")."""
        if index >= len(self.pallet_crates):
            return
        crate = self.pallet_crates.pop(index)
        self._discarded_count += 1
        self._close_serial_modal()
        self._ocr_selected = None
        self.metric_total.set(len(self.pallet_crates))
        self._sync_unread_metrics()
        self._refresh_ocr_list()
        self._refresh_pallet_view()
        cam = crate.get("camera_index")
        cam_text = f"K{cam + 1}" if cam is not None else "?"
        self.log(
            f"Palet: {cam_text} kasa {crate.get('kasa_no')} 'kasa yok' olarak "
            f"silindi; toplam {len(self.pallet_crates)}."
        )
        self._show_toast("🗑 Yanlış tespit sayımdan çıkarıldı")

    def _close_serial_modal(self):
        for widget_name in ("_modal_card", "_modal_blocker"):
            widget = getattr(self, widget_name, None)
            if widget is not None:
                try:
                    widget.destroy()
                except tk.TclError:
                    pass
                setattr(self, widget_name, None)

    def _apply_manual_serial(self, index, serial):
        crate = self.pallet_crates[index]
        crate["serial"] = str(serial)
        crate["manual"] = True
        crate["ocr"] = False  # operatör seçimi OCR etiketini geçersiz kılar
        self._close_serial_modal()
        self._sync_unread_metrics()
        self._refresh_ocr_list()
        self._refresh_pallet_view()
        self._show_toast(f"✓ Kasa {index + 1} → {serial} olarak işaretlendi")
        self.log(f"Palet: kasa {index + 1} manuel olarak {serial} seçildi.")

    # ------------------------------------------------------------ onay ve özet
    def _confirm_pallet(self):
        unread = self._unread_crate_count()
        if unread:
            self._show_toast(
                f"⚠ Önce {unread} okunamayan kasayı tamamlayın", theme.AMBER
            )
            return
        self._show_page("summary")

    def _build_summary_page(self):
        self.summary_body = ctk.CTkFrame(self.summary_page, fg_color="transparent")
        self.summary_body.pack(fill="both", expand=True, pady=(4, 0))

    def _series_counts(self):
        counts = {}
        for crate in self.pallet_crates:
            if crate["serial"]:
                counts[crate["serial"]] = counts.get(crate["serial"], 0) + 1
        return dict(sorted(counts.items()))

    def _build_summary_content(self):
        for child in self.summary_body.winfo_children():
            child.destroy()
        counts = self._series_counts()
        total = sum(counts.values())

        card = ctk.CTkFrame(
            self.summary_body,
            fg_color=theme.SURFACE,
            corner_radius=14,
            border_width=1,
            border_color=theme.BORDER,
        )
        card.pack(fill="both", expand=True)
        ctk.CTkLabel(
            card,
            text="PALET DAĞILIMI",
            text_color=theme.GREEN,
            font=ctk.CTkFont(theme.FONT, 11, "bold"),
        ).pack(pady=(18, 0))
        ctk.CTkLabel(
            card,
            text=f"{total} Kasa · {len(counts)} Seri",
            text_color=theme.TEXT,
            font=ctk.CTkFont(theme.FONT, 22, "bold"),
        ).pack(pady=(0, 2))
        manual_count = sum(1 for crate in self.pallet_crates if crate.get("manual"))
        if manual_count:
            ctk.CTkLabel(
                card,
                text=f"✎ {manual_count} kasa elle işaretlendi",
                text_color=theme.BLUE,
                font=ctk.CTkFont(theme.FONT, 11, "bold"),
            ).pack(pady=(0, 6))
        ocr_count = sum(1 for crate in self.pallet_crates if crate.get("ocr"))
        if ocr_count:
            ctk.CTkLabel(
                card,
                text=f"⚡ {ocr_count} kasa sistem tarafından tespit edildi",
                text_color=theme.CYAN,
                font=ctk.CTkFont(theme.FONT, 11, "bold"),
            ).pack(pady=(0, 6))
        session_text, session_color = self._session_summary_text()
        ctk.CTkLabel(
            card,
            text=f"▤ {session_text}",
            text_color=session_color,
            font=ctk.CTkFont(theme.FONT, 11, "bold"),
        ).pack(pady=(0, 6))
        if self.session is not None:
            remaining = self._session_remaining(include_current=True)
            if remaining is not None:
                if remaining < 0:
                    tail, tail_color = (
                        f"Bu palet eklenirse irsaliye {abs(remaining)} kasa aşılır",
                        theme.RED,
                    )
                elif remaining == 0:
                    tail, tail_color = (
                        "Bu palet eklenince irsaliye adedine tam ulaşılır",
                        theme.GREEN,
                    )
                else:
                    tail, tail_color = (
                        f"Bu palet eklenince {remaining} kasa kalır",
                        theme.TEXT_SOFT,
                    )
                ctk.CTkLabel(
                    card,
                    text=tail,
                    text_color=tail_color,
                    font=ctk.CTkFont(theme.FONT, 11),
                ).pack(pady=(0, 6))

        pie_size = 260
        pie = tk.Canvas(
            card,
            width=pie_size,
            height=pie_size,
            bg=theme.SURFACE,
            bd=0,
            highlightthickness=0,
        )
        pie.pack(pady=(4, 10))
        self._draw_pie(pie, pie_size, counts, total)

        list_frame = ctk.CTkFrame(card, fg_color="transparent")
        list_frame.pack(fill="x", padx=26, pady=(0, 12))
        for position, (serial, count) in enumerate(counts.items()):
            color = self.pallet_series_colors[
                position % len(self.pallet_series_colors)
            ]
            row = ctk.CTkFrame(
                list_frame,
                height=52,
                fg_color=theme.BG_RAISED,
                corner_radius=10,
                border_width=1,
                border_color=theme.BORDER,
            )
            row.pack(fill="x", pady=3)
            row.pack_propagate(False)
            ctk.CTkLabel(
                row, text="", width=14, height=14, corner_radius=7, fg_color=color
            ).pack(side="left", padx=(14, 10))
            ctk.CTkLabel(
                row,
                text=f"Seri {serial}",
                text_color=theme.TEXT,
                font=ctk.CTkFont(theme.MONO, 15, "bold"),
            ).pack(side="left")
            ctk.CTkLabel(
                row,
                text=f"%{count * 100 / total:.0f}" if total else "%0",
                text_color=theme.TEXT_MUTED,
                font=ctk.CTkFont(theme.FONT, 12),
            ).pack(side="right", padx=(0, 14))
            ctk.CTkLabel(
                row,
                text=f"{count} adet",
                text_color=theme.TEXT_SOFT,
                font=ctk.CTkFont(theme.FONT, 14, "bold"),
            ).pack(side="right", padx=12)

        buttons = ctk.CTkFrame(self.summary_body, fg_color="transparent", height=76)
        buttons.pack(fill="x", pady=(8, 0))
        buttons.pack_propagate(False)
        ctk.CTkButton(
            buttons,
            text="‹  Geri",
            width=118,
            height=56,
            corner_radius=14,
            fg_color=theme.SURFACE_LIGHT,
            hover_color=theme.BORDER_ACTIVE,
            text_color=theme.TEXT,
            font=ctk.CTkFont(theme.FONT, 14, "bold"),
            command=lambda: self._show_page("pallet"),
        ).pack(side="left", pady=10)
        ctk.CTkButton(
            buttons,
            text="TXT",
            width=76,
            height=56,
            corner_radius=14,
            fg_color=theme.BG_RAISED,
            hover_color=theme.SURFACE_HOVER,
            text_color=theme.TEXT_MUTED,
            border_width=1,
            border_color=theme.BORDER,
            font=ctk.CTkFont(theme.FONT, 12, "bold"),
            command=self._on_save_summary,
        ).pack(side="left", padx=8, pady=10)

        if self.session is None:
            # Süreç yokken palet eklenemez; operatör sevkiyat açmaya yönlendirilir.
            ctk.CTkButton(
                buttons,
                text="SEVKİYAT BAŞLAT  ›",
                width=250,
                height=56,
                corner_radius=14,
                fg_color=theme.BLUE,
                hover_color=theme.GREEN_HOVER,
                text_color="#FFFFFF",
                font=ctk.CTkFont(theme.FONT, 15, "bold"),
                command=lambda: self._show_page("session"),
            ).pack(side="right", pady=10)
            ctk.CTkLabel(
                buttons,
                text="Deneme modu — palet eklemek için önce sevkiyat açın",
                text_color=theme.AMBER,
                font=ctk.CTkFont(theme.FONT, 11, "bold"),
            ).pack(side="right", padx=12)
            return

        yazma = self._replace_target is not None
        ctk.CTkButton(
            buttons,
            text="PALETİ GÜNCELLE  ›" if yazma else "PALETİ EKLE ve DEVAM  ›",
            width=250,
            height=56,
            corner_radius=14,
            fg_color=theme.GREEN,
            hover_color=theme.GREEN_HOVER,
            text_color="#FFFFFF",
            font=ctk.CTkFont(theme.FONT, 15, "bold"),
            command=lambda: self._add_pallet_to_session(finish=False),
        ).pack(side="right", pady=10)
        ctk.CTkButton(
            buttons,
            text="GÜNCELLE ve BİTİR" if yazma else "EKLE ve SÜRECİ BİTİR",
            width=196,
            height=56,
            corner_radius=14,
            fg_color=theme.SURFACE_LIGHT,
            hover_color=theme.BORDER_ACTIVE,
            text_color=theme.TEXT,
            border_width=1,
            border_color=theme.BORDER_ACTIVE,
            font=ctk.CTkFont(theme.FONT, 13, "bold"),
            command=lambda: self._add_pallet_to_session(finish=True),
        ).pack(side="right", padx=8, pady=10)

    def _draw_pie(self, canvas, size, counts, total):
        if not total:
            return
        pad = 12
        start = 90.0
        for position, (serial, count) in enumerate(counts.items()):
            extent = -360.0 * count / total
            color = self.pallet_series_colors[
                position % len(self.pallet_series_colors)
            ]
            if len(counts) == 1:
                canvas.create_oval(
                    pad, pad, size - pad, size - pad, fill=color, outline=theme.SURFACE
                )
            else:
                canvas.create_arc(
                    pad, pad, size - pad, size - pad,
                    start=start, extent=extent,
                    fill=color, outline=theme.SURFACE, width=2,
                )
            start += extent
        # Ortadaki halka (donut görünümü)
        hole = size * 0.30
        canvas.create_oval(
            size / 2 - hole, size / 2 - hole,
            size / 2 + hole, size / 2 + hole,
            fill=theme.SURFACE, outline=theme.BORDER,
        )
        canvas.create_text(
            size / 2, size / 2 - 10,
            text=str(total), fill=theme.TEXT,
            font=(theme.FONT, 26, "bold"),
        )
        canvas.create_text(
            size / 2, size / 2 + 18,
            text="KASA", fill=theme.TEXT_MUTED,
            font=(theme.FONT, 10, "bold"),
        )

    @staticmethod
    def _desktop_dir():
        """Masaüstü klasörünü bulur (OneDrive yönlendirmesi dahil)."""
        desktop = Path.home() / "Desktop"
        if desktop.is_dir():
            return desktop
        onedrive = os.environ.get("OneDrive")
        if onedrive and (Path(onedrive) / "Desktop").is_dir():
            return Path(onedrive) / "Desktop"
        fallback = BASE_DIR / "kayitlar"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback

    def _on_save_summary(self):
        if not self.pallet_crates:
            self._show_toast("⚠ Kaydedilecek sayım yok", theme.AMBER)
            return
        now = datetime.now()
        counts = self._series_counts()
        total = sum(counts.values())
        barcode_count = sum(
            1
            for crate in self.pallet_crates
            if crate["serial"] and not crate.get("manual") and not crate.get("ocr")
        )
        ocr_count = sum(1 for crate in self.pallet_crates if crate.get("ocr"))
        manual_count = sum(1 for crate in self.pallet_crates if crate.get("manual"))
        unread = self._unread_crate_count()

        lines = [
            "TTO · Trento Toplu Okuma — Sayım Kaydı",
            f"Tarih : {now.strftime('%d.%m.%Y')}",
            f"Saat  : {now.strftime('%H:%M:%S')}",
            "-" * 42,
            f"TOPLAM KASA        : {total}",
        ]
        session = self.session
        if session is not None:
            lines += [
                f"İrsaliye no        : {session['waybill_no'] or '-'}",
                f"Plaka              : {session['plate'] or '-'}",
                f"İrsaliye adedi     : {session['expected'] or '-'}",
                f"Sevkiyatta bu palet: {len(session['pallets']) + 1}. palet",
                f"Önceki paletler    : {self._session_counted()} kasa",
            ]
        else:
            lines.append("Sevkiyat           : yok (deneme modu)")
        lines += ["", "Kasa cinsi dağılımı:"]
        for serial, count in counts.items():
            lines.append(f"  Seri {serial:<6}: {count:>3} adet")
        lines += [
            "-" * 42,
            f"Barkodu okunan     : {barcode_count}",
            f"Sistem tespiti(OCR): {ocr_count}",
            f"Elle işaretlenen   : {manual_count}",
            f"Hâlâ okunamayan    : {unread}",
        ]
        if self._discarded_count:
            lines.append(f"Silinen yanlış tespit: {self._discarded_count}")
        if self.last_aggregate:
            lines += ["-" * 42, "Kamera detayı:"]
            for cam in self.last_aggregate.get("per_camera", []):
                lines.append(
                    f"  K{cam['cam'] + 1}: {cam['kasa']} tespit · "
                    f"{cam['barkod']} barkod · {cam['okunamayan']} okunamayan"
                )
        lines.append("")

        file_path = self._desktop_dir() / f"TTO_sayim_{now.strftime('%Y%m%d_%H%M%S')}.txt"
        try:
            file_path.write_text("\n".join(lines), encoding="utf-8")
        except OSError as exc:
            self._show_toast("⚠ Kayıt yazılamadı", theme.RED)
            self.log(f"Kayıt HATASI: {exc}")
            return
        self._show_toast(f"✓ Kaydedildi: {file_path.name}")
        self.log(f"Sayım kaydı yazıldı: {file_path}")

    # ----------------------------------------------------------------- bildirim
    def _show_toast(self, text, color=theme.GREEN):
        if self._toast_job is not None:
            self.root.after_cancel(self._toast_job)
            self._toast_job = None
        if self._toast_label is not None:
            try:
                self._toast_label.destroy()
            except tk.TclError:
                pass
        self._toast_label = ctk.CTkLabel(
            self.main,
            text=f"   {text}   ",
            height=48,
            corner_radius=14,
            fg_color=color,
            text_color="#FFFFFF",
            font=ctk.CTkFont(theme.FONT, 13, "bold"),
        )
        self._toast_label.place(relx=0.5, rely=0.03, anchor="n")

        def hide():
            self._toast_job = None
            if self._toast_label is not None:
                try:
                    self._toast_label.destroy()
                except tk.TclError:
                    pass
                self._toast_label = None

        self._toast_job = self.root.after(2400, hide)

    # --------------------------------------------------------------- kapanış
    def _back_to_launcher(self):
        self.return_to_launcher = True
        self._close()

    def _close(self):
        self.preview_running = False
        if self._blink_job is not None:
            try:
                self.root.after_cancel(self._blink_job)
            except tk.TclError:
                pass
            self._blink_job = None
        try:
            self.manager.disconnect_all()
            self.manager.finalize_sdk()
        except Exception:
            pass
        close_detector = getattr(self.detector, "close", None)
        if callable(close_detector):
            try:
                close_detector()  # Aremak okuyucu + geçici BMP temizliği
            except Exception:
                pass
        self.root.destroy()

    def run(self):
        self.root.mainloop()
        return self.return_to_launcher


def relaunch_with_ocr_python():
    """paddleocr bu Python'da yoksa uygulamayı sistem Python 3.11 ile
    yeniden başlatır.

    Uygulama hangi yorumlayıcıyla açılırsa açılsın (VS Code, venv, kısayol)
    OCR'ın sessizce devre dışı kalmasını önler. Döngü emniyeti: yeniden
    başlatılan kopya TTO_NO_RELAUNCH=1 ile işaretlenir.
    """
    import importlib.util

    if importlib.util.find_spec("paddleocr") is not None:
        return
    if os.environ.get("TTO_NO_RELAUNCH") == "1":
        return
    required = "paddleocr,customtkinter,cv2,PIL,ultralytics,zxingcpp"
    probe_code = (
        "import importlib.util,sys;"
        f"sys.exit(0 if all(importlib.util.find_spec(m) for m in '{required}'.split(',')) else 1)"
    )
    try:
        probe = subprocess.run(
            ["py", "-3.11", "-c", probe_code],
            capture_output=True,
            timeout=60,
        )
    except Exception:
        return
    if probe.returncode != 0:
        return
    print(
        "[BILGI] paddleocr bu Python ortamında yok — uygulama OCR için "
        "sistem Python 3.11 ile yeniden başlatılıyor…"
    )
    env = dict(os.environ, TTO_NO_RELAUNCH="1")
    result = subprocess.run(
        ["py", "-3.11", str(Path(__file__).resolve())],
        env=env,
        cwd=str(BASE_DIR),
    )
    sys.exit(result.returncode)


def main():
    relaunch_with_ocr_python()
    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("green")

    while True:
        selected = ProductLauncher().run()
        if selected != "64":
            break
        go_back = TTOApplication().run()
        if not go_back:
            break


if __name__ == "__main__":
    main()
