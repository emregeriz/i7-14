# -*- coding: utf-8 -*-
"""TTO sayım süre testi (barkodlu / barkodsuz) — TAŞINABİLİR PAKET.

TTO'nun sayım adımlarını (YOLO + Aremak/HALCON barkod + filtre + tekilleştirme
+ PaddleOCR) başka bir bilgisayarda süre ölçmek için tek başına çalıştırır.
TTO kaynak koduna bağımlı değildir; filtre/tekilleştirme kodu birebir kopyadır.

Kamera olmadan çalışır. Seçilen 1-6 görüntüde TTO'nun sayım adımlarını aynı
sırayla yapar ve her adımın süresini ölçer:

  1) görüntüyü diskten okuma   (sahada kameradan gelir)
  2) ham kare kaydı            (TTO her sayımda raw_*.jpg yazar)
  3) YOLO kasa tespiti         (TTO ile aynı model ve eşik)
  4) barkod okuma (isteğe bağlı), motor seçilir:
       Aremak  TTO gibi kasa kasa: ön işleme + geçici BMP + tarama (USB dongle)
       HALCON  kasa kasa, gri kesit bellekten find_data_code_2d
               (deneme lisansı: 01.10.2026'ya kadar)
     "6 kamera aynı anda" açıkken her kamera kendi okuyucusuyla PARALEL okunur
     (Aremak: 6 okuyucu, tek dongle; HALCON: 6 model + iç paralellik kapalı).
  5) arka plan / boy filtresi  (TTO'nun kendi kodu)
  6) kesit + işaretli görüntü kaydı
  7) tekilleştirme             (barkod açıkken; TTO'nun CrateAggregator'u)
  8) OCR (isteğe bağlı)        barkod açıkken yalnız barkodu okunamayan
                                kasalar, kapalıyken TÜM kasalar

Barkod kapalıyken kameralar arası tekilleştirme yapılamaz; toplam kasa
kameraların toplamıdır (ortak görülen kasalar çift sayılır).

    py -3.11 sure_testi.py      (ya da baslat.bat)
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys


def _relaunch_with_ocr_python():
    """paddleocr bu Python'da yoksa kendini sistem Python 3.11 ile yeniden açar.

    VS Code / venv ile açılınca OCR sessizce kullanılamıyordu (TTO app.py'deki
    relaunch_with_ocr_python ile aynı mantık; ağır importlardan ÖNCE çalışır).
    """
    if importlib.util.find_spec("paddleocr") is not None:
        return
    if os.environ.get("TTO_NO_RELAUNCH") == "1":
        return
    probe = "import importlib.util,sys;sys.exit(0 if importlib.util.find_spec('paddleocr') else 1)"
    try:
        if subprocess.run(["py", "-3.11", "-c", probe], capture_output=True,
                          timeout=60).returncode != 0:
            return
    except Exception:
        return
    print("[BILGI] paddleocr bu Python'da yok - sistem Python 3.11 ile yeniden aciliyor...")
    result = subprocess.run(
        ["py", "-3.11", os.path.abspath(__file__)],
        env=dict(os.environ, TTO_NO_RELAUNCH="1"),
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    _relaunch_with_ocr_python()

import re  # noqa: E402
import tempfile  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402
from tkinter import filedialog  # noqa: E402
import tkinter as tk  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import customtkinter as ctk  # noqa: E402
from PIL import Image, ImageTk  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "gorseller"
OUTPUT_DIR = BASE_DIR / "sonuclar"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from tto_filtre import ArkaPlanFiltresi  # noqa: E402  (TTO app.py filtresi, kopya)
from tto_tespit import (  # noqa: E402
    CONFIDENCE_THRESHOLD,
    preprocess_crop_for_barcode,
    resolve_yolo_model_path,
)
from tekillestirme import CrateAggregator  # noqa: E402  (TTO, kopya)
import ocr_reader  # noqa: E402
import theme  # noqa: E402
from widgets import MetricCard  # noqa: E402

MAX_IMAGES = 6
ENGINES = ("Kapalı", "Aremak", "HALCON")
# HALCON'un nadir uydurma okumalarını eler (barkod_halcon/README 3.2).
SERIAL_PATTERN = re.compile(r"^(64|43)\d{2}A\d{20}$")
AREMAK_BIN = Path(r"C:\AremakCodeReaderCN\bin")
STAGES = (
    ("okuma", "Görüntü okuma (diskten)"),
    ("ham_kayit", "Ham kare kaydı"),
    ("yolo", "YOLO kasa tespiti"),
    ("barkod", "Barkod okuma"),
    ("filtre", "Arka plan / boy filtresi"),
    ("kesit_kayit", "Kesit ve görüntü kaydı"),
    ("tekil", "Tekilleştirme"),
    ("ocr", "OCR"),
)


_Filter = ArkaPlanFiltresi


def make_fullscreen(window):
    window.update_idletasks()
    try:
        window.state("zoomed")
    except tk.TclError:
        window.geometry(
            f"{window.winfo_screenwidth()}x{window.winfo_screenheight()}+0+0"
        )


class ImageTile(ctk.CTkFrame):
    """Tek görüntünün küçük önizlemesi, kasa sayısı ve süresi."""

    def __init__(self, master, index):
        super().__init__(
            master,
            fg_color=theme.SURFACE,
            corner_radius=12,
            border_width=1,
            border_color=theme.BORDER,
        )
        self.index = index
        self._image_bgr = None
        self._photo = None
        # Hızlandırılmış modda önizleme küçük tutulur: 20 MP görüntünün her
        # <Configure>'da yeniden ölçeklenmesi ana iş parçacığında 1-4 sn yiyordu.
        self.light_preview = False

        self.title = ctk.CTkLabel(
            self,
            text=f"Görüntü {index + 1}",
            anchor="w",
            text_color=theme.TEXT,
            font=ctk.CTkFont(theme.FONT, 10, "bold"),
        )
        self.title.pack(fill="x", padx=10, pady=(6, 0))

        self.image_frame = ctk.CTkFrame(
            self, height=150, fg_color=theme.IMAGE_BG, corner_radius=8
        )
        self.image_frame.pack(fill="both", expand=True, padx=8, pady=4)
        self.image_frame.pack_propagate(False)
        self.image_label = tk.Label(
            self.image_frame,
            text="Görüntü bekleniyor",
            bg=theme.IMAGE_BG,
            fg=theme.TEXT_MUTED,
            font=(theme.FONT, 9),
        )
        self.image_label.pack(fill="both", expand=True)
        self.image_frame.bind("<Configure>", lambda _event: self._render())

        self.info = ctk.CTkLabel(
            self,
            text="—",
            anchor="w",
            text_color=theme.TEXT_MUTED,
            font=ctk.CTkFont(theme.FONT, 10, "bold"),
        )
        self.info.pack(fill="x", padx=10, pady=(0, 6))

    def set_file(self, path):
        self._path = path
        self._image_bgr = cv2.imread(str(path))
        self.title.configure(text=f"Görüntü {self.index + 1} · {Path(path).name}")
        self.info.configure(text="Hazır", text_color=theme.TEXT_MUTED)
        self.configure(border_color=theme.BORDER_ACTIVE)
        self._render()

    def clear(self):
        self._image_bgr = None
        self._photo = None
        self.title.configure(text=f"Görüntü {self.index + 1}")
        self.image_label.configure(image="", text="Görüntü bekleniyor")
        self.info.configure(text="—", text_color=theme.TEXT_MUTED)
        self.configure(border_color=theme.BORDER)

    def set_busy(self):
        self.info.configure(text="İşleniyor…", text_color=theme.AMBER)
        self.configure(border_color=theme.AMBER)

    def set_result(self, text, annotated):
        self.info.configure(text=text, text_color=theme.GREEN)
        self.configure(border_color=theme.BORDER_ACTIVE)
        if annotated is not None:
            self._image_bgr = self._preview(annotated) if self.light_preview else annotated
        self._render()

    @staticmethod
    def _preview(image_bgr, max_side=960):
        height, width = image_bgr.shape[:2]
        scale = max_side / max(height, width)
        if scale >= 1:
            return image_bgr
        return cv2.resize(image_bgr, (int(width * scale), int(height * scale)),
                          interpolation=cv2.INTER_AREA)

    def use_light_preview(self, enabled):
        was_light, self.light_preview = self.light_preview, enabled
        if enabled and self._image_bgr is not None:
            self._image_bgr = self._preview(self._image_bgr)
        elif was_light and getattr(self, "_path", None):
            self._image_bgr = cv2.imread(str(self._path))  # normal mod: tam boyut

    def set_error(self, message):
        self.info.configure(text=message, text_color=theme.RED)
        self.configure(border_color=theme.RED)

    def _render(self):
        if self._image_bgr is None:
            return
        try:
            rgb = cv2.cvtColor(self._image_bgr, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            width = max(120, self.image_frame.winfo_width() - 4)
            height = max(80, self.image_frame.winfo_height() - 4)
            scale = min(width / image.width, height / image.height)
            image = image.resize(
                (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
                Image.Resampling.BILINEAR,
            )
            self._photo = ImageTk.PhotoImage(image)
            self.image_label.configure(image=self._photo, text="")
        except Exception:
            pass


class TimingApp:
    def __init__(self):
        self.model = None
        self.device = "cpu"
        self.paths: list[str] = []
        self.tiles: list[ImageTile] = []
        self.stage_labels: dict[str, tuple[ctk.CTkLabel, ctk.CTkLabel]] = {}
        self._running = False
        self._start = 0.0
        self.ocr_ready = False
        # OCR KENDİ iş parçacığında yüklenir ve çalışır (TTO'daki gibi).
        self._ocr_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ocr")
        # "Hızlandırılmış" mod: görüntü okuma/kaydı için (cv2 GIL'i bırakır).
        self._io_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="io")
        # Aremak okuyucu: nesnesi dongle olmadan da oluşur, hata ilk taramada
        # çıkar — bu yüzden hazır sayılması için deneme taraması şart.
        # Kamera başına bir okuyucu/model (slot 0 = sıralı çalıştırmada kullanılan).
        self.aremak_readers: list = []
        self.barcode_ready = False          # Aremak (en az 1 okuyucu)
        self._ha = None
        self.halcon_models: list = []
        self.halcon_ready = False
        self._barcode_probe_running = False
        self._tmp_bmps = [
            os.path.join(tempfile.gettempdir(), f"tto_sure_testi_kasa_{i}.bmp")
            for i in range(MAX_IMAGES)
        ]

        self.root = ctk.CTk()
        self.root.title("TTO · Sayım Süre Testi")
        self.root.minsize(1200, 760)
        self.root.configure(fg_color=theme.BG)
        self.ocr_var = tk.BooleanVar(value=False)
        self.save_var = tk.BooleanVar(value=True)
        self.engine_var = tk.StringVar(value="Kapalı")
        self.parallel_var = tk.BooleanVar(value=False)
        self.fast_var = tk.BooleanVar(value=False)
        self.fullframe_var = tk.BooleanVar(value=False)
        self._build_ui()
        make_fullscreen(self.root)
        self.filter = _Filter(self.log)
        threading.Thread(target=self._load_model, daemon=True).start()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        toolbar = ctk.CTkFrame(
            self.root,
            height=96,
            fg_color=theme.BG_RAISED,
            corner_radius=0,
            border_width=1,
            border_color=theme.BORDER,
        )
        toolbar.pack(fill="x")
        toolbar.pack_propagate(False)

        title = ctk.CTkFrame(toolbar, fg_color="transparent")
        title.pack(side="left", padx=22, pady=12)
        ctk.CTkLabel(
            title,
            text="TRENTO TOPLU OKUMA",
            anchor="w",
            text_color=theme.GREEN,
            font=ctk.CTkFont(theme.FONT, 9, "bold"),
        ).pack(anchor="w")
        ctk.CTkLabel(
            title,
            text="Sayım Süre Testi",
            anchor="w",
            text_color=theme.TEXT,
            font=ctk.CTkFont(theme.FONT, 22, "bold"),
        ).pack(anchor="w")

        self.run_btn = ctk.CTkButton(
            toolbar,
            text="▶ ÇALIŞTIR",
            state="disabled",
            width=150,
            height=44,
            corner_radius=10,
            fg_color=theme.GREEN,
            hover_color=theme.GREEN_HOVER,
            text_color=theme.BG_RAISED,
            font=ctk.CTkFont(theme.FONT, 12, "bold"),
            command=self._start_run,
        )
        self.run_btn.pack(side="right", padx=(4, 22), pady=24)
        self.select_btn = ctk.CTkButton(
            toolbar,
            text="↑ 1–6 Görüntü Seç",
            height=44,
            corner_radius=10,
            fg_color=theme.SURFACE_LIGHT,
            hover_color=theme.BORDER_ACTIVE,
            text_color=theme.TEXT,
            font=ctk.CTkFont(theme.FONT, 11, "bold"),
            command=self._select_files,
        )
        self.select_btn.pack(side="right", padx=4, pady=24)

        options = ctk.CTkFrame(toolbar, fg_color="transparent")
        options.pack(side="right", padx=12)
        barcode_row = ctk.CTkFrame(options, fg_color="transparent")
        barcode_row.pack(anchor="w", pady=2)
        ctk.CTkLabel(
            barcode_row,
            text="Barkod:",
            text_color=theme.TEXT_SOFT,
            font=ctk.CTkFont(theme.FONT, 10, "bold"),
        ).pack(side="left", padx=(0, 6))
        self.engine_switch = ctk.CTkSegmentedButton(
            barcode_row,
            values=list(ENGINES),
            variable=self.engine_var,
            height=24,
            font=ctk.CTkFont(theme.FONT, 10, "bold"),
            selected_color=theme.GREEN,
            selected_hover_color=theme.GREEN_HOVER,
        )
        self.engine_switch.pack(side="left")
        self.engine_state = ctk.CTkLabel(
            barcode_row,
            text="",
            text_color=theme.TEXT_MUTED,
            font=ctk.CTkFont(theme.FONT, 9),
        )
        self.engine_state.pack(side="left", padx=(8, 0))
        self.barcode_retry = ctk.CTkButton(
            barcode_row,
            text="↻ Dongle'ı dene",
            width=110,
            height=22,
            corner_radius=6,
            fg_color=theme.SURFACE_LIGHT,
            hover_color=theme.BORDER_ACTIVE,
            text_color=theme.TEXT,
            font=ctk.CTkFont(theme.FONT, 9, "bold"),
            command=self._start_barcode_probe,
        )
        parallel_row = ctk.CTkFrame(options, fg_color="transparent")
        parallel_row.pack(anchor="w", pady=2)
        self.parallel_check = ctk.CTkCheckBox(
            parallel_row,
            text="6 kamera aynı anda (paralel barkod)",
            variable=self.parallel_var,
            text_color=theme.TEXT_SOFT,
            font=ctk.CTkFont(theme.FONT, 10, "bold"),
        )
        self.parallel_check.pack(side="left")
        # TTO'da OLMAYAN iyileştirmeler; kapalıyken ölçüm TTO ile birebir.
        self.fast_check = ctk.CTkCheckBox(
            parallel_row,
            text="Hızlandırılmış (TTO'dan farklı)",
            variable=self.fast_var,
            text_color=theme.AMBER,
            font=ctk.CTkFont(theme.FONT, 10, "bold"),
        )
        self.fast_check.pack(side="left", padx=(12, 0))
        # Tam kare: Aremak/HALCON'a 300 kesit yerine 6 tam kare verilir; kodlar
        # konumuyla YOLO kutusuna eşlenir, eşleşmeyen kutular bugünkü gibi kesit
        # kesit tekrar okunur. TTO'da yok; kapalıyken kod bugünkü gibi çalışır.
        self.fullframe_check = ctk.CTkCheckBox(
            parallel_row,
            text="Tam kare + okunmayan kesit",
            variable=self.fullframe_var,
            text_color=theme.AMBER,
            font=ctk.CTkFont(theme.FONT, 10, "bold"),
        )
        self.fullframe_check.pack(side="left", padx=(12, 0))
        self.ocr_check = ctk.CTkCheckBox(
            options,
            text="OCR dahil",
            variable=self.ocr_var,
            text_color=theme.TEXT_SOFT,
            font=ctk.CTkFont(theme.FONT, 10),
        )
        self.ocr_check.pack(anchor="w", pady=2)
        ctk.CTkCheckBox(
            options,
            text="Görüntüleri kaydet (TTO gibi)",
            variable=self.save_var,
            text_color=theme.TEXT_SOFT,
            font=ctk.CTkFont(theme.FONT, 10),
        ).pack(anchor="w", pady=2)

        self.status = ctk.CTkLabel(
            toolbar,
            text="●  Model yükleniyor…",
            width=190,
            height=36,
            corner_radius=10,
            fg_color=theme.SURFACE,
            text_color=theme.AMBER,
            font=ctk.CTkFont(theme.FONT, 10, "bold"),
        )
        self.status.pack(side="right", padx=10, pady=30)

        page = ctk.CTkFrame(self.root, fg_color=theme.BG)
        page.pack(fill="both", expand=True, padx=18, pady=14)

        # Üst sıra: büyük kronometre + metrikler
        top = ctk.CTkFrame(page, fg_color="transparent")
        top.pack(fill="x", pady=(0, 10))
        top.grid_columnconfigure(0, weight=2, uniform="top")
        for column in range(1, 5):
            top.grid_columnconfigure(column, weight=1, uniform="top")

        clock = ctk.CTkFrame(
            top,
            fg_color=theme.SURFACE,
            corner_radius=16,
            border_width=1,
            border_color=theme.BORDER,
        )
        clock.grid(row=0, column=0, sticky="nsew", padx=4)
        ctk.CTkLabel(
            clock,
            text="GEÇEN SÜRE",
            text_color=theme.TEXT_MUTED,
            font=ctk.CTkFont(theme.FONT, 10, "bold"),
        ).pack(anchor="w", padx=18, pady=(12, 0))
        self.clock_label = ctk.CTkLabel(
            clock,
            text="0.00 sn",
            text_color=theme.TEXT,
            font=ctk.CTkFont(theme.MONO, 44, "bold"),
        )
        self.clock_label.pack(anchor="w", padx=18, pady=(0, 10))

        self.metric_crates = MetricCard(top, "TOPLAM KASA", theme.GREEN, "▣")
        self.metric_barcodes = MetricCard(top, "TEKİL BARKOD", theme.CYAN, "▥")
        self.metric_unread = MetricCard(top, "OKUNAMAYAN", theme.RED, "!")
        self.metric_dropped = MetricCard(top, "ELENEN KUTU", theme.AMBER, "⌁")
        for column, card in enumerate(
            (self.metric_crates, self.metric_barcodes, self.metric_unread, self.metric_dropped),
            start=1,
        ):
            card.grid(row=0, column=column, sticky="nsew", padx=4)

        body = ctk.CTkFrame(page, fg_color="transparent")
        body.pack(fill="both", expand=True)
        body.grid_columnconfigure(0, weight=3)
        body.grid_columnconfigure(1, weight=2)
        body.grid_rowconfigure(0, weight=1)

        grid = ctk.CTkFrame(body, fg_color="transparent")
        grid.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        for column in range(3):
            grid.grid_columnconfigure(column, weight=1, uniform="tile")
        for row in range(2):
            grid.grid_rowconfigure(row, weight=1, uniform="tile")
        for index in range(MAX_IMAGES):
            tile = ImageTile(grid, index)
            tile.grid(row=index // 3, column=index % 3, sticky="nsew", padx=4, pady=4)
            self.tiles.append(tile)

        side = ctk.CTkFrame(
            body,
            fg_color=theme.SURFACE,
            corner_radius=14,
            border_width=1,
            border_color=theme.BORDER,
        )
        side.grid(row=0, column=1, sticky="nsew")
        ctk.CTkLabel(
            side,
            text="Adım adım süre",
            anchor="w",
            text_color=theme.TEXT,
            font=ctk.CTkFont(theme.FONT, 14, "bold"),
        ).pack(fill="x", padx=16, pady=(14, 6))
        table = ctk.CTkFrame(side, fg_color="transparent")
        table.pack(fill="x", padx=16)
        table.grid_columnconfigure(0, weight=1)
        rows = list(STAGES) + [("toplam", "TOPLAM")]
        for row, (key, label) in enumerate(rows):
            bold = key == "toplam"
            name = ctk.CTkLabel(
                table,
                text=label,
                anchor="w",
                text_color=theme.TEXT if bold else theme.TEXT_SOFT,
                font=ctk.CTkFont(theme.FONT, 12, "bold" if bold else "normal"),
            )
            name.grid(row=row, column=0, sticky="w", pady=3)
            value = ctk.CTkLabel(
                table,
                text="—",
                anchor="e",
                width=80,
                text_color=theme.TEXT,
                font=ctk.CTkFont(theme.MONO, 13, "bold" if bold else "normal"),
            )
            value.grid(row=row, column=1, sticky="e", pady=3)
            share = ctk.CTkLabel(
                table,
                text="",
                anchor="e",
                width=56,
                text_color=theme.TEXT_MUTED,
                font=ctk.CTkFont(theme.MONO, 11),
            )
            share.grid(row=row, column=2, sticky="e", pady=3)
            self.stage_labels[key] = (value, share)

        self.note_label = ctk.CTkLabel(
            side,
            text="",
            justify="left",
            anchor="w",
            text_color=theme.TEXT_MUTED,
            font=ctk.CTkFont(theme.FONT, 10),
        )
        self.note_label.pack(fill="x", padx=16, pady=(10, 6))
        self.device_label = ctk.CTkLabel(
            side,
            text="",
            anchor="w",
            text_color=theme.TEXT_MUTED,
            font=ctk.CTkFont(theme.FONT, 10),
        )
        self.device_label.pack(fill="x", padx=16)

        self.log_text = tk.Text(
            side,
            height=10,
            bg=theme.TERMINAL_BG,
            fg=theme.TERMINAL_TEXT,
            font=(theme.MONO, 9),
            bd=0,
            relief="flat",
            wrap="word",
            padx=8,
            pady=6,
        )
        self.log_text.pack(fill="both", expand=True, padx=12, pady=12)
        self.log_text.configure(state="disabled")

    # --------------------------------------------------------------- model
    def _load_model(self):
        try:
            # SIRA ÖNEMLİ: PaddleOCR, YOLO (torch) HERHANGİ bir tahmin yapmadan
            # ÖNCE yüklenmeli. Tersi olursa OCR kesit başına 0,2 sn yerine
            # 35-45 sn sürüyor (2026-09-22'de ölçüldü). OCR ~40 sn'de yüklenir.
            self.log("OCR modeli yükleniyor (~40 sn)…")
            self.ocr_ready = self._ocr_pool.submit(
                lambda: ocr_reader.get_ocr(log_fn=self.log) is not None
            ).result()
            if self.ocr_ready:
                self.log(f"OCR hazır (cihaz={ocr_reader.active_device()}).")
            else:
                reason = ocr_reader.availability_error() or "bilinmeyen hata"
                self.log(f"OCR KULLANILAMIYOR: {reason}")
                self.log(f"Python: {sys.executable} — kurulum.bat çalıştırıldı mı? "
                         "Uygulamayı baslat.bat ile açın.")
                self.root.after(0, self._disable_ocr_option)

            from ultralytics import YOLO
            import torch

            path = resolve_yolo_model_path()
            if not path:
                raise FileNotFoundError("YOLO .pt modeli bulunamadı")
            self.log(f"Model: {os.path.basename(path)}")
            self.model = YOLO(path)
            self.device = 0 if torch.cuda.is_available() else "cpu"
            device_text = "GPU (CUDA)" if self.device == 0 else "CPU"
            # Isınma: ilk tahmin model kurulumunu içerir, ölçüme girmesin
            # (TTO da modeli açılışta önceden yükler).
            warmup = self._any_image()
            if warmup is not None:
                self.model.predict(
                    source=cv2.imread(str(warmup)),
                    conf=CONFIDENCE_THRESHOLD,
                    device=self.device,
                    verbose=False,
                )
            self.log(f"YOLO hazır · cihaz: {device_text}")
            self.root.after(0, lambda: self.device_label.configure(
                text=f"YOLO cihazı: {device_text} · eşik {CONFIDENCE_THRESHOLD}"
            ))
            self.set_status("Hazır", theme.GREEN)
            self.root.after(0, self._update_run_button)
        except Exception as exc:
            self.log(f"HATA: model yüklenemedi: {exc}")
            self.set_status("Model hatası", theme.RED)
            return
        self._probe_halcon()
        self._probe_barcode()

    def _disable_ocr_option(self):
        self.ocr_var.set(False)
        self.ocr_check.configure(state="disabled", text="OCR kullanılamıyor (kayda bakın)")

    # ------------------------------------------------------------- barkod
    def _refresh_engine_state(self):
        if self.barcode_ready:
            aremak = f"hazır ({len(self.aremak_readers)} okuyucu)"
        elif self._barcode_probe_running:
            aremak = "sınanıyor…"
        else:
            aremak = "dongle yok"
        halcon = "hazır" if self.halcon_ready else "yok"
        self.engine_state.configure(text=f"Aremak: {aremak} · HALCON: {halcon}")
        if self.barcode_ready or self._barcode_probe_running:
            self.barcode_retry.pack_forget()
        else:
            self.barcode_retry.pack(side="left", padx=(8, 0))

    def _new_halcon_model(self):
        ha = self._ha
        model = ha.create_data_code_2d_model(
            "Data Matrix ECC 200", "default_parameters", "maximum_recognition")
        # Kodlarımız DİKDÖRTGEN Data Matrix: 'square' seçilirse aranmaz bile.
        ha.set_data_code_2d_param(model, ["symbol_shape", "polarity"], ["any", "any"])
        ha.set_data_code_2d_param(model, "timeout", 2000)
        return model

    def _probe_halcon(self):
        """HALCON Data Matrix modellerini kurar (kamera başına bir) ve dener."""
        try:
            import halcon as ha

            self._ha = ha
            models = [self._new_halcon_model() for _ in range(MAX_IMAGES)]
            sample = self._any_image()
            frame = cv2.imread(str(sample)) if sample is not None else None
            if frame is not None:
                gray = np.ascontiguousarray(
                    cv2.cvtColor(frame[:400, :2000], cv2.COLOR_BGR2GRAY))
                ha.find_data_code_2d(ha.himage_from_numpy_array(gray), models[0], [], [])
            self.halcon_models = models
            # Tam kare için ayrı model: 'maximum_recognition' + çok kod aramada
            # gerçek bir 20 MP karede segfault görüldü (barkod_halcon/README 3.1);
            # tam karede 'enhanced_recognition' kullanılır.
            self.halcon_ff_models = []
            for _ in range(MAX_IMAGES):
                m = ha.create_data_code_2d_model(
                    "Data Matrix ECC 200", "default_parameters", "enhanced_recognition")
                ha.set_data_code_2d_param(m, ["symbol_shape", "polarity"], ["any", "any"])
                ha.set_data_code_2d_param(m, "timeout", 5000)
                self.halcon_ff_models.append(m)
            self.halcon_ready = True
            self.log("HALCON barkod hazır (Data Matrix, maximum_recognition).")
        except Exception as exc:
            self.halcon_ready = False
            self.log(f"HALCON KULLANILAMIYOR: {str(exc).splitlines()[0][:160]}")
        self.root.after(0, self._refresh_engine_state)

    def _start_barcode_probe(self):
        if self._barcode_probe_running or self._running:
            return
        threading.Thread(target=self._probe_barcode, daemon=True).start()

    def _probe_barcode(self):
        """Aremak okuyucularını kurar (kamera başına bir) ve deneme taraması yapar.

        Okuyucu nesnesi dongle olmadan da oluşur; hata ilk taramada çıkar.
        """
        self._barcode_probe_running = True
        self.root.after(0, self._refresh_engine_state)
        try:
            from aremak_barkod import AremakBarkodReader

            sample = self._any_image()
            frame = cv2.imread(str(sample)) if sample is not None else None
            if frame is None:
                raise RuntimeError("deneme görüntüsü bulunamadı")
            height, width = frame.shape[:2]
            crop = preprocess_crop_for_barcode(frame[height // 3: height // 3 + 200, : width // 2])
            while len(self.aremak_readers) < MAX_IMAGES:
                slot = len(self.aremak_readers)
                reader = self.aremak_readers[slot] if slot < len(self.aremak_readers) else AremakBarkodReader()
                cv2.imwrite(self._tmp_bmps[slot], crop)
                started = time.perf_counter()
                reader.scan(self._tmp_bmps[slot])  # ilk tarama ısınma
                self.aremak_readers.append(reader)
                self.barcode_ready = True
                self.log(f"Aremak okuyucu {slot + 1}/{MAX_IMAGES} hazır "
                         f"(ısınma {time.perf_counter() - started:.1f} sn).")
                self.root.after(0, self._refresh_engine_state)
        except Exception as exc:
            reason = str(exc).splitlines()[0][:160]
            self.log(f"BARKOD KULLANILAMIYOR: {reason}")
            self.log("Aremak USB dongle takılı mı? Takıp '↻ Dongle'ı dene'ye basın.")
            if not self.aremak_readers:
                self.barcode_ready = False
        self._barcode_probe_running = False
        self.root.after(0, self._refresh_engine_state)

    def _scan_frame(self, frame_bgr, engine, slot):
        """Tam kareyi okur; [(kod, cx, cy), ...] döner (tam kare koordinatı)."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if engine == "HALCON":
            ha = self._ha
            try:
                xlds, _, codes = ha.find_data_code_2d(
                    ha.himage_from_numpy_array(np.ascontiguousarray(gray)),
                    self.halcon_ff_models[slot], ["stop_after_result_num"], [120])
            except Exception:
                return []
            out = []
            for i, code in enumerate(codes):
                if not SERIAL_PATTERN.match(str(code)):
                    continue
                _, row, col, _ = ha.area_center_xld(ha.select_obj(xlds, i + 1))
                out.append((str(code), float(col[0]), float(row[0])))
            return out
        # Aremak: tam kare gri BMP (ön işleme yok; kesitteki CLAHE tam kareye
        # uygulanmaz — önceki tam kare denemesi de düz gri ile 302/303 kod buldu)
        if not cv2.imwrite(self._tmp_bmps[slot], gray):
            return []
        try:
            return [(str(c.content), float(c.center_x), float(c.center_y))
                    for c in self.aremak_readers[slot].scan(self._tmp_bmps[slot])]
        except Exception:
            return []

    @staticmethod
    def _assign_codes(codes, crates):
        """Kod merkezini içeren kutuya yazar; birden çok kutu içeriyorsa merkeze
        en yakın olana. Kutuya düşmeyen kod sayısını döner."""
        orphan = 0
        for code, cx, cy in codes:
            best, best_d = None, None
            for kasa in crates:
                x1, y1, x2, y2 = kasa["bbox"]
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    d = abs(cx - (x1 + x2) / 2) + abs(cy - (y1 + y2) / 2)
                    if best is None or d < best_d:
                        best, best_d = kasa, d
            if best is None:
                orphan += 1
            elif code not in best["barkodlar"]:
                best["barkodlar"].append(code)
        return orphan

    def _scan_crate(self, crop_bgr, engine, slot):
        if engine == "HALCON":
            return self._scan_crate_halcon(crop_bgr, slot)
        return self._scan_crate_aremak(crop_bgr, slot)

    def _scan_crate_halcon(self, crop_bgr, slot):
        """Gri kesit bellekten verilir (BMP yok); kalıp süzgecinden geçer."""
        ha = self._ha
        gray = np.ascontiguousarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY))
        try:
            _, _, codes = ha.find_data_code_2d(
                ha.himage_from_numpy_array(gray), self.halcon_models[slot], [], [])
        except Exception:
            return []
        return [str(code) for code in codes if SERIAL_PATTERN.match(str(code))]

    def _scan_crate_aremak(self, crop_bgr, slot):
        """TTO'daki AremakDetectionProcessor ile aynı yol: ön işleme → BMP → tarama."""
        prepared = preprocess_crop_for_barcode(crop_bgr)
        if prepared is None:
            prepared = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        if not cv2.imwrite(self._tmp_bmps[slot], prepared):
            return []
        try:
            return [str(c.content) for c in self.aremak_readers[slot].scan(self._tmp_bmps[slot])]
        except Exception:
            return []

    @staticmethod
    def _any_image():
        """Isınma için gorseller/ altındaki ilk görüntü."""
        for path in sorted(IMAGES_DIR.glob("*.jpg")):
            return path
        return None

    # ---------------------------------------------------------------- akış
    def _select_files(self):
        paths = filedialog.askopenfilenames(
            parent=self.root,
            title="1-6 görüntü seçin (K1…K6 sırasıyla)",
            initialdir=str(IMAGES_DIR),
            filetypes=[
                ("Görüntüler", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                ("Tüm dosyalar", "*.*"),
            ],
        )
        if not paths:
            return
        self.paths = list(paths[:MAX_IMAGES])
        for tile in self.tiles:
            tile.clear()
        for index, path in enumerate(self.paths):
            self.tiles[index].set_file(path)
        self._reset_results()
        self.log(f"{len(self.paths)} görüntü seçildi.")
        self._update_run_button()

    def _update_run_button(self):
        ready = self.model is not None and bool(self.paths) and not self._running
        self.run_btn.configure(state="normal" if ready else "disabled")

    def _reset_results(self):
        self.clock_label.configure(text="0.00 sn", text_color=theme.TEXT)
        for card in (self.metric_crates, self.metric_barcodes,
                     self.metric_unread, self.metric_dropped):
            card.set(0)
        for value, share in self.stage_labels.values():
            value.configure(text="—")
            share.configure(text="")

    def _start_run(self):
        if self._running or not self.paths or self.model is None:
            return
        if self.ocr_var.get() and not self.ocr_ready:
            self.log("OCR henüz hazır değil; 'OCR dahil' kutusunu kapatın.")
            return
        engine = self.engine_var.get()
        if engine == "Aremak" and not self.barcode_ready:
            self.log("Aremak hazır değil (dongle?). '↻ Dongle'ı dene'ye basın "
                     "ya da HALCON seçin.")
            return
        if engine == "HALCON" and not self.halcon_ready:
            self.log("HALCON hazır değil; kayıttaki hataya bakın.")
            return
        with_barcode = None if engine == "Kapalı" else engine
        with_ocr = self.ocr_var.get()
        parallel = bool(with_barcode) and self.parallel_var.get()
        if parallel and engine == "Aremak" and len(self.aremak_readers) < len(self.paths):
            self.log(f"Aremak okuyucuları hazırlanıyor ({len(self.aremak_readers)}/"
                     f"{MAX_IMAGES}); birkaç saniye sonra tekrar deneyin.")
            return
        self._running = True
        self._reset_results()
        if with_barcode:
            mode = "6 kamera aynı anda" if parallel else "kameralar sırayla"
            note = (f"Barkod motoru: {with_barcode} ({mode}). Kameralar arası tekilleştirme\n"
                    "yapılır; OCR yalnız barkodu okunamayan kasalara uygulanır.")
        else:
            note = ("* Barkod kapalı: tekilleştirme yapılamaz; toplam,\n"
                    "kameraların toplamıdır. OCR tüm kasalara uygulanır.")
        self.note_label.configure(text=note)
        self.run_btn.configure(state="disabled", text="ÇALIŞIYOR…")
        self.select_btn.configure(state="disabled")
        self.engine_switch.configure(state="disabled")
        self.parallel_check.configure(state="disabled")
        self.fast_check.configure(state="disabled")
        self.fullframe_check.configure(state="disabled")
        for tile in self.tiles[: len(self.paths)]:
            tile.use_light_preview(self.fast_var.get())
            tile.info.configure(text="Sırada", text_color=theme.TEXT_MUTED)
        self.set_status("Çalışıyor", theme.AMBER)
        self._start = time.perf_counter()
        self._tick()
        threading.Thread(
            target=self._worker,
            args=(list(self.paths), with_barcode, with_ocr, self.save_var.get(), parallel,
                  self.fast_var.get(), self.fullframe_var.get()),
            daemon=True,
        ).start()

    def _tick(self):
        if not self._running:
            return
        self.clock_label.configure(
            text=f"{time.perf_counter() - self._start:.2f} sn", text_color=theme.AMBER
        )
        self.root.after(50, self._tick)

    def _worker(self, paths, with_barcode, with_ocr, save, parallel, fast=False,
                fullframe=False):
        """Üç aşama: (1) okuma+kayıt+YOLO sırayla, (2) barkod — sırayla ya da
        6 kamera aynı anda, (3) filtre+kayıt, ardından tekilleştirme ve OCR.

        fast: görüntüler paralel okunur, ham kare YOLO'yla eşzamanlı yazılır,
        kesitler paralel yazılır, paralel barkod ortak kuyruktan okunur."""
        times = {key: 0.0 for key, _ in STAGES}
        enabled = {"barkod": with_barcode, "tekil": with_barcode, "ocr": with_ocr}
        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = OUTPUT_DIR / run_stamp
        per_image = []
        results: list[dict | None] = [None] * len(paths)
        frames: list = [None] * len(paths)
        items: list[dict] = []
        total_dropped = 0
        summary = {}
        raw_writes = []
        try:
            prefetched = None
            if fast:
                t = time.perf_counter()
                prefetched = list(self._io_pool.map(cv2.imread, paths))
                times["okuma"] += time.perf_counter() - t

            # ---- 1) okuma + ham kayıt + YOLO (sırayla)
            for index, path in enumerate(paths):
                tile = self.tiles[index]
                self.root.after(0, tile.set_busy)
                own = 0.0

                t = time.perf_counter()
                frame = prefetched[index] if fast else cv2.imread(path)
                elapsed = time.perf_counter() - t
                times["okuma"] += elapsed
                own += elapsed
                if frame is None:
                    self.root.after(0, lambda tl=tile: tl.set_error("Görüntü okunamadı"))
                    self.log(f"G{index + 1}: görüntü okunamadı: {path}")
                    continue

                save_dir = run_dir / f"K{index + 1}"
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                if save:
                    t = time.perf_counter()
                    (save_dir / "crops").mkdir(parents=True, exist_ok=True)
                    raw_path = str(save_dir / f"raw_{timestamp}.jpg")
                    if fast:
                        # YOLO'yla eşzamanlı yazılır; bekleme aşama sonunda ölçülür
                        raw_writes.append(self._io_pool.submit(cv2.imwrite, raw_path, frame))
                    else:
                        cv2.imwrite(raw_path, frame)
                    elapsed = time.perf_counter() - t
                    times["ham_kayit"] += elapsed
                    own += elapsed

                t = time.perf_counter()
                prediction = self.model.predict(
                    source=frame,
                    conf=CONFIDENCE_THRESHOLD,
                    device=self.device,
                    verbose=False,
                )[0]
                elapsed = time.perf_counter() - t
                times["yolo"] += elapsed
                own += elapsed

                height, width = frame.shape[:2]
                crates = []
                for number, (x1, y1, x2, y2) in enumerate(
                    prediction.boxes.xyxy.cpu().numpy(), start=1
                ):
                    x1, y1 = max(0, int(x1)), max(0, int(y1))
                    x2, y2 = min(width, int(x2)), min(height, int(y2))
                    if x2 > x1 and y2 > y1:
                        crates.append({
                            "kasa_no": number,
                            "bbox": (x1, y1, x2, y2),
                            "barkodlar": [],
                            "barkod_okundu": False,
                        })
                items.append({
                    "index": index, "path": path, "frame": frame, "crates": crates,
                    "save_dir": save_dir, "timestamp": timestamp, "own": own,
                    "barcode_sec": 0.0,
                })
                self.root.after(0, lambda tl=tile, n=len(crates): tl.info.configure(
                    text=f"{n} kutu · barkod bekliyor" if with_barcode else f"{n} kutu",
                    text_color=theme.AMBER))
                self._show_times(times, time.perf_counter() - self._start, enabled)

            if raw_writes:
                t = time.perf_counter()
                for future in raw_writes:
                    future.result()
                times["ham_kayit"] += time.perf_counter() - t

            # ---- 2) barkod: TTO gibi filtreden ÖNCE, YOLO'nun her kutusunda
            if with_barcode and items:
                self._read_barcodes(items, with_barcode, parallel, fast, fullframe)

                # duvar saati: paralelde en uzun kamera, sıralıda toplam
                times["barkod"] += (max(i["barcode_sec"] for i in items) if parallel
                                    else sum(i["barcode_sec"] for i in items))
                self._show_times(times, time.perf_counter() - self._start, enabled)

            # ---- 3) filtre + kesit/görüntü kaydı
            for item in items:
                index, frame, crates = item["index"], item["frame"], item["crates"]
                save_dir, timestamp = item["save_dir"], item["timestamp"]
                height, width = frame.shape[:2]
                tile = self.tiles[index]

                t = time.perf_counter()
                annotated = frame.copy()
                read = sum(len(k["barkodlar"]) for k in crates)
                result = {
                    "toplam_kasa": len(crates),
                    "okunan_barkod": read,
                    "okunamayan_barkod": sum(1 for k in crates if not k["barkodlar"]),
                    "kasalar": crates,
                    "annotated_image": annotated,
                    "barkod_bulunamayan_isimler": [
                        f"x_kasa_{k['kasa_no']}." for k in crates if not k["barkodlar"]
                    ],
                    "barkod_bulunamayan_yollar": [],
                    "image_width": width,
                    "image_height": height,
                }
                raw_count = len(crates)
                self.filter._filter_background_crates(index, result, frame_height=height)
                kept = result["kasalar"]
                dropped = raw_count - len(kept)
                elapsed = time.perf_counter() - t
                times["filtre"] += elapsed
                own = item["own"] + item["barcode_sec"] + elapsed

                t = time.perf_counter()
                for kasa in kept:
                    x1, y1, x2, y2 = kasa["bbox"]
                    ok = kasa["barkodlar"] or not with_barcode
                    color = (0, 200, 0) if ok else (0, 0, 255)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 6)
                if save:
                    # TTO: her kasa için crops/ + review/ kesiti ve işaretli görüntü
                    review_dir = save_dir / "review"
                    review_dir.mkdir(parents=True, exist_ok=True)
                    writes = []
                    for kasa in kept:
                        x1, y1, x2, y2 = kasa["bbox"]
                        crop = frame[y1:y2, x1:x2]
                        prefix = "" if kasa["barkodlar"] or not with_barcode else "OKUNAMADI_"
                        review_path = review_dir / f"{timestamp}_kasa_{kasa['kasa_no']}.jpg"
                        writes.append((str(save_dir / "crops"
                                           / f"{prefix}{timestamp}_kasa_{kasa['kasa_no']}.jpg"),
                                       crop))
                        writes.append((str(review_path), crop))
                        kasa["tto_crop_path"] = str(review_path)
                    writes.append((str(save_dir / f"annotated_{timestamp}.jpg"), annotated))
                    if fast:
                        list(self._io_pool.map(lambda job: cv2.imwrite(*job), writes))
                    else:
                        for job in writes:
                            cv2.imwrite(*job)
                    elapsed = time.perf_counter() - t
                    times["kesit_kayit"] += elapsed
                    own += elapsed

                results[index] = result
                frames[index] = frame
                total_dropped += dropped
                unread = result["okunamayan_barkod"]
                per_image.append({
                    "goruntu": os.path.basename(item["path"]),
                    "kasa": len(kept),
                    "barkod": result["okunan_barkod"],
                    "okunamayan": unread if with_barcode else None,
                    "elenen": dropped,
                    "barkod_sn": round(item["barcode_sec"], 3) if with_barcode else None,
                    "sure_sn": round(own, 3),
                })
                text = f"{len(kept)} kasa"
                if with_barcode:
                    text += (f" · {result['okunan_barkod']} barkod · {unread} okunamayan"
                             f" · barkod {item['barcode_sec']:.2f} sn")
                if dropped:
                    text += f" · {dropped} elendi"
                text += f" · {own:.2f} sn"
                self.log(f"G{index + 1}: {text}")
                self.root.after(
                    0, lambda tl=tile, tx=text, a=annotated: tl.set_result(tx, a)
                )
                partial = sum(len(r["kasalar"]) for r in results if r)
                self.root.after(
                    0,
                    lambda c=partial, d=total_dropped: (
                        self.metric_crates.set(c), self.metric_dropped.set(d)
                    ),
                )
                self._show_times(times, time.perf_counter() - self._start, enabled)

            summary = self._finish_count(
                results, frames, run_dir, times, with_barcode, with_ocr
            )
            if with_barcode:
                summary["barkod_modu"] = "paralel" if parallel else "sirali"
            if with_barcode and fullframe:
                summary["tam_kare"] = {
                    "tam_karede_okunan": sum(i.get("ff_matched", 0) for i in items),
                    "kesitte_ek_okunan": sum(i.get("ff_extra", 0) for i in items),
                    "kutuya_dusmeyen_kod": sum(i.get("ff_orphan", 0) for i in items),
                }
            total = time.perf_counter() - self._start
            self._show_times(times, total, enabled)
            self._write_report(run_dir, run_stamp, paths, times, total, per_image,
                               with_barcode, with_ocr, save, summary, fast)
            self.set_status(f"Bitti · {total:.2f} sn", theme.GREEN)
            self.log(f"TOPLAM: {total:.2f} sn · {summary.get('kasa', 0)} kasa"
                     + ("" if with_barcode else " (tekilleştirme yok)"))
        except Exception as exc:
            self.set_status("Hata", theme.RED)
            self.log(f"HATA: {exc}")
            self.log(traceback.format_exc())
        finally:
            final = time.perf_counter() - self._start
            self._running = False

            def finish():
                self.clock_label.configure(text=f"{final:.2f} sn", text_color=theme.GREEN)
                self.run_btn.configure(text="▶ ÇALIŞTIR")
                self.select_btn.configure(state="normal")
                self.engine_switch.configure(state="normal")
                self.parallel_check.configure(state="normal")
                self.fast_check.configure(state="normal")
                self.fullframe_check.configure(state="normal")
                self._update_run_button()

            self.root.after(0, finish)

    def _read_barcodes(self, items, engine, parallel, fast=False, fullframe=False):
        """Her kameranın kutularını okur; item["barcode_sec"] kamera süresidir.

        fast + parallel: kameraya bağlı kalmadan tüm kasalar tek kuyruktan,
        büyükten küçüğe dağıtılır. Kasa sayısı az olan kamera erken bitip
        boş beklemez; okuyucu sayısı (6) ve lisans kullanımı aynı kalır.
        item["barcode_sec"] = o kameranın son kasasının bittiği an."""

        def read_camera(item, slot):
            started = time.perf_counter()
            frame = item["frame"]
            crates = item["crates"]
            if fullframe:
                # 1) tam kare → kodları kutulara eşle
                for kasa in crates:
                    kasa["barkodlar"] = []
                found = self._scan_frame(frame, engine, slot)
                item["ff_orphan"] = self._assign_codes(found, crates)
                item["ff_matched"] = sum(1 for k in crates if k["barkodlar"])
                item["ff_time"] = time.perf_counter() - started
                # 2) eşleşmeyen kutular bugünkü gibi kesit kesit
                extra = 0
                for kasa in crates:
                    if kasa["barkodlar"]:
                        continue
                    x1, y1, x2, y2 = kasa["bbox"]
                    kasa["barkodlar"] = self._scan_crate(frame[y1:y2, x1:x2], engine, slot)
                    extra += bool(kasa["barkodlar"])
                item["ff_extra"] = extra
                for kasa in crates:
                    kasa["barkod_okundu"] = bool(kasa["barkodlar"])
                item["barcode_sec"] = time.perf_counter() - started
                self.log(f"K{item['index'] + 1}: tam kare {len(found)} kod → "
                         f"{item['ff_matched']} kasa eşleşti ({item['ff_orphan']} kutu dışı), "
                         f"{len(crates) - item['ff_matched']} kesit tekrar okundu, "
                         f"+{extra} · tam kare {item['ff_time']:.2f} sn, toplam "
                         f"{item['barcode_sec']:.2f} sn")
                return
            for kasa in crates:
                x1, y1, x2, y2 = kasa["bbox"]
                codes = self._scan_crate(frame[y1:y2, x1:x2], engine, slot)
                kasa["barkodlar"] = codes
                kasa["barkod_okundu"] = bool(codes)
            item["barcode_sec"] = time.perf_counter() - started

        if not parallel:
            if engine == "HALCON":
                self._ha.set_system("parallelize_operators", "true")  # HALCON varsayılanı
            for item in items:
                read_camera(item, 0)
            return

        if engine == "HALCON":
            # Küçük kesitlerde HALCON'un iç paralelliği yavaşlatıyor (6,2 → 4,5 sn);
            # çekirdekleri kameralar arasında paylaştırmak daha verimli.
            self._ha.set_system("parallelize_operators", "false")
        # Aremak sarmalayıcısı her taramada cwd'yi bin'e alıp ESKİSİNE döndürüyor;
        # eşzamanlı taramalarda biri çıkarken diğerinin cwd'sini bozar. Aşama
        # boyunca cwd'yi bin'de sabitleyince herkesin "eskisi" de bin olur.
        previous_cwd = os.getcwd()
        pin_cwd = engine == "Aremak" and AREMAK_BIN.is_dir()
        if pin_cwd:
            os.chdir(AREMAK_BIN)
        try:
            if fast and fullframe:
                self._read_barcodes_fullframe_queue(items, engine)
            elif fast and not fullframe:
                self._read_barcodes_queue(items, engine)
            else:
                with ThreadPoolExecutor(max_workers=len(items)) as pool:
                    list(pool.map(read_camera, items, range(len(items))))
        finally:
            if pin_cwd:
                os.chdir(previous_cwd)

    def _read_barcodes_fullframe_queue(self, items, engine):
        """Tam kare + kesit, ORTAK havuz: 6 okuyucu önce tam kareleri alır; tam
        karesi biten okuyucu o kameranın eşleşmeyen kutularını havuza atar,
        boşa çıkan her okuyucu havuzdan kesit çeker. Işığı kötü tek kamerada
        biriken kesitler böylece 6 okuyucuya yayılır."""
        lock = threading.Lock()
        jobs = [("frame", item, None) for item in items]  # önce tam kareler
        started = time.perf_counter()
        for item in items:
            for kasa in item["crates"]:
                kasa["barkodlar"] = []
            item["barcode_sec"] = 0.0
            item["ff_extra"] = 0

        def worker(slot):
            while True:
                with lock:
                    if not jobs:
                        return
                    kind, item, kasa = jobs.pop(0)
                if kind == "frame":
                    found = self._scan_frame(item["frame"], engine, slot)
                    with lock:
                        item["ff_orphan"] = self._assign_codes(found, item["crates"])
                        item["ff_matched"] = sum(1 for k in item["crates"] if k["barkodlar"])
                        item["ff_time"] = time.perf_counter() - started
                        item["ff_found"] = len(found)
                        unread = [k for k in item["crates"] if not k["barkodlar"]]
                        jobs.extend(("crop", item, k) for k in unread)
                        item["barcode_sec"] = max(item["barcode_sec"], item["ff_time"])
                else:
                    x1, y1, x2, y2 = kasa["bbox"]
                    codes = self._scan_crate(item["frame"][y1:y2, x1:x2], engine, slot)
                    done = time.perf_counter() - started
                    with lock:
                        kasa["barkodlar"] = codes
                        item["ff_extra"] += bool(codes)
                        item["barcode_sec"] = max(item["barcode_sec"], done)

        with ThreadPoolExecutor(max_workers=len(items)) as pool:
            list(pool.map(worker, range(len(items))))
        for item in items:
            for kasa in item["crates"]:
                kasa["barkod_okundu"] = bool(kasa["barkodlar"])
            n = len(item["crates"])
            self.log(f"K{item['index'] + 1}: tam kare {item.get('ff_found', 0)} kod → "
                     f"{item.get('ff_matched', 0)} kasa eşleşti ({item.get('ff_orphan', 0)} kutu dışı), "
                     f"{n - item.get('ff_matched', 0)} kesit havuzda okundu, +{item['ff_extra']} · "
                     f"tam kare {item.get('ff_time', 0):.2f} sn, son kesit {item['barcode_sec']:.2f} sn")

    def _read_barcodes_queue(self, items, engine):
        jobs = []
        for item in items:
            item["barcode_sec"] = 0.0
            for kasa in item["crates"]:
                x1, y1, x2, y2 = kasa["bbox"]
                jobs.append(((x2 - x1) * (y2 - y1), item, kasa))
        jobs.sort(key=lambda job: -job[0])
        lock = threading.Lock()
        started = time.perf_counter()

        def worker(slot):
            while True:
                with lock:
                    if not jobs:
                        return
                    _, item, kasa = jobs.pop(0)
                x1, y1, x2, y2 = kasa["bbox"]
                codes = self._scan_crate(item["frame"][y1:y2, x1:x2], engine, slot)
                kasa["barkodlar"] = codes
                kasa["barkod_okundu"] = bool(codes)
                done = time.perf_counter() - started
                with lock:
                    item["barcode_sec"] = max(item["barcode_sec"], done)

        with ThreadPoolExecutor(max_workers=len(items)) as pool:
            list(pool.map(worker, range(len(items))))

    def _finish_count(self, results, frames, run_dir, times, with_barcode, with_ocr):
        """Tekilleştirme + OCR; özet sayıları döner ve metriklere yazar."""
        summary = {"kasa": sum(len(r["kasalar"]) for r in results if r)}
        # OCR hedefleri: [(etiket, [kesit yolları…])] — ilk başarılı kesit yeter
        targets = []
        if with_barcode:
            t = time.perf_counter()
            aggregate = CrateAggregator().aggregate(results)
            times["tekil"] += time.perf_counter() - t
            crates = aggregate["crates"]
            unread_groups = []
            for members in aggregate["groups"]:
                if any(crates[m]["barkodlar"] for m in members):
                    continue
                unread_groups.append(members)
            summary.update(
                kasa=aggregate["unique_total"],
                ham=aggregate["raw_total"],
                cakisan=aggregate["duplicates"],
                tekil_barkod=aggregate["unique_barcodes"],
                okunamayan=len(unread_groups),
            )
            for number, members in enumerate(unread_groups, start=1):
                paths = []
                for m in sorted(members, key=lambda i: crates[i]["cam"]):
                    crate = crates[m]
                    for kasa in results[crate["cam"]]["kasalar"]:
                        if kasa["kasa_no"] == crate["kasa_no"]:
                            paths.append(self._crop_path(kasa, frames[crate["cam"]],
                                                         run_dir, crate["cam"]))
                            break
                targets.append((f"kasa{number}", [p for p in paths if p]))
            self.log(
                f"Tekilleştirme: ham {aggregate['raw_total']} → {aggregate['unique_total']} "
                f"kasa ({aggregate['duplicates']} çakışan) · "
                f"{aggregate['unique_barcodes']} tekil barkod · {len(unread_groups)} okunamayan"
            )
        else:
            for cam, result in enumerate(results):
                if not result:
                    continue
                for kasa in result["kasalar"]:
                    path = self._crop_path(kasa, frames[cam], run_dir, cam)
                    targets.append((f"K{cam + 1}#{kasa['kasa_no']}", [path] if path else []))

        def apply():
            self.metric_crates.set(summary.get("kasa", 0))
            self.metric_barcodes.set(summary.get("tekil_barkod", 0))
            self.metric_unread.set(summary.get("okunamayan", 0))

        self.root.after(0, apply)

        if with_ocr and targets:
            t = time.perf_counter()
            solved = attempts = 0
            for _label, paths in targets:
                for path in paths:
                    attempts += 1
                    serial = self._ocr_pool.submit(
                        ocr_reader.read_crate_serial, path, "64"
                    ).result()
                    if serial:
                        solved += 1
                        break
            elapsed = time.perf_counter() - t
            times["ocr"] += elapsed
            summary.update(ocr_hedef=len(targets), ocr_cozulen=solved, ocr_deneme=attempts)
            self.log(f"OCR: {len(targets)} kasa, {attempts} kesit denendi, "
                     f"{solved} seri bulundu · {elapsed:.1f} sn")
        return summary

    def _crop_path(self, kasa, frame, run_dir, cam):
        """Kasanın kayıtlı kesiti; kayıt kapalıysa OCR için geçici kesit yazar."""
        path = kasa.get("tto_crop_path")
        if path:
            return path
        if frame is None:
            return None
        x1, y1, x2, y2 = kasa["bbox"]
        folder = run_dir / f"K{cam + 1}" / "review"
        folder.mkdir(parents=True, exist_ok=True)
        path = str(folder / f"ocr_{kasa['kasa_no']}.jpg")
        cv2.imwrite(path, frame[y1:y2, x1:x2])
        return path

    def _show_times(self, times, total, enabled):
        def apply():
            for key, _label in STAGES:
                value, share = self.stage_labels[key]
                if not enabled.get(key, True):
                    value.configure(text="kapalı")
                    share.configure(text="")
                    continue
                seconds = times[key]
                value.configure(text=f"{seconds:.2f} sn")
                share.configure(text=f"%{100 * seconds / total:.0f}" if total > 0 else "")
            value, share = self.stage_labels["toplam"]
            value.configure(text=f"{total:.2f} sn")
            share.configure(text="%100")

        self.root.after(0, apply)

    def _write_report(self, run_dir, stamp, paths, times, total, per_image,
                      with_barcode, with_ocr, save, summary, fast=False):
        summary = dict(summary)
        run_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "tarih": stamp,
            "yolo_cihazi": "gpu" if self.device == 0 else "cpu",
            "barkod_motoru": with_barcode or "kapalı",
            "barkod_modu": summary.get("barkod_modu"),
            "hizlandirilmis": fast,
            "tam_kare": summary.get("tam_kare") is not None,
            "ocr_dahil": with_ocr,
            "kayit_dahil": save,
            "toplam_sn": round(total, 3),
            "adimlar_sn": {key: round(value, 3) for key, value in times.items()},
            "ozet": summary,
            "goruntuler": per_image,
            "dosyalar": paths,
        }
        path = run_dir / "sure_raporu.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=1), "utf-8")
        self.log(f"Rapor: {path}")

    # ------------------------------------------------------------ yardımcı
    def set_status(self, text, color):
        self.root.after(
            0, lambda: self.status.configure(text=f"●  {text}", text_color=color)
        )

    def log(self, message):
        def append():
            self.log_text.configure(state="normal")
            self.log_text.insert(
                "end", f"[{datetime.now().strftime('%H:%M:%S')}]  {message}\n"
            )
            self.log_text.see("end")
            self.log_text.configure(state="disabled")

        try:
            self.root.after(0, append)
        except (RuntimeError, tk.TclError):
            print(message)

    def run(self):
        self.root.mainloop()


def main():
    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("green")
    TimingApp().run()


if __name__ == "__main__":
    main()
