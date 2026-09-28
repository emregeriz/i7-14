# -*- coding: utf-8 -*-
"""
Çoklu Kamera GUI — Aremak Barkod Okuyucu sürümü (2026-05-11)
============================================================
`multi_camera_gui.py` ile birebir aynı algoritmayı uygular
(3x Hikrobot kamera, YOLO ile kasa tespiti, CLAHE + keskinleştirme
ön-işleme, kasa kırpma, annotated + barcode_preprocess çıktıları,
çoklu kamera tekilleştirme/dedup). Tek fark: barkod okuma motoru
olarak `zxing-cpp` yerine `barkod_aremak/aremak_barkod` paketindeki
`AremakBarkodReader` kullanılır.

Çalıştırma:
    py Kamera_gui\\multi_camera_gui_20260511.py

Bağımlılıklar:
  - Hikrobot MVS SDK (kamera için)
  - Ultralytics + torch (YOLO için)
  - Aremak Kod Okuyucu kurulumu (`C:\\AremakCodeReaderCN`) + USB dongle
  - pythonnet, Pillow
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import traceback
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox

import cv2
import customtkinter as ctk
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

# ----------------------------------------------------------------------------
#  Import yolu ayarları
# ----------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Aremak paketi: <repo>/barkod_aremak/aremak_barkod
_REPO_ROOT = os.path.dirname(_HERE)
_AREMAK_PKG_ROOT = os.path.join(_REPO_ROOT, "barkod_aremak")
if _AREMAK_PKG_ROOT not in sys.path:
    sys.path.insert(0, _AREMAK_PKG_ROOT)

try:
    from aremak_barkod import AremakBarkodReader, AremakBarkodHatasi  # noqa: E402
    AREMAK_AVAILABLE = True
except ImportError as _ex:  # pragma: no cover
    AREMAK_AVAILABLE = False
    AremakBarkodReader = None  # type: ignore
    AremakBarkodHatasi = RuntimeError  # type: ignore
    print(f"[UYARI] aremak_barkod paketi import edilemedi: {_ex}")

# camera_gui_v2: SDK kurulumu, yardımcılar, ön-işleme ve diğer sabitler
import camera_gui_v2 as cv2gui  # noqa: E402
from camera_gui_v2 import (  # noqa: E402
    Colors,
    CONFIDENCE_THRESHOLD,
    CROP_EXT,
    HAS_YOLO,
    MVS_AVAILABLE,
    TARGET_CLASS_NAME,
    _candidate_yolo_model_paths,
    preprocess_crop_for_barcode,
    resolve_yolo_model_path,
)

# multi_camera_gui: SingleCamera + CameraPanel + CrateAggregator + MultiCamApp
import multi_camera_gui as mcg  # noqa: E402
from multi_camera_gui import MultiCamApp  # noqa: E402

# YOLO/torch: lazy benzeri import (camera_gui_v2 zaten yüklemiş olabilir)
try:
    from ultralytics import YOLO  # noqa: E402
    import torch  # noqa: E402
except ImportError:  # pragma: no cover
    YOLO = None
    torch = None


# ----------------------------------------------------------------------------
#  Aremak destekli işlemci
# ----------------------------------------------------------------------------
class AremakDetectionProcessor:
    """`camera_gui_v2.DetectionProcessor` ile aynı sözleşmeyi sunar.

    Algoritma birebir aynıdır; yalnızca barkod okuma `zxing-cpp` yerine
    `AremakBarkodReader.scan(path)` ile yapılır. Aremak ALGO motoru
    dosya yolundan okuduğu için ön-işlenmiş gri kasa görüntüsü geçici
    bir BMP dosyasına yazılıp taranır.
    """

    def __init__(self, install_dir: str | None = None):
        self.model = None
        self.model_names: dict = {}
        self._install_dir = install_dir
        self._reader: AremakBarkodReader | None = None
        self._reader_lock = threading.Lock()
        self._tmp_bmp_path: str | None = None
        # Başlatma teşhis durumu — kullanıcı tekrar denemek isterse her çağrıda yeniden dener.
        self._reader_init_error: str | None = None
        self._scan_error_logged_count = 0

    # ------------------------------------------------------------------
    def load_model(self, log_fn=None):
        """YOLO modelini ve Aremak okuyucusunu (lazy) hazırlar."""
        if not HAS_YOLO:
            raise Exception("Ultralytics/YOLO yüklü değil! pip install ultralytics")

        if self.model is None:
            if log_fn:
                log_fn("YOLO modeli yükleniyor...")
            model_path = resolve_yolo_model_path()
            if not model_path:
                aranan = "\n".join(f"  • {p}" for p in _candidate_yolo_model_paths())
                raise FileNotFoundError(
                    "YOLO .pt modeli bulunamadı. Dosyayı şu yollardan birine koyun:\n" + aranan
                )
            self.model = YOLO(model_path)
            if torch is not None and torch.cuda.is_available():
                self.model.to("cuda")
                if log_fn:
                    log_fn("GPU (CUDA) kullanılıyor.")
            else:
                if log_fn:
                    log_fn("CPU kullanılıyor.")
            self.model_names = getattr(self.model, "names", {})
            if log_fn:
                log_fn("Model yüklendi.")

        # Aremak okuyucu — her process_image çağrısında, henüz başarılı olmadıysa yeniden dene.
        if self._reader is None:
            if not AREMAK_AVAILABLE:
                if log_fn:
                    log_fn(
                        "[HATA] aremak_barkod paketi import edilemedi. "
                        "barkod_aremak/aremak_barkod klasörünü kontrol edin."
                    )
                return

            # Kurulum klasörü teşhisi
            install_path = Path(self._install_dir) if self._install_dir else Path(r"C:\AremakCodeReaderCN")
            algo_dll = install_path / "bin" / "Aremak.CodeReader.Algo.dll"
            if not algo_dll.is_file():
                msg = (
                    f"[HATA] Aremak DLL bulunamadı: {algo_dll}\n"
                    f"        Aremak Kod Okuyucu kurulu mu? (Klasör: {install_path})"
                )
                if log_fn:
                    log_fn("─" * 60)
                    log_fn(msg)
                    log_fn("─" * 60)
                self._reader_init_error = msg
                return

            try:
                if log_fn:
                    log_fn(f"Aremak okuyucu başlatılıyor... (kurulum: {install_path})")
                    log_fn("İlk taramada ~5sn warm-up olabilir, dongle takılı olmalı.")
                if self._install_dir:
                    self._reader = AremakBarkodReader(install_dir=self._install_dir)
                else:
                    self._reader = AremakBarkodReader()
                self._reader_init_error = None
                if log_fn:
                    log_fn("✔ Aremak barkod okuyucu hazır.")
            except AremakBarkodHatasi as ex:
                tb = traceback.format_exc()
                self._reader_init_error = str(ex)
                self._reader = None
                if log_fn:
                    log_fn("─" * 60)
                    log_fn(f"[HATA] Aremak başlatılamadı (AremakBarkodHatasi): {ex}")
                    log_fn("Olası nedenler: USB dongle takılı değil, lisans hatası,")
                    log_fn("  veya DLL bağımlılıkları (Visual C++ Runtime) eksik.")
                    for line in tb.splitlines():
                        log_fn("  " + line)
                    log_fn("─" * 60)
            except Exception as ex:
                tb = traceback.format_exc()
                self._reader_init_error = f"{type(ex).__name__}: {ex}"
                self._reader = None
                if log_fn:
                    log_fn("─" * 60)
                    log_fn(f"[HATA] Aremak beklenmedik hata ({type(ex).__name__}): {ex}")
                    for line in tb.splitlines():
                        log_fn("  " + line)
                    log_fn("─" * 60)

    # ------------------------------------------------------------------
    def _scan_with_aremak(self, image_np, log_fn=None):
        """Verilen numpy görüntüyü (gri veya BGR) geçici BMP'ye yazıp Aremak ile tarar."""
        if self._reader is None or image_np is None:
            return []
        if getattr(image_np, "size", 0) == 0:
            return []
        with self._reader_lock:
            try:
                if self._tmp_bmp_path is None:
                    fd, tmp = tempfile.mkstemp(prefix="aremak_kasa_", suffix=".bmp")
                    os.close(fd)
                    self._tmp_bmp_path = tmp
                # OpenCV BMP'yi uzantıdan tanır; gri veya BGR fark etmez.
                ok = cv2.imwrite(self._tmp_bmp_path, image_np)
                if not ok:
                    if log_fn and self._scan_error_logged_count < 3:
                        log_fn(f"  [HATA] Geçici BMP yazılamadı: {self._tmp_bmp_path}")
                        self._scan_error_logged_count += 1
                    return []
                return list(self._reader.scan(self._tmp_bmp_path))
            except AremakBarkodHatasi as ex:
                if log_fn and self._scan_error_logged_count < 3:
                    log_fn(f"  [HATA] Aremak tarama hatası: {ex}")
                    self._scan_error_logged_count += 1
                return []
            except Exception as ex:
                if log_fn and self._scan_error_logged_count < 3:
                    log_fn(f"  [HATA] Tarama sırasında beklenmedik hata ({type(ex).__name__}): {ex}")
                    self._scan_error_logged_count += 1
                return []

    # ------------------------------------------------------------------
    def process_image(self, img_bgr, save_dir, log_fn=None):
        """`DetectionProcessor.process_image` ile birebir aynı çıktı sözleşmesi.

        Tek fark: barkod kısmında `zx.read_barcodes(np)` yerine
        `AremakBarkodReader.scan(path)` çağrılır.
        """
        self.load_model(log_fn)

        def out(msg):
            if log_fn:
                log_fn(msg)

        os.makedirs(save_dir, exist_ok=True)
        crops_dir = os.path.join(save_dir, "crops")
        os.makedirs(crops_dir, exist_ok=True)

        device = 0 if (torch is not None and torch.cuda.is_available()) else "cpu"

        out("YOLO tespiti yapılıyor...")
        results = self.model.predict(
            source=img_bgr, save=False, conf=CONFIDENCE_THRESHOLD, device=device
        )

        if not results:
            out("YOLO sonuç döndürmedi!")
            return {
                "toplam_kasa": 0,
                "okunan_barkod": 0,
                "okunamayan_barkod": 0,
                "annotated_path": None,
                "annotated_image": None,
                "barcode_preprocess_image": None,
                "barcode_preprocess_path": None,
                "barkod_bulunamayan_isimler": [],
                "barkod_bulunamayan_yollar": [],
                "kasalar": [],
                "image_width": int(img_bgr.shape[1]) if img_bgr is not None else 0,
                "image_height": int(img_bgr.shape[0]) if img_bgr is not None else 0,
                "barkod_icerikler": [],
            }

        result = results[0]
        crate_count = 0
        boxes = getattr(result, "boxes", None)
        if boxes is not None and getattr(boxes, "cls", None) is not None:
            cls_array = boxes.cls.cpu().numpy()
            if TARGET_CLASS_NAME:
                for cls_idx in cls_array:
                    if str(self.model_names.get(int(cls_idx), "")).lower() == TARGET_CLASS_NAME.lower():
                        crate_count += 1
            else:
                crate_count = len(cls_array)

        out(f"Tespit edilen kasa sayısı: {crate_count}")

        toplam_barkod = 0
        barkod_bulunamayan_isimler: list[str] = []
        barkod_bulunamayan_yollar: list[str] = []
        box_barkod_durumu: dict[int, bool] = {}
        barkod_icerikler: list[str] = []
        kasalar: list[dict] = []

        orig_img = getattr(result, "orig_img", img_bgr)
        if len(orig_img.shape) == 3 and orig_img.shape[2] >= 3:
            g0 = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
        else:
            g0 = np.asarray(orig_img, dtype=np.uint8)
        barcode_vis = cv2.cvtColor(g0, cv2.COLOR_GRAY2BGR)

        annotated = orig_img.copy()
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        if boxes is not None and hasattr(boxes, "xyxy"):
            xyxy = boxes.xyxy.cpu().numpy()
            for i, (x1, y1, x2, y2) in enumerate(xyxy):
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                h, w = orig_img.shape[:2]
                x1, x2 = max(0, x1), min(w, x2)
                y1, y2 = max(0, y1), min(h, y2)
                if x2 > x1 and y2 > y1:
                    crop_color = orig_img[y1:y2, x1:x2].copy()
                    proc_hi = preprocess_crop_for_barcode(crop_color)
                    pw, ph = x2 - x1, y2 - y1
                    if proc_hi is not None:
                        fit = cv2.resize(proc_hi, (pw, ph), interpolation=cv2.INTER_AREA)
                        barcode_vis[y1:y2, x1:x2] = cv2.cvtColor(fit, cv2.COLOR_GRAY2BGR)
                    else:
                        fit = cv2.resize(
                            cv2.cvtColor(crop_color, cv2.COLOR_BGR2GRAY),
                            (pw, ph),
                            interpolation=cv2.INTER_AREA,
                        )
                        barcode_vis[y1:y2, x1:x2] = cv2.cvtColor(fit, cv2.COLOR_GRAY2BGR)

                    kasa_no = i + 1
                    crop_name = f"{timestamp_str}_kasa_{kasa_no}{CROP_EXT}"
                    barkod_okundu = True
                    icerikler: list[str] = []

                    if self._reader is not None:
                        scan_src = (
                            proc_hi
                            if proc_hi is not None
                            else cv2.cvtColor(crop_color, cv2.COLOR_BGR2GRAY)
                        )
                        codes = self._scan_with_aremak(scan_src, log_fn=out)
                        n_barcode = len(codes)
                        if n_barcode == 0:
                            barkod_okundu = False
                            crop_name = f"OKUNAMADI_{timestamp_str}_kasa_{kasa_no}{CROP_EXT}"
                            crop_full_path = os.path.join(crops_dir, crop_name)
                            barkod_bulunamayan_isimler.append(crop_name)
                            barkod_bulunamayan_yollar.append(crop_full_path)
                            out(f"  Barkod bulunamadı: Kasa #{kasa_no}")
                        else:
                            toplam_barkod += n_barcode
                            icerikler = [str(c.content) for c in codes]
                            barkod_icerikler.extend(icerikler)
                            out(f"  Kasa #{kasa_no} barkod: {n_barcode} adet → {icerikler}")
                    else:
                        # Aremak okuyucu yüklenememiş — sebep yukarıdaki [HATA] satır(lar)ında.
                        if self._reader_init_error:
                            out(
                                f"  Kasa #{kasa_no}: Barkod atlandı (Aremak başlatılamadı: "
                                f"{self._reader_init_error[:120]})"
                            )
                        else:
                            out(
                                f"  Kasa #{kasa_no}: Barkod atlandı (Aremak okuyucu henüz başlatılmadı)"
                            )

                    box_barkod_durumu[i] = barkod_okundu
                    kasalar.append(
                        {
                            "kasa_no": kasa_no,
                            "bbox": (x1, y1, x2, y2),
                            "barkodlar": list(icerikler),
                            "barkod_okundu": barkod_okundu,
                        }
                    )
                    crop_path = os.path.join(crops_dir, crop_name)
                    cv2.imwrite(crop_path, crop_color)
            if len(xyxy) == 0:
                pf0 = preprocess_crop_for_barcode(orig_img)
                if pf0 is not None:
                    barcode_vis = cv2.cvtColor(pf0, cv2.COLOR_GRAY2BGR)
        else:
            pf = preprocess_crop_for_barcode(orig_img)
            if pf is not None:
                barcode_vis = cv2.cvtColor(pf, cv2.COLOR_GRAY2BGR)

        if boxes is not None and hasattr(boxes, "xyxy"):
            xyxy = boxes.xyxy.cpu().numpy()
            COLOR_OK = (255, 255, 0)
            COLOR_FAIL = (0, 0, 255)
            for i, (x1, y1, x2, y2) in enumerate(xyxy):
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                okundu = box_barkod_durumu.get(i, True)
                color = COLOR_OK if okundu else COLOR_FAIL
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

                kasa_no = i + 1
                numara = str(kasa_no)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2
                (tw, th), _ = cv2.getTextSize(numara, font, font_scale, thickness)
                tx = x2 - tw - 4
                ty = y1 + th + 4
                cv2.rectangle(
                    annotated,
                    (tx - 2, ty - th - 2),
                    (tx + tw + 2, ty + 2),
                    (255, 255, 255),
                    -1,
                )
                cv2.putText(
                    annotated, numara, (tx, ty), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA
                )

        annotated_path = os.path.join(save_dir, f"annotated_{timestamp_str}.jpg")
        cv2.imwrite(annotated_path, annotated)

        barcode_prep_path = os.path.join(save_dir, f"barcode_preprocess_{timestamp_str}.jpg")
        cv2.imwrite(barcode_prep_path, barcode_vis)

        out("")
        out("=" * 50)
        out("ÖZET (Aremak)")
        out("=" * 50)
        out(f"Toplam kasa: {crate_count}")
        out(f"Okunan barkod: {toplam_barkod}")
        out(f"Okunamayan barkod: {len(barkod_bulunamayan_isimler)}")
        if barkod_icerikler:
            out(f"Barkod içerikleri: {barkod_icerikler}")
        out("=" * 50)

        image_h, image_w = orig_img.shape[:2]
        return {
            "toplam_kasa": crate_count,
            "okunan_barkod": toplam_barkod,
            "okunamayan_barkod": len(barkod_bulunamayan_isimler),
            "annotated_path": annotated_path,
            "annotated_image": annotated,
            "barcode_preprocess_image": barcode_vis,
            "barcode_preprocess_path": barcode_prep_path,
            "barkod_bulunamayan_isimler": barkod_bulunamayan_isimler,
            "barkod_bulunamayan_yollar": barkod_bulunamayan_yollar,
            "kasalar": kasalar,
            "image_width": int(image_w),
            "image_height": int(image_h),
            "barkod_icerikler": list(barkod_icerikler),
        }

    # ------------------------------------------------------------------
    def close(self):
        if self._reader is not None:
            try:
                self._reader.close()
            except Exception:
                pass
            self._reader = None
        if self._tmp_bmp_path:
            try:
                Path(self._tmp_bmp_path).unlink(missing_ok=True)
            except Exception:
                pass
            self._tmp_bmp_path = None


# ----------------------------------------------------------------------------
#  Fotoğraf yükleme yardımcı widget'ları
# ----------------------------------------------------------------------------
class _UploadImagePanel(ctk.CTkFrame):
    """Yüklenen / annotated görüntüyü oranı koruyarak gösteren basit panel."""

    def __init__(self, master, title: str):
        super().__init__(
            master,
            fg_color=Colors.BG_CARD,
            corner_radius=10,
            border_width=1,
            border_color=Colors.BORDER,
        )
        ctk.CTkLabel(
            self,
            text=title,
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            text_color=Colors.TEXT_PRIMARY,
        ).pack(anchor="w", padx=10, pady=(8, 2))
        self.wrap = ctk.CTkFrame(
            self,
            fg_color=Colors.IMAGE_CANVAS,
            corner_radius=8,
            border_width=1,
            border_color=Colors.BORDER,
        )
        self.wrap.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.label = tk.Label(
            self.wrap,
            text="—",
            bg=Colors.IMAGE_CANVAS,
            fg=Colors.TEXT_MUTED,
            font=("Segoe UI", 12),
        )
        self.label.pack(fill="both", expand=True)
        self._photo_ref = None
        self._np_bgr = None

    def set_image(self, np_bgr):
        if np_bgr is None:
            return
        self._np_bgr = np_bgr
        try:
            if len(np_bgr.shape) == 2:
                rgb = cv2.cvtColor(np_bgr, cv2.COLOR_GRAY2RGB)
            else:
                rgb = cv2.cvtColor(np_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            self.wrap.update_idletasks()
            box_w = max(320, self.wrap.winfo_width())
            box_h = max(240, self.wrap.winfo_height())
            w, h = pil.size
            ratio = min(box_w / max(1, w), box_h / max(1, h), 1.0)
            pil = pil.resize(
                (max(1, int(w * ratio)), max(1, int(h * ratio))), Image.LANCZOS
            )
            self._photo_ref = ImageTk.PhotoImage(pil)
            self.label.configure(image=self._photo_ref, text="")
        except Exception:
            self.label.configure(image="", text="Önizleme oluşturulamadı")


# ----------------------------------------------------------------------------
#  Aremak destekli çoklu kamera uygulaması
# ----------------------------------------------------------------------------
class AremakMultiCamApp(MultiCamApp):
    """`MultiCamApp` ile aynı; ek olarak Aremak ile fotoğraf yükleme alanı sunar."""

    def __init__(self):
        super().__init__()
        self.detector = AremakDetectionProcessor()
        try:
            self.app.title("Çoklu Kamera + YOLO — Aremak Barkod (3x)")
        except Exception:
            pass

        # Fotoğraf yükleme durumu
        self._upload_save_root = os.path.join(_HERE, "images_upload_aremak")
        os.makedirs(self._upload_save_root, exist_ok=True)
        self._upload_result_window: ctk.CTkToplevel | None = None
        self._upload_last_dir: str | None = None

        # Toolbar'a "Fotoğraf Yükle" butonu enjekte et
        self._inject_upload_button()

        self.log(
            "Barkod okuma motoru: Aremak Kod Okuyucu (barkod_aremak/aremak_barkod)"
        )
        if not AREMAK_AVAILABLE:
            self.log(
                "[HATA] aremak_barkod paketi import edilemedi — barkod okuma devre dışı."
            )

        # Aremak okuyucuyu arka planda proaktif başlat ve gerçek bir TEST SCAN yap
        # (sadece nesne oluşturmak yetmez — lisans / dongle hatası ilk taramada görünür).
        threading.Thread(target=self._warmup_detector, daemon=True).start()

    def _log_dual(self, msg: str):
        """GUI log paneline + standart çıktıya birlikte yazar (terminale de düşsün)."""
        try:
            print(msg, flush=True)
        except Exception:
            pass
        try:
            self.log(msg)
        except Exception:
            pass

    def _warmup_detector(self):
        """Detector'ı (YOLO + Aremak) ön-yükler ve küçük bir test scan ile doğrular."""
        try:
            self._log_dual("Detector ön-yükleme başlatıldı (YOLO + Aremak)...")
            self.detector.load_model(log_fn=self._log_dual)

            if self.detector._reader is None:
                err = self.detector._reader_init_error or "bilinmiyor"
                self._log_dual("[HATA] Aremak okuyucu başlatılamadı.")
                self._log_dual(f"        Sebep: {err}")
                self._show_aremak_alert(
                    "Aremak başlatılamadı",
                    f"Aremak barkod okuyucu başlatılamadı.\n\nSebep:\n{err}\n\n"
                    "Log panelindeki [HATA] satırlarını kontrol edin.\n"
                    "Sık nedenler: USB dongle takılı değil, lisans hatası,\n"
                    "VC++ Runtime eksik veya kurulum eksik.",
                )
                return

            # Gerçek test scan: küçük bir BMP yazıp tarayalım, lisans aktif mi?
            try:
                self._log_dual("Aremak test taraması yapılıyor...")
                # 64x64 boş gri görüntü — gerçek barkod beklemiyoruz, sadece SDK
                # çalışıyor mu (lisans, runtime) anlamak için.
                test_img = np.full((64, 64), 200, dtype=np.uint8)
                fd, tmp = tempfile.mkstemp(prefix="aremak_warmup_", suffix=".bmp")
                os.close(fd)
                try:
                    cv2.imwrite(tmp, test_img)
                    codes = self.detector._reader.scan(tmp)
                    self._log_dual(
                        f"✔ Aremak hazır ve çalışıyor (test scan: {len(codes)} barkod)."
                    )
                finally:
                    try:
                        Path(tmp).unlink(missing_ok=True)
                    except Exception:
                        pass
            except Exception as ex:
                tb = traceback.format_exc()
                self._log_dual("─" * 60)
                self._log_dual(
                    f"[HATA] Aremak test taraması başarısız ({type(ex).__name__}): {ex}"
                )
                for line in tb.splitlines():
                    self._log_dual("  " + line)
                self._log_dual("─" * 60)
                # Reader'ı invalide et — diğer çağrılar da skip etsin
                self.detector._reader_init_error = f"{type(ex).__name__}: {ex}"
                try:
                    self.detector._reader.close()
                except Exception:
                    pass
                self.detector._reader = None
                self._show_aremak_alert(
                    "Aremak test taraması başarısız",
                    f"Test taraması sırasında hata:\n\n{type(ex).__name__}: {ex}\n\n"
                    "USB dongle takılı mı? Lisans aktif mi?\n"
                    "Detaylar log panelinde.",
                )
        except Exception as ex:
            self._log_dual(f"Detector ön-yükleme hatası: {ex}")
            for line in traceback.format_exc().splitlines():
                self._log_dual("  " + line)

    def _show_aremak_alert(self, title: str, body: str):
        """Ana thread'de bir uyarı pencere göster (background thread'den de çağrılabilir)."""
        def _do():
            try:
                messagebox.showerror(title, body, parent=self.app)
            except Exception:
                pass
        try:
            self.app.after(0, _do)
        except Exception:
            pass

    # ------------------------------------------------------------------
    def _inject_upload_button(self):
        """Mevcut toolbar'a yeni bir 'Fotoğraf Yükle' butonu ekler."""
        try:
            inner = self.connect_btn.master
        except Exception as ex:
            self.log(f"Yükle butonu eklenemedi (toolbar bulunamadı): {ex}")
            return
        self.upload_btn = ctk.CTkButton(
            inner,
            text="🖼 Fotoğraf Yükle & İşle",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            fg_color=Colors.ACCENT_PURPLE,
            hover_color=Colors.BTN_HOVER,
            corner_radius=8,
            height=36,
            width=200,
            command=self._on_upload_click,
        )
        self.upload_btn.pack(side="right", padx=8)

    def _on_upload_click(self):
        """Dosya seç → okunur → arka planda YOLO + Aremak ile işle."""
        path = filedialog.askopenfilename(
            parent=self.app,
            title="Fotoğraf seç",
            filetypes=[
                ("Görseller", "*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff"),
                ("Tüm Dosyalar", "*.*"),
            ],
        )
        if not path:
            return
        try:
            img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception as ex:
            img = None
            self.log(f"Görsel okuma hatası: {ex}")
        if img is None:
            self.log(f"Görsel okunamadı: {path}")
            return
        self.log(f"Yüklenen fotoğraf: {path}  ({img.shape[1]}x{img.shape[0]})")
        try:
            self.upload_btn.configure(state="disabled", text="⏳ İşleniyor…")
        except Exception:
            pass
        threading.Thread(
            target=self._process_uploaded_image,
            args=(path, img),
            daemon=True,
        ).start()

    def _process_uploaded_image(self, src_path: str, img):
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = os.path.join(self._upload_save_root, f"run_{ts}")
            os.makedirs(run_dir, exist_ok=True)
            self._upload_last_dir = run_dir

            raw_name = f"raw_{ts}_{os.path.basename(src_path)}"
            raw_path = os.path.join(run_dir, raw_name)
            try:
                cv2.imwrite(raw_path, img)
            except Exception:
                pass

            self.set_status("Fotoğraf işleniyor…", Colors.ACCENT_YELLOW)
            with self._gui_lock:
                result = self.detector.process_image(
                    img,
                    run_dir,
                    log_fn=lambda m: self._log_dual(f"[YÜKLEME] {m}"),
                )

            self.app.after(0, lambda: self._show_upload_result(src_path, img, result))
            self.set_status(
                f"{len(self.manager.cameras)} kamera bağlı"
                if self.manager.cameras
                else "Fotoğraf işlendi",
                Colors.ACCENT_GREEN,
            )
        except Exception as ex:
            self.log(f"[YÜKLEME] HATA: {ex}")
            self.log(traceback.format_exc())
            self.set_status("Hata", Colors.ACCENT_RED)
        finally:

            def _restore_btn():
                try:
                    self.upload_btn.configure(
                        state="normal", text="🖼 Fotoğraf Yükle & İşle"
                    )
                except Exception:
                    pass

            self.app.after(0, _restore_btn)

    def _show_upload_result(self, src_path: str, src_img, result: dict):
        """Sonuçları ayrı bir Toplevel pencerede gösterir; mevcut pencere yeniden kullanılır."""
        win = self._upload_result_window
        if win is None or not bool(win.winfo_exists()):
            win = ctk.CTkToplevel(self.app)
            win.title("Fotoğraf Yükleme Sonucu — Aremak")
            win.geometry("1280x880")
            win.minsize(960, 700)
            win.configure(fg_color=Colors.BG_DARK)
            self._upload_result_window = win
        else:
            for w in win.winfo_children():
                w.destroy()
            win.deiconify()
            win.lift()

        # Üst şerit
        header = ctk.CTkFrame(win, fg_color=Colors.BG_CARD, corner_radius=0, height=50)
        header.pack(fill="x")
        header.pack_propagate(False)
        ctk.CTkLabel(
            header,
            text=f"🖼  {os.path.basename(src_path)}",
            font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"),
            text_color=Colors.TEXT_PRIMARY,
        ).pack(side="left", padx=14)
        ctk.CTkLabel(
            header,
            text=f"{src_img.shape[1]}×{src_img.shape[0]}px",
            font=ctk.CTkFont(family="Consolas", size=11),
            text_color=Colors.TEXT_MUTED,
        ).pack(side="left", padx=(0, 14))
        ctk.CTkButton(
            header,
            text="📁 Çıktı klasörünü aç",
            width=170,
            fg_color=Colors.BTN_PRIMARY,
            hover_color=Colors.BTN_HOVER,
            command=self._open_upload_output_dir,
        ).pack(side="right", padx=10, pady=8)

        # Gövde
        body = ctk.CTkFrame(win, fg_color=Colors.BG_DARK)
        body.pack(fill="both", expand=True, padx=12, pady=10)
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)
        body.grid_rowconfigure(1, weight=0)
        body.grid_rowconfigure(2, weight=1)

        in_panel = _UploadImagePanel(body, "Yüklenen Görsel")
        in_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=(0, 6))
        in_panel.set_image(src_img)

        out_panel = _UploadImagePanel(body, "İşlenmiş Görsel (Aremak)")
        out_panel.grid(row=0, column=1, sticky="nsew", padx=(6, 0), pady=(0, 6))
        annotated = result.get("annotated_image")
        out_panel.set_image(annotated if annotated is not None else src_img)

        # İstatistik kartları
        stats = ctk.CTkFrame(
            body,
            fg_color=Colors.BG_CARD,
            corner_radius=10,
            border_width=1,
            border_color=Colors.BORDER,
        )
        stats.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        for i in range(4):
            stats.grid_columnconfigure(i, weight=1)

        def _stat(col, title, value, color):
            f = ctk.CTkFrame(
                stats,
                fg_color=Colors.BG_SURFACE,
                corner_radius=8,
                border_width=1,
                border_color=Colors.BORDER,
            )
            f.grid(row=0, column=col, padx=5, pady=8, sticky="ew")
            ctk.CTkLabel(
                f,
                text=title,
                font=ctk.CTkFont(family="Segoe UI", size=10),
                text_color=Colors.TEXT_SECONDARY,
            ).pack(pady=(6, 0))
            ctk.CTkLabel(
                f,
                text=str(value),
                font=ctk.CTkFont(family="Segoe UI", size=24, weight="bold"),
                text_color=color,
            ).pack(pady=(0, 6))

        icerikler = result.get("barkod_icerikler", []) or []
        _stat(0, "TOPLAM KASA", result.get("toplam_kasa", 0), Colors.ACCENT_GREEN)
        _stat(1, "OKUNAN BARKOD", result.get("okunan_barkod", 0), Colors.ACCENT_CYAN)
        _stat(2, "OKUNAMAYAN", result.get("okunamayan_barkod", 0), Colors.ACCENT_RED)
        _stat(3, "TEKİL BARKOD", len(set(icerikler)), Colors.ACCENT_BLUE)

        # Barkod listesi
        bc_card = ctk.CTkFrame(
            body,
            fg_color=Colors.BG_CARD,
            corner_radius=10,
            border_width=1,
            border_color=Colors.BORDER,
        )
        bc_card.grid(row=2, column=0, columnspan=2, sticky="nsew")
        ctk.CTkLabel(
            bc_card,
            text="📋 Okunan Barkodlar",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            text_color=Colors.TEXT_SECONDARY,
        ).pack(anchor="w", padx=10, pady=(8, 2))
        bc_text = tk.Text(
            bc_card,
            bg=Colors.TERMINAL_BG,
            fg=Colors.TERMINAL_FG,
            font=("Consolas", 11),
            bd=0,
            relief="flat",
            wrap="word",
            insertbackground=Colors.TERMINAL_FG,
        )
        bc_text.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        if icerikler:
            for i, bc in enumerate(icerikler, 1):
                bc_text.insert("end", f"{i:>3}. {bc}\n")
        else:
            bc_text.insert("end", "(Okunan barkod yok)")
        bc_text.configure(state="disabled")

    def _open_upload_output_dir(self):
        target = self._upload_last_dir or self._upload_save_root
        if os.path.isdir(target):
            try:
                os.startfile(target)
            except Exception as ex:
                self.log(f"Klasör açılamadı: {ex}")
            return
        self.log(f"Klasör bulunamadı: {target}")

    def _on_close(self):
        try:
            if self._upload_result_window is not None and bool(
                self._upload_result_window.winfo_exists()
            ):
                self._upload_result_window.destroy()
        except Exception:
            pass
        try:
            if hasattr(self.detector, "close"):
                self.detector.close()
        except Exception:
            pass
        super()._on_close()


def main():
    if not MVS_AVAILABLE:
        print(
            "[HATA] MVS SDK bulunamadı. Hikrobot MVS'i kurun ve"
            " MVCAM_COMMON_RUNENV ortam değişkeninin set olduğundan emin olun."
        )
        sys.exit(1)
    try:
        AremakMultiCamApp().run()
    except Exception as ex:
        print(f"[FATAL] {ex}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
