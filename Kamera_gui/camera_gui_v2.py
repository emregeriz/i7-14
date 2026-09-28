# -*- coding: utf-8 -*-
"""
Entegre Kamera + YOLO + Barkod GUI Uygulaması — MODERN TASARIM (v2)
====================================================================
Hikrobot MVS SDK kameradan tek kare fotoğraf çeker,
YOLO ile kasa tespit eder ve barkodları sayar.

CustomTkinter ile premium dark-theme arayüz.
"""

import sys
import os
import traceback
import platform
import time
import threading
import numpy as np
from ctypes import *
from datetime import datetime

import customtkinter as ctk
import tkinter as tk
from PIL import Image, ImageTk

# ============================================================================
#  MVS SDK - import yolu ayarları
# ============================================================================
current_system = platform.system()
if current_system == "Windows":
    mvcam_path = os.getenv("MVCAM_COMMON_RUNENV")
    if mvcam_path:
        sys.path.append(os.path.join(mvcam_path, "Samples", "Python", "MvImport"))

    mvs_runtime_candidates = [
        r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64",
        r"C:\Program Files\Common Files\MVS\Runtime\Win64_x64",
        r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win32_i86",
    ]
    for runtime_dir in mvs_runtime_candidates:
        if os.path.isdir(runtime_dir):
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(runtime_dir)
            if runtime_dir not in os.environ.get("PATH", ""):
                os.environ["PATH"] = runtime_dir + os.pathsep + os.environ.get("PATH", "")

    mvimport_path = os.path.join(os.path.dirname(__file__), "..", "jetsonProject", "Python", "MvImport")
    if os.path.isdir(mvimport_path):
        sys.path.append(mvimport_path)
    alt_mvimport = r"C:\Users\Administrator\Documents\GitHub\jetsonProject\Python\MvImport"
    if os.path.isdir(alt_mvimport):
        sys.path.append(alt_mvimport)
else:
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "MvImport"))

# MVS SDK
try:
    from MvCameraControl_class import *
    from MvErrorDefine_const import *
    from CameraParams_header import *

    MVS_AVAILABLE = True
    try:
        MVCC_FLOATVALUE  # noqa: B018
    except NameError:

        class MVCC_FLOATVALUE(Structure):
            _fields_ = [
                ("fCurValue", c_float),
                ("fMax", c_float),
                ("fMin", c_float),
                ("nReserved", c_uint * 4),
            ]

except ImportError:
    MVS_AVAILABLE = False
    MVCC_FLOATVALUE = None
    print("[UYARI] MVS SDK bulunamadı. Kamera fonksiyonları devre dışı.")

# OpenCV
try:
    import cv2

    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    print("[UYARI] OpenCV bulunamadı. pip install opencv-python")

# YOLO
try:
    from ultralytics import YOLO
    import torch

    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False
    print("[UYARI] Ultralytics/YOLO bulunamadı. pip install ultralytics")

# Barkod okuma
try:
    import zxingcpp as zx

    BARCODE_AVAILABLE = True
except ImportError:
    BARCODE_AVAILABLE = False
    zx = None
    print("[UYARI] zxing-cpp bulunamadı. pip install zxing-cpp")


# ============================================================================
#  AYARLAR
# ============================================================================
_MODEL_BASENAME = "YOLOV11-1000IMG800SZ250EPCFULLAUGLAST (1).pt"


def _candidate_yolo_model_paths():
    """GUI klasörü ve repo kökünde .pt modelini ara."""
    _dir = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(_dir)
    return [
        os.path.join(_dir, "Models", _MODEL_BASENAME),
        os.path.join(_dir, _MODEL_BASENAME),
        os.path.join(repo, _MODEL_BASENAME),
        os.path.join(repo, "Models", _MODEL_BASENAME),
    ]


def resolve_yolo_model_path():
    # ODAI_YOLO_MODEL ortam değişkeni varsayılan modeli ezer (TTO farklı
    # model denemek için kullanır); dosya yoksa normal aramaya düşülür.
    override = os.environ.get("ODAI_YOLO_MODEL", "").strip()
    if override and os.path.isfile(override):
        return override
    for p in _candidate_yolo_model_paths():
        if os.path.isfile(p):
            return p
    return None


SAVE_DIR = os.path.join(os.path.dirname(__file__), "images")
CONFIDENCE_THRESHOLD = 0.65
TARGET_CLASS_NAME = None
CROP_EXT = ".jpg"


# ============================================================================
#  RENK PALETI (Açık tema: beyaz + Trento yeşili)
# ============================================================================
class Colors:
    BRAND_GREEN = "#2d6a45"
    BRAND_GREEN_DIM = "#245238"
    BRAND_GREEN_SOFT = "#3a8558"  # Başlıklar (açık zeminde)
    BG_DARK = "#f5faf7"  # Ana pencere — beyaza yakın hafif nane
    BG_CARD = "#ffffff"
    BG_SURFACE = "#e8f2ec"
    BG_INPUT = "#ffffff"
    IMAGE_CANVAS = "#e6ece9"  # tk görüntü alanı (fotoğraf çerçevesi)
    BORDER = "#b8d8c4"
    BORDER_LIGHT = "#d4ebe0"
    TEXT_PRIMARY = "#1a2e22"
    TEXT_SECONDARY = "#3d5c48"
    TEXT_MUTED = "#5a7568"
    ACCENT_BLUE = "#2a7ab0"
    ACCENT_CYAN = "#1f8a72"
    ACCENT_GREEN = "#20874a"
    ACCENT_RED = "#c42d2d"
    ACCENT_YELLOW = "#b8860b"
    ACCENT_PURPLE = "#6b5b95"
    BTN_PRIMARY = "#2d6a45"
    BTN_HOVER = "#3a8558"
    BTN_DANGER = "#b83232"
    BTN_BLUE = "#256d5a"
    BTN_BLUE_HOVER = "#328a72"
    TERMINAL_BG = "#f0f5f2"
    TERMINAL_FG = "#1a2e22"


_TRENTO_LOGO_FILENAMES = ("trentoLogo.png", "trentoLogo.jpg")


def resolve_trento_logo_path():
    """Şeffaf arka plan için trentoLogo.png (öncelik), yoksa .jpg. Kamera_gui → repo → üst klasör."""
    base = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(base)
    for folder in (base, repo, os.path.normpath(os.path.join(repo, ".."))):
        for name in _TRENTO_LOGO_FILENAMES:
            p = os.path.join(folder, name)
            if os.path.isfile(p):
                return p
    return None


def _hex_to_rgb_triplet(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.strip().lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def pil_trento_logo_to_rgb(pil_img: Image.Image, bg_hex: str) -> Image.Image:
    """RGBA şeffaflığını bg_hex rengine birleştirir; CTkImage için RGB (siyah kutu yok)."""
    rgba = pil_img.convert("RGBA")
    bg = Image.new("RGBA", rgba.size, _hex_to_rgb_triplet(bg_hex) + (255,))
    bg.paste(rgba, (0, 0), rgba)
    return bg.convert("RGB")


# ============================================================================
#  MVS SDK Yardımcı Fonksiyonlar
# ============================================================================


def to_hex(num):
    if num < 0:
        num = num + 2**32
    return "0x%X" % num


def decoding_char(ctypes_char_array):
    byte_str = memoryview(ctypes_char_array).tobytes()
    null_index = byte_str.find(b"\x00")
    if null_index != -1:
        byte_str = byte_str[:null_index]
    for encoding in ["utf-8", "gbk", "latin-1"]:
        try:
            return byte_str.decode(encoding)
        except UnicodeDecodeError:
            continue
    return byte_str.decode("latin-1", errors="replace")


def is_mono_pixel(pixel_type):
    mono_types = [
        PixelType_Gvsp_Mono8,
        PixelType_Gvsp_Mono10,
        PixelType_Gvsp_Mono10_Packed,
        PixelType_Gvsp_Mono12,
        PixelType_Gvsp_Mono12_Packed,
    ]
    return pixel_type in mono_types


def is_color_pixel(pixel_type):
    color_types = [
        PixelType_Gvsp_BayerGR8,
        PixelType_Gvsp_BayerRG8,
        PixelType_Gvsp_BayerGB8,
        PixelType_Gvsp_BayerBG8,
        PixelType_Gvsp_BayerGR10,
        PixelType_Gvsp_BayerRG10,
        PixelType_Gvsp_BayerGB10,
        PixelType_Gvsp_BayerBG10,
        PixelType_Gvsp_BayerGR12,
        PixelType_Gvsp_BayerRG12,
        PixelType_Gvsp_BayerGB12,
        PixelType_Gvsp_BayerBG12,
        PixelType_Gvsp_YUV422_Packed,
        PixelType_Gvsp_YUV422_YUYV_Packed,
    ]
    return pixel_type in color_types


def frame_to_numpy(cam, buf_data, frame_info):
    """Kamera verisini numpy array'e çevirir."""
    if is_mono_pixel(frame_info.enPixelType):
        if frame_info.enPixelType == PixelType_Gvsp_Mono8:
            image = np.frombuffer(buf_data, dtype=np.uint8)
            image = image.reshape(frame_info.nHeight, frame_info.nWidth)
            return image
        else:
            nConvertSize = frame_info.nWidth * frame_info.nHeight
            convert_buf = (c_ubyte * nConvertSize)()
            stConvertParam = MV_CC_PIXEL_CONVERT_PARAM()
            memset(byref(stConvertParam), 0, sizeof(stConvertParam))
            stConvertParam.nWidth = frame_info.nWidth
            stConvertParam.nHeight = frame_info.nHeight
            stConvertParam.pSrcData = cast(buf_data, POINTER(c_ubyte))
            stConvertParam.nSrcDataLen = frame_info.nFrameLen
            stConvertParam.enSrcPixelType = frame_info.enPixelType
            stConvertParam.enDstPixelType = PixelType_Gvsp_Mono8
            stConvertParam.pDstBuffer = convert_buf
            stConvertParam.nDstBufferSize = nConvertSize
            ret = cam.MV_CC_ConvertPixelType(stConvertParam)
            if ret != 0:
                return None
            image = np.frombuffer(convert_buf, dtype=np.uint8)
            image = image.reshape(frame_info.nHeight, frame_info.nWidth)
            return image

    elif is_color_pixel(frame_info.enPixelType):
        nConvertSize = frame_info.nWidth * frame_info.nHeight * 3
        convert_buf = (c_ubyte * nConvertSize)()
        stConvertParam = MV_CC_PIXEL_CONVERT_PARAM()
        memset(byref(stConvertParam), 0, sizeof(stConvertParam))
        stConvertParam.nWidth = frame_info.nWidth
        stConvertParam.nHeight = frame_info.nHeight
        stConvertParam.pSrcData = cast(buf_data, POINTER(c_ubyte))
        stConvertParam.nSrcDataLen = frame_info.nFrameLen
        stConvertParam.enSrcPixelType = frame_info.enPixelType
        stConvertParam.enDstPixelType = PixelType_Gvsp_BGR8_Packed
        stConvertParam.pDstBuffer = convert_buf
        stConvertParam.nDstBufferSize = nConvertSize
        ret = cam.MV_CC_ConvertPixelType(stConvertParam)
        if ret != 0:
            return None
        image = np.frombuffer(convert_buf, dtype=np.uint8)
        image = image.reshape(frame_info.nHeight, frame_info.nWidth, 3)
        return image

    return None


# ============================================================================
#  Kamera Yönetici Sınıfı
# ============================================================================


class CameraManager:
    """Hikrobot MVS SDK ile kamera bağlantı ve çekim yönetimi."""

    def __init__(self):
        self.cam = None
        self.device_list = None
        self.connected = False
        self.grabbing = False
        self.sdk_initialized = False

    def initialize_sdk(self):
        if not MVS_AVAILABLE:
            raise Exception("MVS SDK yüklenemedi!")
        MvCamera.MV_CC_Initialize()
        self.sdk_initialized = True

    def enum_devices(self):
        self.device_list = MV_CC_DEVICE_INFO_LIST()
        tlayer_type = (
            MV_GIGE_DEVICE
            | MV_USB_DEVICE
            | MV_GENTL_CAMERALINK_DEVICE
            | MV_GENTL_CXP_DEVICE
            | MV_GENTL_XOF_DEVICE
        )

        ret = MvCamera.MV_CC_EnumDevices(tlayer_type, self.device_list)
        if ret != 0:
            raise Exception("Cihaz taraması başarısız! ret = %s" % to_hex(ret))

        if self.device_list.nDeviceNum == 0:
            raise Exception("Hiç kamera bulunamadı!")

        info_list = []
        for i in range(self.device_list.nDeviceNum):
            dev_info = cast(self.device_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            if dev_info.nTLayerType == MV_GIGE_DEVICE or dev_info.nTLayerType == MV_GENTL_GIGE_DEVICE:
                model = decoding_char(dev_info.SpecialInfo.stGigEInfo.chModelName)
                nip1 = (dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xFF000000) >> 24
                nip2 = (dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00FF0000) >> 16
                nip3 = (dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000FF00) >> 8
                nip4 = dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000FF
                ip_str = "%d.%d.%d.%d" % (nip1, nip2, nip3, nip4)
                info_list.append("[%d] GigE | %s | IP: %s" % (i, model, ip_str))
            elif dev_info.nTLayerType == MV_USB_DEVICE:
                model = decoding_char(dev_info.SpecialInfo.stUsb3VInfo.chModelName)
                serial = decoding_char(dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber)
                info_list.append("[%d] USB3 | %s | SN: %s" % (i, model, serial))
            else:
                info_list.append("[%d] Diğer cihaz" % i)
        return info_list

    def connect(self, cam_index=0):
        if self.cam is not None:
            self.disconnect()

        self.cam = MvCamera()
        dev_info = cast(self.device_list.pDeviceInfo[cam_index], POINTER(MV_CC_DEVICE_INFO)).contents

        ret = self.cam.MV_CC_CreateHandle(dev_info)
        if ret != 0:
            raise Exception("Handle oluşturulamadı! ret = %s" % to_hex(ret))

        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            self.cam.MV_CC_DestroyHandle()
            self.cam = None
            raise Exception("Cihaz açılamadı! ret = %s" % to_hex(ret))

        if dev_info.nTLayerType in (MV_GIGE_DEVICE, MV_GENTL_GIGE_DEVICE):
            nPacketSize = self.cam.MV_CC_GetOptimalPacketSize()
            if int(nPacketSize) > 0:
                self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)

        self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        # 0=Off, 1=Once, 2=Continuous — başlangıçta otomatik pozlama
        self.cam.MV_CC_SetEnumValue("ExposureAuto", 2)
        self.connected = True
        self.start_grabbing()

    def start_grabbing(self):
        if not self.connected or self.cam is None:
            return
        if self.grabbing:
            return
        ret = self.cam.MV_CC_StartGrabbing()
        if ret == 0:
            self.grabbing = True
        else:
            print("[UYARI] Grabbing başlatılamadı! ret = %s" % to_hex(ret))

    def stop_grabbing(self):
        if self.grabbing and self.cam is not None:
            try:
                self.cam.MV_CC_StopGrabbing()
            except Exception:
                pass
            self.grabbing = False

    def read_exposure_time_us(self):
        """ExposureTime (µs) için min / güncel / max; desteklenmiyorsa None."""
        if not MVS_AVAILABLE or self.cam is None or not MVCC_FLOATVALUE:
            return None
        try:
            st = MVCC_FLOATVALUE()
            memset(byref(st), 0, sizeof(st))
            for key in ("ExposureTime", "ExposureTimeAbs"):
                ret = self.cam.MV_CC_GetFloatValue(key, byref(st))
                if ret == 0 and float(st.fMax) > float(st.fMin):
                    return {
                        "key": key,
                        "min": float(st.fMin),
                        "max": float(st.fMax),
                        "cur": float(st.fCurValue),
                    }
        except Exception:
            pass
        return None

    def set_exposure_auto(self, continuous=True):
        """continuous=True → ExposureAuto sürekli (2), False → kapalı (0)."""
        if not MVS_AVAILABLE or self.cam is None:
            return False
        try:
            mode = 2 if continuous else 0
            return self.cam.MV_CC_SetEnumValue("ExposureAuto", mode) == 0
        except Exception:
            return False

    def set_exposure_time_us(self, exposure_us):
        """Manuel pozlama (µs). Önce otomatik kapatılır."""
        if not MVS_AVAILABLE or self.cam is None:
            return False
        try:
            self.cam.MV_CC_SetEnumValue("ExposureAuto", 0)
            ret = self.cam.MV_CC_SetFloatValue("ExposureTime", float(exposure_us))
            if ret != 0:
                ret = self.cam.MV_CC_SetFloatValue("ExposureTimeAbs", float(exposure_us))
            return ret == 0
        except Exception:
            return False

    def capture_single_frame(self, timeout_sec=10):
        if not self.connected or self.cam is None:
            raise Exception("Kamera bağlı değil!")

        if not self.grabbing:
            self.start_grabbing()
            if not self.grabbing:
                raise Exception("Görüntü yakalama başlatılamadı!")

        stOutFrame = MV_FRAME_OUT()
        memset(byref(stOutFrame), 0, sizeof(stOutFrame))

        sdk_timeout_ms = min(timeout_sec * 1000, 5000)
        max_retries = max(1, timeout_sec // 5)

        for attempt in range(max_retries):
            memset(byref(stOutFrame), 0, sizeof(stOutFrame))
            ret = self.cam.MV_CC_GetImageBuffer(stOutFrame, int(sdk_timeout_ms))
            if ret == 0 and stOutFrame.pBufAddr is not None:
                frame_len = stOutFrame.stFrameInfo.nFrameLen
                buf_data = (c_ubyte * frame_len)()
                memmove(buf_data, stOutFrame.pBufAddr, frame_len)

                frame_info = MV_FRAME_OUT_INFO_EX()
                memmove(byref(frame_info), byref(stOutFrame.stFrameInfo), sizeof(MV_FRAME_OUT_INFO_EX))

                self.cam.MV_CC_FreeImageBuffer(stOutFrame)

                np_image = frame_to_numpy(self.cam, buf_data, frame_info)
                if np_image is None:
                    raise Exception("Görüntü formatı dönüştürülemedi!")

                if len(np_image.shape) == 2:
                    np_image = cv2.cvtColor(np_image, cv2.COLOR_GRAY2BGR)

                return np_image

        self.stop_grabbing()
        raise Exception(
            "Kare alınamadı! Kamera yanıt vermiyor. "
            "'Yeniden Başlat' butonunu kullanarak kamerayı yeniden bağlayın."
        )

    def disconnect(self):
        if self.cam is not None:
            self.stop_grabbing()
            try:
                self.cam.MV_CC_CloseDevice()
                self.cam.MV_CC_DestroyHandle()
            except Exception:
                pass
            self.cam = None
            self.connected = False

    def reconnect(self, cam_index=0):
        self.disconnect()
        if self.sdk_initialized:
            try:
                MvCamera.MV_CC_Finalize()
            except Exception:
                pass
            self.sdk_initialized = False
        self.initialize_sdk()
        self.enum_devices()
        self.connect(cam_index)

    def finalize_sdk(self):
        if self.sdk_initialized:
            try:
                MvCamera.MV_CC_Finalize()
            except Exception:
                pass
            self.sdk_initialized = False


# ============================================================================
#  YOLO + Barkod İşleme
# ============================================================================


def preprocess_crop_for_barcode(crop_bgr):
    """
    Renkli kasa kırpmasını barkod okumaya uygun gri görüntüye çevirir:
    gri tonlama, CLAHE kontrast, hafif keskinleştirme; küçük kırpmaları sınırlı ölçüde büyütür.
    """
    if crop_bgr is None or getattr(crop_bgr, "size", 0) == 0:
        return None
    if len(crop_bgr.shape) == 3 and crop_bgr.shape[2] >= 3:
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    elif len(crop_bgr.shape) == 2:
        gray = np.ascontiguousarray(crop_bgr, dtype=np.uint8)
    else:
        return None

    h, w = gray.shape[:2]
    if h < 2 or w < 2:
        return gray

    min_edge = min(h, w)
    target_min = 360
    scale = 1.0
    if min_edge < target_min:
        scale = float(target_min) / float(min_edge)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    max_side = max(nh, nw, 1)
    if max_side > 2400:
        scale *= 2400.0 / float(max_side)
        nh, nw = int(round(h * scale)), int(round(w * scale))
    if nh != h or nw != w:
        interp = cv2.INTER_CUBIC if nh > h else cv2.INTER_AREA
        gray = cv2.resize(gray, (max(1, nw), max(1, nh)), interpolation=interp)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.2)
    sharpened = cv2.addWeighted(enhanced, 1.45, blur, -0.45, 0.0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


class DetectionProcessor:
    """YOLO kasa tespiti ve barkod okuma işlemcisi."""

    def __init__(self):
        self.model = None
        self.model_names = {}

    def load_model(self, log_fn=None):
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
            if log_fn:
                log_fn(f"Model: {os.path.basename(model_path)}")
            self.model = YOLO(model_path)
            if torch.cuda.is_available():
                self.model.to("cuda")
                if log_fn:
                    log_fn("GPU (CUDA) kullanılıyor.")
            else:
                if log_fn:
                    log_fn("CPU kullanılıyor.")
            self.model_names = getattr(self.model, "names", {})
            if log_fn:
                log_fn("Model yüklendi.")

    def process_image(self, img_bgr, save_dir, log_fn=None):
        self.load_model(log_fn)

        def out(msg):
            if log_fn:
                log_fn(msg)

        os.makedirs(save_dir, exist_ok=True)
        crops_dir = os.path.join(save_dir, "crops")
        os.makedirs(crops_dir, exist_ok=True)

        device = 0 if torch.cuda.is_available() else "cpu"

        out("YOLO tespiti yapılıyor...")
        results = self.model.predict(source=img_bgr, save=False, conf=CONFIDENCE_THRESHOLD, device=device)

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
        barkod_bulunamayan_isimler = []
        barkod_bulunamayan_yollar = []
        box_barkod_durumu = {}
        barkod_icerikler = []
        kasalar = []  # Kasa başına detay: {bbox, barkodlar, barkod_okundu}

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
                        fit = cv2.resize(cv2.cvtColor(crop_color, cv2.COLOR_BGR2GRAY), (pw, ph), interpolation=cv2.INTER_AREA)
                        barcode_vis[y1:y2, x1:x2] = cv2.cvtColor(fit, cv2.COLOR_GRAY2BGR)

                    kasa_no = i + 1
                    crop_name = f"{timestamp_str}_kasa_{kasa_no}{CROP_EXT}"
                    barkod_okundu = True
                    icerikler = []

                    if BARCODE_AVAILABLE:
                        zx_src = proc_hi if proc_hi is not None else cv2.cvtColor(crop_color, cv2.COLOR_BGR2GRAY)
                        barcodes = zx.read_barcodes(zx_src)
                        n_barcode = len(barcodes)
                        if n_barcode == 0:
                            barkod_okundu = False
                            crop_name = f"OKUNAMADI_{timestamp_str}_kasa_{kasa_no}{CROP_EXT}"
                            crop_full_path = os.path.join(crops_dir, crop_name)
                            barkod_bulunamayan_isimler.append(crop_name)
                            barkod_bulunamayan_yollar.append(crop_full_path)
                            out(f"  Barkod bulunamadı: Kasa #{kasa_no}")
                        else:
                            toplam_barkod += n_barcode
                            icerikler = [b.text for b in barcodes]
                            barkod_icerikler.extend(icerikler)
                            out(f"  Kasa #{kasa_no} barkod: {n_barcode} adet → {icerikler}")
                    else:
                        out(f"  Kasa #{kasa_no} (barkod okuma devre dışı)")

                    box_barkod_durumu[i] = barkod_okundu
                    kasalar.append({
                        "kasa_no": kasa_no,
                        "bbox": (x1, y1, x2, y2),
                        "barkodlar": list(icerikler),
                        "barkod_okundu": barkod_okundu,
                    })
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
                cv2.putText(annotated, numara, (tx, ty), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

        annotated_path = os.path.join(save_dir, f"annotated_{timestamp_str}.jpg")
        cv2.imwrite(annotated_path, annotated)

        barcode_prep_path = os.path.join(save_dir, f"barcode_preprocess_{timestamp_str}.jpg")
        cv2.imwrite(barcode_prep_path, barcode_vis)

        out("")
        out("=" * 50)
        out("ÖZET")
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


# ============================================================================
#  Yardımcı GUI Fonksiyonları
# ============================================================================


def _fit_image(pil_img, max_w, max_h):
    w, h = pil_img.size
    ratio = min(max_w / w, max_h / h, 1.0)
    return pil_img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)


def _fit_image_allow_upscale(pil_img, max_w, max_h):
    """Görüntüyü max_w x max_h alanına sığdırır; gerekirse büyütür (zoom için)."""
    w, h = pil_img.size
    if w < 1 or h < 1:
        return pil_img
    ratio = min(max_w / float(w), max_h / float(h))
    return pil_img.resize((max(1, int(round(w * ratio))), max(1, int(round(h * ratio)))), Image.LANCZOS)


def numpy_to_photoimage(np_img, max_w, max_h):
    """BGR (veya tek kanal gri) numpy görüntüyü Tkinter PhotoImage'e çevirir."""
    if np_img is None:
        return None
    if len(np_img.shape) == 2:
        rgb = cv2.cvtColor(np_img, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    pil_img = _fit_image(pil_img, max_w, max_h)
    return ImageTk.PhotoImage(pil_img)


def numpy_to_photoimage_viewport(np_img, max_w, max_h):
    """Kırpılmış görüntüyü alana sığdırır; küçük kırpma büyütülebilir."""
    if np_img is None:
        return None
    if len(np_img.shape) == 2:
        rgb = cv2.cvtColor(np_img, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    pil_img = _fit_image_allow_upscale(pil_img, max_w, max_h)
    return ImageTk.PhotoImage(pil_img)


# ============================================================================
#  MODERN STAT CARD WİDGET
# ============================================================================


class StatCard(ctk.CTkFrame):
    """İstatistik kartı — başlık + büyük sayı gösterir."""

    def __init__(self, master, title, icon="", value="—", value_color=Colors.TEXT_PRIMARY, **kwargs):
        super().__init__(
            master,
            fg_color=Colors.BG_CARD,
            corner_radius=12,
            border_width=1,
            border_color=Colors.BORDER,
            **kwargs,
        )

        inner = ctk.CTkFrame(self, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=16, pady=12)

        header = ctk.CTkFrame(inner, fg_color="transparent")
        header.pack(fill="x")
        ctk.CTkLabel(
            header,
            text=f"{icon}  {title}",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color=Colors.TEXT_SECONDARY,
            anchor="w",
        ).pack(side="left")

        self.value_label = ctk.CTkLabel(
            inner,
            text=str(value),
            font=ctk.CTkFont(family="Segoe UI", size=32, weight="bold"),
            text_color=value_color,
            anchor="w",
        )
        self.value_label.pack(fill="x", pady=(6, 0))

    def set_value(self, val, color=None):
        self.value_label.configure(text=str(val))
        if color:
            self.value_label.configure(text_color=color)


# ============================================================================
#  STATUS PILL WİDGET
# ============================================================================


class StatusPill(ctk.CTkFrame):
    """Durum göstergesi — renkli nokta + metin."""

    def __init__(self, master, text="Bağlı değil", color=Colors.TEXT_MUTED, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)

        self._dot = ctk.CTkLabel(
            self,
            text="●",
            font=ctk.CTkFont(size=14),
            text_color=color,
            width=18,
        )
        self._dot.pack(side="left", padx=(0, 6))

        self._label = ctk.CTkLabel(
            self,
            text=text,
            font=ctk.CTkFont(family="Segoe UI", size=13),
            text_color=Colors.TEXT_SECONDARY,
        )
        self._label.pack(side="left")

    def set_status(self, text, color):
        self._dot.configure(text_color=color)
        self._label.configure(text=text)


# ============================================================================
#  ANA GUI UYGULAMASI
# ============================================================================


def run_gui():
    camera_mgr = CameraManager()
    detector = DetectionProcessor()

    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("green")

    app = ctk.CTk()
    app.title("Kamera + Kasa Tespit + Barkod Okuma")
    app.minsize(1200, 800)
    app.geometry("1400x900")
    app.configure(fg_color=Colors.BG_DARK)

    captured_image = [None]
    captured_path = [None]
    last_result = [None]
    img_tk_ref = [None]
    current_display_np = [None]
    img_zoom = {"z": 1.0, "cx": 0.5, "cy": 0.5}
    pan_drag = {"active": False, "lx": 0, "ly": 0}
    img_resize_after = [None]

    toolbar = ctk.CTkFrame(app, fg_color=Colors.BG_CARD, corner_radius=0, height=60)
    toolbar.pack(fill="x", padx=0, pady=0)
    toolbar.pack_propagate(False)

    toolbar_inner = ctk.CTkFrame(toolbar, fg_color="transparent")
    toolbar_inner.pack(fill="both", expand=True, padx=16, pady=8)

    # Sol üst: Trento amblemi (PNG şeffaf arka plan → araç çubuğu rengine birleşir)
    logo_wrap = ctk.CTkFrame(toolbar_inner, fg_color="transparent")
    logo_wrap.pack(side="left", padx=(0, 16), pady=2)
    app._trento_logo_ctk = None
    logo_path = resolve_trento_logo_path()
    if logo_path:
        try:
            pil_logo = pil_trento_logo_to_rgb(Image.open(logo_path), Colors.BG_CARD)
            max_h = 38
            lw, lh = pil_logo.size
            new_w = max(1, int(lw * max_h / lh))
            pil_logo = pil_logo.resize((new_w, max_h), Image.Resampling.LANCZOS)
            app._trento_logo_ctk = ctk.CTkImage(
                light_image=pil_logo, dark_image=pil_logo, size=(new_w, max_h)
            )
            ctk.CTkLabel(
                logo_wrap,
                image=app._trento_logo_ctk,
                text="",
                fg_color="transparent",
            ).pack()
        except OSError:
            app._trento_logo_ctk = None

    ctk.CTkLabel(
        toolbar_inner,
        text="📸  KAMERA SİSTEMİ",
        font=ctk.CTkFont(family="Segoe UI", size=16, weight="bold"),
        text_color=Colors.BRAND_GREEN_SOFT,
    ).pack(side="left", padx=(0, 24))

    status_pill = StatusPill(toolbar_inner, text="Kamera: Bağlı değil", color=Colors.TEXT_MUTED)
    status_pill.pack(side="left", padx=(0, 20))

    btn_frame = ctk.CTkFrame(toolbar_inner, fg_color="transparent")
    btn_frame.pack(side="left", padx=(8, 0))

    btn_connect = ctk.CTkButton(
        btn_frame,
        text="🔌  Bağlan",
        width=120,
        height=36,
        font=ctk.CTkFont(size=13, weight="bold"),
        fg_color=Colors.BTN_PRIMARY,
        hover_color=Colors.BTN_HOVER,
        corner_radius=8,
    )
    btn_connect.pack(side="left", padx=4)

    btn_capture = ctk.CTkButton(
        btn_frame,
        text="📷  Fotoğraf Çek",
        width=140,
        height=36,
        font=ctk.CTkFont(size=13, weight="bold"),
        fg_color=Colors.BTN_BLUE,
        hover_color=Colors.BTN_BLUE_HOVER,
        corner_radius=8,
        state="disabled",
    )
    btn_capture.pack(side="left", padx=4)

    btn_process = ctk.CTkButton(
        btn_frame,
        text="🔍  YOLO + Barkod",
        width=160,
        height=36,
        font=ctk.CTkFont(size=13, weight="bold"),
        fg_color=Colors.BTN_BLUE,
        hover_color=Colors.BTN_BLUE_HOVER,
        corner_radius=8,
        state="disabled",
    )
    btn_process.pack(side="left", padx=4)

    btn_capture_process = ctk.CTkButton(
        btn_frame,
        text="⚡  Çek ve İşle",
        width=140,
        height=36,
        font=ctk.CTkFont(size=13, weight="bold"),
        fg_color=Colors.BRAND_GREEN_DIM,
        hover_color=Colors.BRAND_GREEN,
        corner_radius=8,
        state="disabled",
    )
    btn_capture_process.pack(side="left", padx=4)

    ctk.CTkFrame(btn_frame, fg_color=Colors.BORDER, width=2, height=28).pack(side="left", padx=12)

    btn_restart = ctk.CTkButton(
        btn_frame,
        text="🔄  Yeniden Başlat",
        width=150,
        height=36,
        font=ctk.CTkFont(size=13, weight="bold"),
        fg_color=Colors.BTN_DANGER,
        hover_color="#f85149",
        corner_radius=8,
        state="disabled",
    )
    btn_restart.pack(side="left", padx=4)

    ctk.CTkFrame(app, fg_color=Colors.BRAND_GREEN_DIM, height=2, corner_radius=0).pack(fill="x")

    main_container = ctk.CTkFrame(app, fg_color="transparent")
    main_container.pack(fill="both", expand=True, padx=16, pady=12)

    sidebar = ctk.CTkFrame(main_container, fg_color="transparent", width=260)
    sidebar.pack(side="left", fill="y", padx=(0, 12))
    sidebar.pack_propagate(False)

    ctk.CTkLabel(
        sidebar,
        text="SONUÇLAR",
        font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
        text_color=Colors.BRAND_GREEN_SOFT,
        anchor="w",
    ).pack(fill="x", pady=(0, 8))

    card_kasa = StatCard(sidebar, title="Toplam Kasa", icon="📦", value_color=Colors.BRAND_GREEN_SOFT)
    card_kasa.pack(fill="x", pady=(0, 8))

    card_okunan = StatCard(sidebar, title="Okunan Barkod", icon="✅", value_color=Colors.ACCENT_GREEN)
    card_okunan.pack(fill="x", pady=(0, 8))

    card_okunamayan = StatCard(sidebar, title="Okunamayan Barkod", icon="❌", value_color=Colors.ACCENT_RED)
    card_okunamayan.pack(fill="x", pady=(0, 12))

    info_card = ctk.CTkFrame(
        sidebar,
        fg_color=Colors.BG_CARD,
        corner_radius=12,
        border_width=1,
        border_color=Colors.BORDER,
    )
    info_card.pack(fill="x", pady=(0, 12))

    info_inner = ctk.CTkFrame(info_card, fg_color="transparent")
    info_inner.pack(fill="x", padx=16, pady=12)

    ctk.CTkLabel(
        info_inner,
        text="📐  GÖRÜNTÜ BİLGİSİ",
        font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
        text_color=Colors.BRAND_GREEN_SOFT,
        anchor="w",
    ).pack(fill="x", pady=(0, 8))

    lbl_img_size = ctk.CTkLabel(
        info_inner,
        text="Boyut:  —",
        font=ctk.CTkFont(family="Segoe UI", size=12),
        text_color=Colors.TEXT_MUTED,
        anchor="w",
    )
    lbl_img_size.pack(fill="x", pady=1)

    lbl_img_time = ctk.CTkLabel(
        info_inner,
        text="Zaman:  —",
        font=ctk.CTkFont(family="Segoe UI", size=12),
        text_color=Colors.TEXT_MUTED,
        anchor="w",
    )
    lbl_img_time.pack(fill="x", pady=1)

    lbl_img_file = ctk.CTkLabel(
        info_inner,
        text="Dosya:  —",
        font=ctk.CTkFont(family="Segoe UI", size=12),
        text_color=Colors.TEXT_MUTED,
        anchor="w",
        wraplength=210,
    )
    lbl_img_file.pack(fill="x", pady=1)

    btn_load_image = ctk.CTkButton(
        info_inner,
        text="📁  Görsel yükle",
        height=26,
        font=ctk.CTkFont(size=11, weight="bold"),
        fg_color=Colors.BG_SURFACE,
        hover_color=Colors.BORDER_LIGHT,
        border_width=1,
        border_color=Colors.BORDER,
        corner_radius=6,
    )
    btn_load_image.pack(fill="x", pady=(8, 0))

    exposure_card = ctk.CTkFrame(
        sidebar,
        fg_color=Colors.BG_CARD,
        corner_radius=12,
        border_width=1,
        border_color=Colors.BORDER,
    )
    exposure_card.pack(fill="x", pady=(0, 12))

    exp_inner = ctk.CTkFrame(exposure_card, fg_color="transparent")
    exp_inner.pack(fill="x", padx=14, pady=12)

    ctk.CTkLabel(
        exp_inner,
        text="☀️  POZLAMA (Exposure)",
        font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
        text_color=Colors.BRAND_GREEN_SOFT,
        anchor="w",
    ).pack(fill="x", pady=(0, 8))

    sw_exp_auto = ctk.CTkSwitch(
        exp_inner,
        text="Otomatik pozlama",
        font=ctk.CTkFont(size=12),
        text_color=Colors.TEXT_PRIMARY,
        fg_color=Colors.BORDER,
        progress_color=Colors.BTN_BLUE,
        button_color=Colors.TEXT_MUTED,
        button_hover_color=Colors.TEXT_SECONDARY,
    )
    sw_exp_auto.pack(anchor="w", pady=(0, 6))

    lbl_exp_val = ctk.CTkLabel(
        exp_inner,
        text="—  µs",
        font=ctk.CTkFont(family="Segoe UI", size=12),
        text_color=Colors.TEXT_MUTED,
        anchor="w",
    )
    lbl_exp_val.pack(fill="x", pady=(0, 4))

    sld_exp = ctk.CTkSlider(
        exp_inner,
        from_=100.0,
        to=80000.0,
        number_of_steps=799,
        height=16,
        button_color=Colors.ACCENT_CYAN,
        button_hover_color=Colors.ACCENT_CYAN,
        progress_color=Colors.BTN_BLUE,
        fg_color=Colors.BG_SURFACE,
        state="disabled",
    )
    sld_exp.pack(fill="x", pady=(0, 0))

    exp_bounds = [50.0, 200000.0]
    exp_apply_after = [None]

    def _exp_is_auto():
        v = sw_exp_auto.get()
        return v in (1, True, "on", "On")

    def set_exposure_widgets_state(enabled):
        st = "normal" if enabled and MVS_AVAILABLE else "disabled"
        sw_exp_auto.configure(state=st)
        if enabled and MVS_AVAILABLE and not _exp_is_auto():
            sld_exp.configure(state="normal")
        else:
            sld_exp.configure(state="disabled")

    def refresh_exp_label_from_slider():
        try:
            v = int(float(sld_exp.get()))
            lbl_exp_val.configure(text=f"{v}  µs", text_color=Colors.TEXT_PRIMARY)
        except (ValueError, tk.TclError):
            pass

    def on_exp_slider(_val):
        refresh_exp_label_from_slider()
        if not camera_mgr.connected or _exp_is_auto():
            return
        if exp_apply_after[0] is not None:
            try:
                app.after_cancel(exp_apply_after[0])
            except Exception:
                pass

        def apply():
            try:
                v = float(sld_exp.get())
                lo, hi = exp_bounds[0], exp_bounds[1]
                v = max(lo, min(hi, v))
                camera_mgr.set_exposure_time_us(v)
            except Exception:
                pass
            exp_apply_after[0] = None

        exp_apply_after[0] = app.after(120, apply)

    def on_exp_auto_toggled():
        if not camera_mgr.connected:
            return
        if _exp_is_auto():
            camera_mgr.set_exposure_auto(True)
            sld_exp.configure(state="disabled")
        else:
            camera_mgr.set_exposure_auto(False)
            try:
                v = float(sld_exp.get())
                lo, hi = exp_bounds[0], exp_bounds[1]
                v = max(lo, min(hi, v))
                camera_mgr.set_exposure_time_us(v)
            except Exception:
                pass
            sld_exp.configure(state="normal")
        refresh_exp_label_from_slider()

    sw_exp_auto.configure(command=on_exp_auto_toggled)
    sld_exp.configure(command=on_exp_slider)
    sw_exp_auto.select()
    set_exposure_widgets_state(False)

    def refresh_exposure_from_camera():
        if not camera_mgr.connected or not MVS_AVAILABLE:
            set_exposure_widgets_state(False)
            return
        sw_exp_auto.select()
        set_exposure_widgets_state(True)
        data = camera_mgr.read_exposure_time_us()
        if data:
            lo, hi, cur = data["min"], data["max"], data["cur"]
            exp_bounds[0], exp_bounds[1] = lo, hi
            span = max(hi - lo, 1.0)
            steps = min(600, max(40, int(span / max(span / 500.0, 1.0))))
            try:
                sld_exp.configure(from_=lo, to=hi, number_of_steps=steps)
            except Exception:
                try:
                    sld_exp.configure(from_=lo, to=hi)
                except Exception:
                    pass
            cur = max(lo, min(hi, cur))
            sld_exp.set(cur)
            lbl_exp_val.configure(text=f"{int(cur)}  µs", text_color=Colors.TEXT_PRIMARY)
        else:
            lbl_exp_val.configure(
                text="Süre okunamadı (kayıdırıcı yine de denenebilir)",
                text_color=Colors.ACCENT_YELLOW,
            )

    unread_header_sb = ctk.CTkFrame(sidebar, fg_color="transparent")
    unread_header_sb.pack(fill="x", pady=(8, 4))
    ctk.CTkLabel(
        unread_header_sb,
        text="⚠️  OKUNAMAYAN",
        font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
        text_color=Colors.ACCENT_YELLOW,
        anchor="w",
    ).pack(side="left")
    ctk.CTkLabel(
        unread_header_sb,
        text="(çift tık → crop)",
        font=ctk.CTkFont(family="Segoe UI", size=10),
        text_color=Colors.TEXT_MUTED,
    ).pack(side="left", padx=(6, 0))

    list_frame_sb = ctk.CTkFrame(
        sidebar,
        fg_color=Colors.BG_CARD,
        corner_radius=10,
        border_width=1,
        border_color=Colors.BORDER,
    )
    list_frame_sb.pack(fill="both", expand=True, pady=(0, 0))

    right_panel = ctk.CTkFrame(main_container, fg_color="transparent")
    right_panel.pack(side="left", fill="both", expand=True)

    img_section = ctk.CTkFrame(
        right_panel,
        fg_color=Colors.BG_CARD,
        corner_radius=12,
        border_width=1,
        border_color=Colors.BORDER,
    )
    img_section.pack(fill="both", expand=True, pady=(0, 10))

    img_toggle_bar = ctk.CTkFrame(img_section, fg_color="transparent")
    img_toggle_bar.pack(fill="x", padx=12, pady=(10, 4))

    btn_show_raw = ctk.CTkButton(
        img_toggle_bar,
        text="🖼  Ham Görüntü",
        width=130,
        height=30,
        font=ctk.CTkFont(size=12),
        fg_color=Colors.BG_SURFACE,
        hover_color=Colors.BORDER_LIGHT,
        corner_radius=6,
    )
    btn_show_raw.pack(side="left", padx=(0, 6))

    btn_show_annotated = ctk.CTkButton(
        img_toggle_bar,
        text="🎯  İşlenmiş Görüntü",
        width=150,
        height=30,
        font=ctk.CTkFont(size=12),
        fg_color=Colors.BG_SURFACE,
        hover_color=Colors.BORDER_LIGHT,
        corner_radius=6,
        state="disabled",
    )
    btn_show_annotated.pack(side="left", padx=(0, 6))

    btn_show_barcode_prep = ctk.CTkButton(
        img_toggle_bar,
        text="📷  Barkod (ön işlem)",
        width=168,
        height=30,
        font=ctk.CTkFont(size=12),
        fg_color=Colors.BG_SURFACE,
        hover_color=Colors.BORDER_LIGHT,
        corner_radius=6,
        state="disabled",
    )
    btn_show_barcode_prep.pack(side="left", padx=(0, 6))

    btn_fullscreen = ctk.CTkButton(
        img_toggle_bar,
        text="⛶  Tam Ekran",
        width=120,
        height=30,
        font=ctk.CTkFont(size=12),
        fg_color=Colors.BG_SURFACE,
        hover_color=Colors.BORDER_LIGHT,
        corner_radius=6,
        state="disabled",
    )
    btn_fullscreen.pack(side="left", padx=(0, 12))

    lbl_img_info = ctk.CTkLabel(
        img_toggle_bar,
        text="Fotoğraf çekilmedi",
        font=ctk.CTkFont(family="Segoe UI", size=12),
        text_color=Colors.TEXT_MUTED,
    )
    lbl_img_info.pack(side="left")

    ctk.CTkLabel(
        img_toggle_bar,
        text="Tekerlek: yakın/uzak · Sürükle: kaydır",
        font=ctk.CTkFont(size=10),
        text_color=Colors.TEXT_MUTED,
    ).pack(side="left", padx=(8, 0))

    lbl_image = tk.Label(
        img_section,
        bg=Colors.IMAGE_CANVAS,
        relief="flat",
        bd=0,
    )
    lbl_image.pack(fill="both", expand=True, padx=12, pady=(4, 12))

    read_header = ctk.CTkFrame(right_panel, fg_color="transparent")
    read_header.pack(fill="x", pady=(0, 4))
    ctk.CTkLabel(
        read_header,
        text="📋  OKUNAN BARKODLAR (içerik)",
        font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
        text_color=Colors.BRAND_GREEN_SOFT,
        anchor="w",
    ).pack(side="left")

    txt_barkodlar = ctk.CTkTextbox(
        right_panel,
        font=ctk.CTkFont(family="Consolas", size=11),
        fg_color=Colors.BG_CARD,
        text_color=Colors.ACCENT_GREEN,
        border_width=1,
        border_color=Colors.BORDER,
        corner_radius=10,
        height=160,
        wrap="word",
    )
    txt_barkodlar.pack(fill="x", pady=(0, 0))

    listbox = tk.Listbox(
        list_frame_sb,
        height=6,
        font=("Consolas", 10),
        bg=Colors.BG_CARD,
        fg=Colors.ACCENT_RED,
        selectbackground=Colors.BTN_DANGER,
        selectforeground="#ffffff",
        activestyle="none",
        highlightthickness=0,
        bd=0,
        relief="flat",
    )
    listbox_scroll = ctk.CTkScrollbar(list_frame_sb, command=listbox.yview)
    listbox.config(yscrollcommand=listbox_scroll.set)
    listbox.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=8)
    listbox_scroll.pack(side="right", fill="y", padx=(0, 4), pady=8)

    def refresh_image_view():
        np_img = current_display_np[0]
        if np_img is None:
            return
        app.update_idletasks()
        max_w = max(lbl_image.winfo_width() - 20, 240)
        max_h = max(lbl_image.winfo_height() - 20, 180)
        sh, sw = np_img.shape[:2]
        if sh < 1 or sw < 1:
            return
        z = max(1.0, min(float(img_zoom["z"]), 8.0))
        img_zoom["z"] = z
        vw = max(1, int(round(sw / z)))
        vh = max(1, int(round(sh / z)))
        cx = max(vw / (2.0 * sw), min(1.0 - vw / (2.0 * sw), float(img_zoom["cx"])))
        cy = max(vh / (2.0 * sh), min(1.0 - vh / (2.0 * sh), float(img_zoom["cy"])))
        img_zoom["cx"], img_zoom["cy"] = cx, cy
        x0 = int(round(cx * sw - vw / 2.0))
        y0 = int(round(cy * sh - vh / 2.0))
        x0 = max(0, min(sw - vw, x0))
        y0 = max(0, min(sh - vh, y0))
        crop = np_img[y0 : y0 + vh, x0 : x0 + vw]
        tk_img = numpy_to_photoimage_viewport(crop, max_w, max_h)
        if tk_img:
            img_tk_ref[0] = tk_img
            lbl_image.config(image=tk_img)
        lbl_image.configure(cursor="fleur" if z > 1.02 else "")

    def zoom_at_mouse(delta, mx, my):
        np_img = current_display_np[0]
        if np_img is None or delta == 0:
            return
        sh, sw = np_img.shape[:2]
        z = max(1.0, min(float(img_zoom["z"]), 8.0))
        vw = max(1, int(round(sw / z)))
        vh = max(1, int(round(sh / z)))
        x0 = int(round(img_zoom["cx"] * sw - vw / 2.0))
        y0 = int(round(img_zoom["cy"] * sh - vh / 2.0))
        x0 = max(0, min(sw - vw, x0))
        y0 = max(0, min(sh - vh, y0))
        W = max(lbl_image.winfo_width(), 1)
        H = max(lbl_image.winfo_height(), 1)
        nx = max(0.0, min(1.0, mx / float(W)))
        ny = max(0.0, min(1.0, my / float(H)))
        src_x = x0 + nx * vw
        src_y = y0 + ny * vh
        factor = 1.12 if delta > 0 else 1.0 / 1.12
        z_new = max(1.0, min(8.0, z * factor))
        if abs(z_new - z) < 1e-6:
            return
        vw2 = max(1, int(round(sw / z_new)))
        vh2 = max(1, int(round(sh / z_new)))
        img_zoom["z"] = z_new
        img_zoom["cx"] = src_x / float(sw)
        img_zoom["cy"] = src_y / float(sh)
        img_zoom["cx"] = max(vw2 / (2.0 * sw), min(1.0 - vw2 / (2.0 * sw), img_zoom["cx"]))
        img_zoom["cy"] = max(vh2 / (2.0 * sh), min(1.0 - vh2 / (2.0 * sh), img_zoom["cy"]))
        refresh_image_view()

    def on_image_wheel(event):
        zoom_at_mouse(getattr(event, "delta", 0), event.x, event.y)

    def on_linux_wheel_in(event):
        zoom_at_mouse(120, event.x, event.y)

    def on_linux_wheel_out(event):
        zoom_at_mouse(-120, event.x, event.y)

    def on_img_enter(_event):
        try:
            lbl_image.focus_set()
        except Exception:
            pass

    def on_img_press(event):
        if float(img_zoom["z"]) <= 1.02:
            return
        pan_drag["active"] = True
        pan_drag["lx"], pan_drag["ly"] = event.x, event.y

    def on_img_release(_event):
        pan_drag["active"] = False

    def on_img_motion(event):
        if not pan_drag["active"] or current_display_np[0] is None:
            return
        np_img = current_display_np[0]
        sh, sw = np_img.shape[:2]
        W = max(lbl_image.winfo_width(), 1)
        H = max(lbl_image.winfo_height(), 1)
        z = max(1.0, min(float(img_zoom["z"]), 8.0))
        vw = sw / z
        vh = sh / z
        dx = event.x - pan_drag["lx"]
        dy = event.y - pan_drag["ly"]
        pan_drag["lx"], pan_drag["ly"] = event.x, event.y
        dcx = -(dx / float(W)) * (vw / sw)
        dcy = -(dy / float(H)) * (vh / sh)
        img_zoom["cx"] += dcx
        img_zoom["cy"] += dcy
        img_zoom["cx"] = max(vw / (2.0 * sw), min(1.0 - vw / (2.0 * sw), img_zoom["cx"]))
        img_zoom["cy"] = max(vh / (2.0 * sh), min(1.0 - vh / (2.0 * sh), img_zoom["cy"]))
        refresh_image_view()

    def on_img_section_configure(_event):
        if current_display_np[0] is None:
            return
        if img_resize_after[0] is not None:
            try:
                app.after_cancel(img_resize_after[0])
            except Exception:
                pass
        img_resize_after[0] = app.after(100, refresh_image_view)

    lbl_image.bind("<MouseWheel>", on_image_wheel)
    lbl_image.bind("<Button-4>", on_linux_wheel_in)
    lbl_image.bind("<Button-5>", on_linux_wheel_out)
    lbl_image.bind("<Enter>", on_img_enter)
    lbl_image.bind("<ButtonPress-1>", on_img_press)
    lbl_image.bind("<ButtonRelease-1>", on_img_release)
    lbl_image.bind("<B1-Motion>", on_img_motion)
    img_section.bind("<Configure>", on_img_section_configure)

    def show_image_on_label(np_img, info_text=""):
        img_zoom["z"] = 1.0
        img_zoom["cx"] = 0.5
        img_zoom["cy"] = 0.5
        current_display_np[0] = np_img
        if np_img is None:
            return
        refresh_image_view()
        btn_fullscreen.configure(state="normal")
        if info_text:
            lbl_img_info.configure(text=info_text, text_color=Colors.TEXT_PRIMARY)

    def on_show_raw():
        if captured_image[0] is not None:
            show_image_on_label(captured_image[0], "🖼  Ham Görüntü")

    def on_show_annotated():
        if last_result[0] and last_result[0].get("annotated_image") is not None:
            show_image_on_label(last_result[0]["annotated_image"], "🎯  İşlenmiş Görüntü (YOLO + Barkod)")

    def on_show_barcode_prep():
        if last_result[0] and last_result[0].get("barcode_preprocess_image") is not None:
            show_image_on_label(
                last_result[0]["barcode_preprocess_image"],
                "📷  Barkod okuma — ön işlenmiş (gri/kontrast)",
            )

    btn_show_raw.configure(command=on_show_raw)
    btn_show_annotated.configure(command=on_show_annotated)
    btn_show_barcode_prep.configure(command=on_show_barcode_prep)

    def open_image_fullscreen():
        np_img = current_display_np[0]
        if np_img is None:
            return

        fs = tk.Toplevel(app)
        fs.title("Görüntü — Tam ekran")
        fs.configure(bg=Colors.BG_DARK)
        fs.attributes("-fullscreen", True)
        fs.focus_force()

        def close_fs(event=None):
            fs.destroy()

        fs.bind("<Escape>", close_fs)
        fs.protocol("WM_DELETE_WINDOW", close_fs)

        top_bar = tk.Frame(fs, bg=Colors.BG_CARD, height=44)
        top_bar.pack(fill="x")
        top_bar.pack_propagate(False)
        tk.Button(
            top_bar,
            text="✕  Kapat  (Esc)",
            command=close_fs,
            bg=Colors.BTN_DANGER,
            fg="#ffffff",
            activebackground="#f85149",
            activeforeground="#ffffff",
            font=("Segoe UI", 11, "bold"),
            relief="flat",
            padx=14,
            pady=6,
            cursor="hand2",
        ).pack(side="right", padx=10, pady=6)

        body = tk.Frame(fs, bg=Colors.BG_DARK)
        body.pack(fill="both", expand=True)

        sw = app.winfo_screenwidth()
        sh = app.winfo_screenheight()
        pad_x, pad_y = 32, 72
        tk_img = numpy_to_photoimage(np_img, max(sw - pad_x, 400), max(sh - pad_y, 300))
        if not tk_img:
            fs.destroy()
            return
        lbl_fs = tk.Label(body, image=tk_img, bg=Colors.IMAGE_CANVAS)
        lbl_fs.image = tk_img
        lbl_fs.pack(expand=True, fill="both", padx=12, pady=12)

    btn_fullscreen.configure(command=open_image_fullscreen)

    def on_listbox_dblclick(event):
        sel = listbox.curselection()
        if not sel or last_result[0] is None:
            return
        idx = sel[0]
        yollar = last_result[0].get("barkod_bulunamayan_yollar", [])
        if idx < len(yollar):
            path = yollar[idx]
            if os.path.isfile(path):
                win = ctk.CTkToplevel(app)
                win.title(os.path.basename(path))
                win.geometry("800x600")
                win.configure(fg_color=Colors.BG_CARD)
                pil_img = _fit_image(Image.open(path), 780, 580)
                tk_img = ImageTk.PhotoImage(pil_img)
                lbl = tk.Label(win, image=tk_img, bg=Colors.IMAGE_CANVAS)
                lbl.image = tk_img
                lbl.pack(fill="both", expand=True)

    listbox.bind("<Double-Button-1>", on_listbox_dblclick)

    def do_connect():
        btn_connect.configure(state="disabled")
        status_pill.set_status("Bağlanıyor...", Colors.ACCENT_YELLOW)

        def work():
            try:
                camera_mgr.initialize_sdk()
                camera_mgr.enum_devices()
                camera_mgr.connect(0)
                app.after(0, lambda: on_connect_done(None))
            except Exception as e:
                app.after(0, lambda err=e: on_connect_done(err))

        def on_connect_done(err):
            if err:
                status_pill.set_status("Hata!", Colors.ACCENT_RED)
                btn_connect.configure(state="normal")
                set_exposure_widgets_state(False)
            else:
                status_pill.set_status("Bağlı", Colors.ACCENT_GREEN)
                btn_connect.configure(state="disabled")
                btn_capture.configure(state="normal")
                btn_capture_process.configure(state="normal")
                btn_restart.configure(state="normal")
                app.after(220, refresh_exposure_from_camera)

        threading.Thread(target=work, daemon=True).start()

    btn_connect.configure(command=do_connect)

    def do_load_image_from_disk():
        if not HAS_OPENCV:
            lbl_img_info.configure(
                text="❌  OpenCV yok; görsel yüklenemiyor.",
                text_color=Colors.ACCENT_RED,
            )
            return
        path = tk.filedialog.askopenfilename(
            parent=app,
            title="İşlenecek görseli seçin",
            filetypes=[
                ("Görüntü dosyaları", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff"),
                ("JPEG", "*.jpg *.jpeg"),
                ("PNG", "*.png"),
                ("Tüm dosyalar", "*.*"),
            ],
        )
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            lbl_img_info.configure(
                text="❌  Görüntü okunamadı (format veya dosya hatası).",
                text_color=Colors.ACCENT_RED,
            )
            return
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif len(img.shape) == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        captured_image[0] = img
        captured_path[0] = path
        h, w = img.shape[:2]
        lbl_img_size.configure(text=f"Boyut:  {w} x {h}")
        lbl_img_time.configure(text=f"Zaman:  {datetime.now().strftime('%H:%M:%S')}")
        lbl_img_file.configure(text=f"Dosya:  {os.path.basename(path)}")

        show_image_on_label(img, f"🖼  Dosyadan — {w}x{h}")
        last_result[0] = None
        btn_show_annotated.configure(state="disabled")
        btn_show_barcode_prep.configure(state="disabled")
        btn_process.configure(state="normal")
        lbl_img_info.configure(
            text="📁  Görüntü yüklendi, YOLO başlatılıyor…",
            text_color=Colors.ACCENT_YELLOW,
        )
        app.after(80, do_process_image)

    btn_load_image.configure(command=do_load_image_from_disk)

    def do_capture(then_process=False):
        btn_capture.configure(state="disabled")
        btn_process.configure(state="disabled")
        btn_capture_process.configure(state="disabled")
        btn_load_image.configure(state="disabled")
        lbl_img_info.configure(text="📷  Fotoğraf çekiliyor...", text_color=Colors.ACCENT_YELLOW)

        def work():
            try:
                img = camera_mgr.capture_single_frame()
                os.makedirs(SAVE_DIR, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(SAVE_DIR, f"capture_{timestamp}.jpg")
                cv2.imwrite(save_path, img)
                app.after(
                    0,
                    lambda i=img, p=save_path, tp=then_process: on_capture_done(i, p, None, tp),
                )
            except Exception as e:
                app.after(0, lambda err=e: on_capture_done(None, None, err, False))

        def on_capture_done(img, path, err, do_process):
            if err:
                lbl_img_info.configure(text="❌  Çekim hatası!", text_color=Colors.ACCENT_RED)
                btn_capture.configure(state="normal")
                btn_capture_process.configure(state="normal")
                btn_load_image.configure(state="normal")
                return

            captured_image[0] = img
            captured_path[0] = path
            h, w = img.shape[:2]
            lbl_img_size.configure(text=f"Boyut:  {w} x {h}")
            lbl_img_time.configure(text=f"Zaman:  {datetime.now().strftime('%H:%M:%S')}")
            lbl_img_file.configure(text=f"Dosya:  {os.path.basename(path)}")

            show_image_on_label(img, f"🖼  Ham Görüntü — {w}x{h}")
            last_result[0] = None
            btn_show_annotated.configure(state="disabled")
            btn_show_barcode_prep.configure(state="disabled")
            btn_capture.configure(state="normal")
            btn_process.configure(state="normal")
            btn_capture_process.configure(state="normal")
            btn_load_image.configure(state="normal")

            if do_process:
                app.after(100, do_process_image)

        threading.Thread(target=work, daemon=True).start()

    btn_capture.configure(command=lambda: do_capture(False))
    btn_capture_process.configure(command=lambda: do_capture(True))

    def do_process_image():
        if captured_image[0] is None:
            return

        btn_capture.configure(state="disabled")
        btn_process.configure(state="disabled")
        btn_capture_process.configure(state="disabled")
        btn_load_image.configure(state="disabled")
        lbl_img_info.configure(text="🔍  YOLO + Barkod işleniyor...", text_color=Colors.ACCENT_YELLOW)
        card_kasa.set_value("…")
        card_okunan.set_value("…")
        card_okunamayan.set_value("…")
        listbox.delete(0, tk.END)
        txt_barkodlar.delete("1.0", "end")

        def work():
            try:
                result = detector.process_image(captured_image[0], SAVE_DIR, log_fn=lambda _msg: None)
                app.after(0, lambda r=result: on_process_done(r, None))
            except Exception as e:
                app.after(0, lambda err=e: on_process_done(None, err))

        def on_process_done(result, err):
            btn_capture.configure(state="normal")
            btn_process.configure(state="normal")
            btn_capture_process.configure(state="normal")
            btn_load_image.configure(state="normal")

            if err:
                card_kasa.set_value("—")
                card_okunan.set_value("—")
                card_okunamayan.set_value("—")
                btn_show_annotated.configure(state="disabled")
                btn_show_barcode_prep.configure(state="disabled")
                traceback.print_exception(type(err), err, err.__traceback__, file=sys.stderr)
                err_line = str(err).strip().replace("\n", " ")
                if len(err_line) > 100:
                    err_line = err_line[:97] + "…"
                lbl_img_info.configure(
                    text=f"❌  {err_line}",
                    text_color=Colors.ACCENT_RED,
                    wraplength=720,
                )
                return

            last_result[0] = result
            card_kasa.set_value(result.get("toplam_kasa", 0))
            card_okunan.set_value(result.get("okunan_barkod", 0))
            card_okunamayan.set_value(result.get("okunamayan_barkod", 0))

            if result.get("annotated_image") is not None:
                show_image_on_label(result["annotated_image"], "🎯  İşlenmiş Görüntü — YOLO + Barkod")
                btn_show_annotated.configure(state="normal")
            if result.get("barcode_preprocess_image") is not None:
                btn_show_barcode_prep.configure(state="normal")

            icerikler = result.get("barkod_icerikler", [])
            if icerikler:
                for bc in icerikler:
                    txt_barkodlar.insert("end", bc + "\n")
            else:
                txt_barkodlar.insert("end", "(Okunan barkod yok)")

            listbox.delete(0, tk.END)
            for name in result.get("barkod_bulunamayan_isimler", []):
                listbox.insert(tk.END, name)

        threading.Thread(target=work, daemon=True).start()

    btn_process.configure(command=do_process_image)

    def do_restart():
        btn_restart.configure(state="disabled")
        btn_capture.configure(state="disabled")
        btn_process.configure(state="disabled")
        btn_capture_process.configure(state="disabled")
        btn_load_image.configure(state="disabled")
        btn_connect.configure(state="disabled")
        set_exposure_widgets_state(False)
        status_pill.set_status("Yeniden başlatılıyor...", Colors.ACCENT_YELLOW)

        def work():
            try:
                camera_mgr.reconnect(0)
                app.after(0, lambda: on_restart_done(None))
            except Exception as e:
                app.after(0, lambda err=e: on_restart_done(err))

        def on_restart_done(err):
            if err:
                status_pill.set_status("Yeniden bağlanma hatası!", Colors.ACCENT_RED)
                btn_connect.configure(state="normal")
                btn_restart.configure(state="normal")
                btn_load_image.configure(state="normal")
                set_exposure_widgets_state(False)
            else:
                status_pill.set_status("Bağlı (yeniden)", Colors.ACCENT_GREEN)
                btn_capture.configure(state="normal")
                btn_capture_process.configure(state="normal")
                btn_load_image.configure(state="normal")
                btn_restart.configure(state="normal")
                app.after(220, refresh_exposure_from_camera)

        threading.Thread(target=work, daemon=True).start()

    btn_restart.configure(command=do_restart)

    def on_closing():
        try:
            set_exposure_widgets_state(False)
        except Exception:
            pass
        try:
            camera_mgr.disconnect()
            camera_mgr.finalize_sdk()
        except Exception:
            pass
        app.destroy()

    app.protocol("WM_DELETE_WINDOW", on_closing)
    app.mainloop()


# ============================================================================
if __name__ == "__main__":
    run_gui()

