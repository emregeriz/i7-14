# -*- coding: utf-8 -*-
"""
Çoklu Kamera GUI (3x Hikrobot MV-CS200-10GC, GigE switch üzerinden)
====================================================================
Aynı anda 3 kameraya bağlanır, her biri için AYRI AYRI:
  * Canlı önizleme (opsiyonel)
  * Tek kare çekme (Çek butonu)
  * YOLO ile kasa tespiti + (varsa) zxing-cpp ile barkod okuma
Ve "Tümünü Çek" butonu ile 3 kamerayı paralel tetikler.

camera_gui_v2.py içindeki MVS SDK kurulumunu, DetectionProcessor'ü ve
yardımcı fonksiyonları tekrar kullanır.
"""

import os
import sys
import threading
import traceback
from ctypes import byref, cast, c_ubyte, memmove, memset, POINTER, sizeof
from datetime import datetime

import numpy as np
import cv2
import customtkinter as ctk
import tkinter as tk
from PIL import Image, ImageTk

# camera_gui_v2.py MVS SDK path ayarını, sabitleri ve yardımcıları yapar.
# Aynı dizinde olduğumuz için doğrudan import edebiliyoruz.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import camera_gui_v2 as cv2gui  # noqa: E402
from camera_gui_v2 import (  # noqa: E402
    MVS_AVAILABLE,
    MVCC_FLOATVALUE,
    Colors,
    DetectionProcessor,
    decoding_char,
    frame_to_numpy,
    pil_trento_logo_to_rgb,
    resolve_trento_logo_path,
    to_hex,
)

if MVS_AVAILABLE:
    from MvCameraControl_class import (  # noqa: E402
        MvCamera,
        MV_CC_DEVICE_INFO_LIST,
        MV_CC_DEVICE_INFO,
        MV_FRAME_OUT,
        MV_FRAME_OUT_INFO_EX,
    )
    from MvErrorDefine_const import *  # noqa: E402,F401,F403
    from CameraParams_header import (  # noqa: E402
        MV_ACCESS_Exclusive,
        MV_GIGE_DEVICE,
        MV_USB_DEVICE,
        MV_GENTL_CAMERALINK_DEVICE,
        MV_GENTL_CXP_DEVICE,
        MV_GENTL_XOF_DEVICE,
        MV_GENTL_GIGE_DEVICE,
        MV_TRIGGER_MODE_OFF,
    )

# ----------------------------------------------------------------------------
#  Ayarlar
# ----------------------------------------------------------------------------
MAX_CAMERAS = 3
SAVE_ROOT = os.path.join(_HERE, "images_multi")
os.makedirs(SAVE_ROOT, exist_ok=True)

PREVIEW_FPS = 5  # Canlı önizleme için hedef FPS (düşük tutuyoruz — 3 kamera + CPU)
PREVIEW_MAX_W = 520
PREVIEW_MAX_H = 360

# Kasa tekilleştirme (dedup) ayarları
# ----------------------------------------------------------------------------
# İki kamerada okunan aynı barkoddan kameralar arası (dx, dy) ofsetini çıkarır,
# ardından barkodu okunamamış kasaları da bu ofset ile eşleştirerek aynı
# kasanın iki kere sayılmasını önler.
DEDUP_TOLERANCE_RATIO = 0.5   # ortak barkodla hizalamada kasa boyutunun % oranı
DEDUP_MIN_SHARED = 1          # pair için gerekli min ortak barkod
DEDUP_BBOX_IOU = 0.25         # spatial eşleşme için min IoU



# ----------------------------------------------------------------------------
#  Tekil kamera tutucu
# ----------------------------------------------------------------------------
class SingleCamera:
    """Tek bir Hikrobot kamera için bağlantı + kare yakalama."""

    def __init__(self, dev_info, index, label, ip="", serial=""):
        self.dev_info = dev_info
        self.index = index
        self.label = label      # GUI başlığında görünecek kısa ad
        self.ip = ip
        self.serial = serial
        self.cam = None
        self.connected = False
        self.grabbing = False
        self.lock = threading.Lock()   # capture/grab serileştirmesi
        self.last_frame = None

    # ---- bağlantı ----
    def connect(self):
        with self.lock:
            if self.connected:
                return
            self.cam = MvCamera()
            ret = self.cam.MV_CC_CreateHandle(self.dev_info)
            if ret != 0:
                self.cam = None
                raise Exception(f"[{self.label}] Handle oluşturulamadı (ret={to_hex(ret)})")

            ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
            if ret != 0:
                self.cam.MV_CC_DestroyHandle()
                self.cam = None
                raise Exception(f"[{self.label}] Cihaz açılamadı (ret={to_hex(ret)})")

            if self.dev_info.nTLayerType in (MV_GIGE_DEVICE, MV_GENTL_GIGE_DEVICE):
                try:
                    nPacketSize = self.cam.MV_CC_GetOptimalPacketSize()
                    if int(nPacketSize) > 0:
                        self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
                except Exception:
                    pass

                # 3 kamera 1 Gbit switch'i paylaşıyor → bant sınırı koy
                # ~33 MB/s (264 Mbps) her kameraya, üçü = ~800 Mbps total.
                for key in ("DeviceLinkThroughputLimit", "GevSCPD"):
                    try:
                        if key == "DeviceLinkThroughputLimit":
                            self.cam.MV_CC_SetIntValue(key, 33_000_000)
                        else:
                            # Paketler arası bekleme (tick). Daha küçük = daha hızlı.
                            self.cam.MV_CC_SetIntValue(key, 2000)
                    except Exception:
                        pass
                try:
                    self.cam.MV_CC_SetBoolValue("DeviceLinkThroughputLimitMode", True)
                except Exception:
                    pass

            # ÇOK ÖNEMLİ: Çoklu kamera için Software Trigger kullan.
            # Aksi halde 3 kamera sürekli akışta switch'i doyurur → GetImageBuffer timeout.
            try:
                self.cam.MV_CC_SetEnumValue("TriggerMode", 1)           # On
            except Exception:
                pass
            try:
                self.cam.MV_CC_SetEnumValue("TriggerSource", 7)         # 7 = Software (Hikrobot)
            except Exception:
                # Yedek: string ile dene
                try:
                    self.cam.MV_CC_SetEnumValueByString("TriggerSource", "Software")
                except Exception:
                    pass
            try:
                self.cam.MV_CC_SetEnumValue("AcquisitionMode", 2)       # Continuous (ama trigger bekleyecek)
            except Exception:
                pass
            try:
                self.cam.MV_CC_SetEnumValue("ExposureAuto", 2)          # Continuous auto-exposure
            except Exception:
                pass

            self.connected = True
            self._start_grabbing_unlocked()
            if not self.grabbing:
                # Grabbing gerçekten başlamadıysa temizle ve hata fırlat
                try:
                    self.cam.MV_CC_CloseDevice()
                    self.cam.MV_CC_DestroyHandle()
                except Exception:
                    pass
                self.cam = None
                self.connected = False
                raise Exception(f"[{self.label}] MV_CC_StartGrabbing başarısız oldu")

    def _start_grabbing_unlocked(self):
        if self.grabbing or self.cam is None:
            return
        ret = self.cam.MV_CC_StartGrabbing()
        if ret == 0:
            self.grabbing = True

    def start_grabbing(self):
        with self.lock:
            self._start_grabbing_unlocked()

    def stop_grabbing(self):
        with self.lock:
            if self.grabbing and self.cam is not None:
                try:
                    self.cam.MV_CC_StopGrabbing()
                except Exception:
                    pass
                self.grabbing = False

    def disconnect(self):
        with self.lock:
            if self.cam is not None:
                if self.grabbing:
                    try:
                        self.cam.MV_CC_StopGrabbing()
                    except Exception:
                        pass
                    self.grabbing = False
                try:
                    self.cam.MV_CC_CloseDevice()
                except Exception:
                    pass
                try:
                    self.cam.MV_CC_DestroyHandle()
                except Exception:
                    pass
                self.cam = None
            self.connected = False

    # ---- pozlama (exposure) ----
    def read_exposure_info(self):
        """
        ExposureTime (µs) için bilgi döner.
        {key, min, max, cur, auto, error} — okuma kısmen başarısız olsa bile
        mümkün olan alanlar doldurulur. UI'ın kilitlenmemesi için asla None dönmez.
        """
        info = {"key": None, "min": None, "max": None, "cur": None, "auto": None, "error": None}
        if self.cam is None:
            info["error"] = "cam=None"
            return info
        if MVCC_FLOATVALUE is None:
            info["error"] = "MVCC_FLOATVALUE yok"
            return info

        try:
            st = MVCC_FLOATVALUE()
            last_ret = None
            for key in ("ExposureTime", "ExposureTimeAbs"):
                memset(byref(st), 0, sizeof(st))
                ret = self.cam.MV_CC_GetFloatValue(key, byref(st))
                last_ret = ret
                if ret == 0:
                    info["key"] = key
                    info["cur"] = float(st.fCurValue)
                    info["min"] = float(st.fMin)
                    info["max"] = float(st.fMax)
                    break
            if info["key"] is None and last_ret is not None:
                info["error"] = f"GetFloatValue ret={to_hex(last_ret)}"
        except Exception as e:
            info["error"] = f"float okuma hatası: {e}"

        # ExposureAuto oku (0=Off, 1=Once, 2=Continuous)
        try:
            from MvCameraControl_class import MVCC_ENUMVALUE
            ev = MVCC_ENUMVALUE()
            memset(byref(ev), 0, sizeof(ev))
            ret = self.cam.MV_CC_GetEnumValue("ExposureAuto", ev)
            if ret == 0:
                info["auto"] = int(ev.nCurValue)
        except Exception:
            pass
        return info

    def set_exposure_auto(self, continuous=True):
        """continuous=True → ExposureAuto=2, False → 0. (ret, hata_str)."""
        if self.cam is None:
            return False, "cam=None"
        try:
            mode = 2 if continuous else 0
            ret = self.cam.MV_CC_SetEnumValue("ExposureAuto", mode)
            if ret != 0:
                # fallback: string ile
                try:
                    ret = self.cam.MV_CC_SetEnumValueByString(
                        "ExposureAuto", "Continuous" if continuous else "Off"
                    )
                except Exception:
                    pass
            return ret == 0, to_hex(ret) if ret != 0 else None
        except Exception as e:
            return False, str(e)

    def set_exposure_us(self, exposure_us: float):
        """Manuel pozlama (µs). Auto önce kapatılır. (ret, hata_str)."""
        if self.cam is None:
            return False, "cam=None"
        try:
            self.cam.MV_CC_SetEnumValue("ExposureAuto", 0)
        except Exception:
            pass
        try:
            ret = self.cam.MV_CC_SetFloatValue("ExposureTime", float(exposure_us))
            if ret != 0:
                ret2 = self.cam.MV_CC_SetFloatValue("ExposureTimeAbs", float(exposure_us))
                if ret2 == 0:
                    return True, None
                return False, f"SetFloat ret={to_hex(ret)}/{to_hex(ret2)}"
            return True, None
        except Exception as e:
            return False, str(e)

    # ---- kare alma ----
    def capture(self, timeout_ms=5000):
        """Software trigger ile tek kare yakalar."""
        with self.lock:
            if not self.connected or self.cam is None:
                raise Exception(f"[{self.label}] Bağlı değil")
            if not self.grabbing:
                ret = self.cam.MV_CC_StartGrabbing()
                if ret != 0:
                    raise Exception(f"[{self.label}] Grabbing başlatılamadı (ret={to_hex(ret)})")
                self.grabbing = True

            # Software trigger komutu gönder
            try:
                ret_t = self.cam.MV_CC_SetCommandValue("TriggerSoftware")
                if ret_t != 0:
                    # Trigger On değilse bu başarısız olur; önizlemede trigger off olabilir.
                    pass
            except Exception:
                pass

            stOutFrame = MV_FRAME_OUT()
            memset(byref(stOutFrame), 0, sizeof(stOutFrame))
            ret = self.cam.MV_CC_GetImageBuffer(stOutFrame, int(timeout_ms))
            if ret != 0 or stOutFrame.pBufAddr is None:
                raise Exception(f"[{self.label}] Kare alınamadı (ret={to_hex(ret)})")

            try:
                frame_len = stOutFrame.stFrameInfo.nFrameLen
                buf_data = (c_ubyte * frame_len)()
                memmove(buf_data, stOutFrame.pBufAddr, frame_len)

                frame_info = MV_FRAME_OUT_INFO_EX()
                memmove(byref(frame_info), byref(stOutFrame.stFrameInfo),
                        sizeof(MV_FRAME_OUT_INFO_EX))
            finally:
                try:
                    self.cam.MV_CC_FreeImageBuffer(stOutFrame)
                except Exception:
                    pass

            np_image = frame_to_numpy(self.cam, buf_data, frame_info)
            if np_image is None:
                raise Exception(f"[{self.label}] Görüntü formatı çözümlenemedi")
            if len(np_image.shape) == 2:
                np_image = cv2.cvtColor(np_image, cv2.COLOR_GRAY2BGR)
            self.last_frame = np_image
            return np_image


# ----------------------------------------------------------------------------
#  Kasa tekilleştirme (birden fazla kamerada aynı kasayı 1 kere say)
# ----------------------------------------------------------------------------
class _UnionFind:
    """Basit union-find (disjoint-set) yapısı."""

    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


class CrateAggregator:
    """
    3 kameradan gelen kasa tespitlerini birleştirip gerçek toplam kasa sayısını
    hesaplar.

    Mantık:
      1) Aynı barkod birden çok kamerada okunduysa → aynı kasadır (birleşir).
      2) Ortak barkodlar kameralar arası (dx, dy) ofseti verir.
         Tespit edilen ofset + medyan kasa boyutu kullanılarak, barkodu
         okunamamış komşu kasalar (ortak barkodun altı/üstü/yanı) da
         spatial eşleştirmeyle tekilleştirilir.
      3) Hiç ortak barkod yoksa (kameralar kesişmiyor) dedup yapılmaz,
         tüm tespitler kendine özgü sayılır.
    """

    def __init__(
        self,
        tolerance_ratio: float = DEDUP_TOLERANCE_RATIO,
        min_shared: int = DEDUP_MIN_SHARED,
        bbox_iou: float = DEDUP_BBOX_IOU,
    ):
        self.tolerance_ratio = tolerance_ratio
        self.min_shared = min_shared
        self.bbox_iou = bbox_iou

    # ---- yardımcılar ----
    @staticmethod
    def _bbox_iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
        area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _flatten(self, camera_results):
        """Tüm kasaları global listeye düzleştir."""
        crates = []
        for cam_id, res in enumerate(camera_results):
            if res is None:
                continue
            for k in res.get("kasalar", []) or []:
                x1, y1, x2, y2 = k["bbox"]
                w = max(1.0, float(x2 - x1))
                h = max(1.0, float(y2 - y1))
                bcs = [str(b).strip() for b in (k.get("barkodlar") or []) if b]
                crates.append({
                    "cam": cam_id,
                    "bbox": (float(x1), float(y1), float(x2), float(y2)),
                    "cx": (float(x1) + float(x2)) / 2.0,
                    "cy": (float(y1) + float(y2)) / 2.0,
                    "w": w,
                    "h": h,
                    "barkodlar": bcs,
                })
        return crates

    def aggregate(self, camera_results):
        """
        camera_results: her kamera için process_image dönüşü (veya None).

        Döner:
          {
            "raw_total": int,        # kameraların tespit toplamı (çift sayımlı)
            "unique_total": int,     # tekilleştirilmiş gerçek kasa sayısı
            "duplicates": int,       # çift sayılan kasa adedi
            "unique_barcodes": int,  # tekil barkod sayısı
            "per_camera": [ {cam, kasa, barkod, okunamayan} ... ],
            "pair_offsets": { (a,b): {dx,dy,shared,tol_x,tol_y} },
            "shared_barcodes_total": int,
            "barkod_cakisan": int,   # barkod üzerinden tekilleştirilen adet
            "spatial_cakisan": int,  # spatial eşleştirmeyle tekilleştirilen
          }
        """
        crates = self._flatten(camera_results)
        n = len(crates)
        per_cam = []
        for cam_id, res in enumerate(camera_results):
            if res is None:
                per_cam.append({"cam": cam_id, "kasa": 0, "barkod": 0, "okunamayan": 0})
            else:
                per_cam.append({
                    "cam": cam_id,
                    "kasa": int(res.get("toplam_kasa", 0) or 0),
                    "barkod": int(res.get("okunan_barkod", 0) or 0),
                    "okunamayan": int(res.get("okunamayan_barkod", 0) or 0),
                })

        if n == 0:
            return {
                "raw_total": 0,
                "unique_total": 0,
                "duplicates": 0,
                "unique_barcodes": 0,
                "per_camera": per_cam,
                "pair_offsets": {},
                "shared_barcodes_total": 0,
                "barkod_cakisan": 0,
                "spatial_cakisan": 0,
            }

        uf = _UnionFind(n)

        # 1) Barkod tabanlı birleştirme
        barkod_to_idx = {}
        for i, c in enumerate(crates):
            for bc in c["barkodlar"]:
                barkod_to_idx.setdefault(bc, []).append(i)

        barkod_cakisan = 0
        for bc, idxs in barkod_to_idx.items():
            # Tüm idxs'leri ilkine bağla. Farklı kamerada aynı barkod varsa dup.
            cams_set = {crates[i]["cam"] for i in idxs}
            if len(cams_set) > 1:
                # Her ekstra kamera temsilcisi kadar çakışma sayılır
                barkod_cakisan += len(idxs) - len(cams_set) + (len(cams_set) - 1)
                # Not: len(idxs) - 1 basitçe de doğru sayıyı verir; aşağıda bunu
                # tekrar hesaplayacağımız için bu değer sadece log/gösterim için.
            for j in idxs[1:]:
                uf.union(idxs[0], j)

        # 2) Kamera çiftleri için ortak barkoddan (dx, dy) ofseti çıkar
        pair_offsets = {}
        # Kamera-id → (kasa_idx, barkod) map'i
        per_cam_bc = {}
        for i, c in enumerate(crates):
            for bc in c["barkodlar"]:
                per_cam_bc.setdefault((c["cam"], bc), []).append(i)

        cams_in = sorted({c["cam"] for c in crates})
        for a in cams_in:
            for b in cams_in:
                if a >= b:
                    continue
                # a ve b'de aynı barkodu olan kasa çiftleri
                shared = []
                # a kamerasının barkodlarını bul
                a_barkodlar = {bc: idxs for (cam, bc), idxs in per_cam_bc.items() if cam == a}
                b_barkodlar = {bc: idxs for (cam, bc), idxs in per_cam_bc.items() if cam == b}
                for bc, a_idxs in a_barkodlar.items():
                    if bc in b_barkodlar:
                        # Her iki taraftan ilk eşleşmeyi kullan (aynı kasa iki kere okunmuş olabilir)
                        ia = a_idxs[0]
                        ib = b_barkodlar[bc][0]
                        shared.append((ia, ib))

                if len(shared) < self.min_shared:
                    continue

                dxs = [crates[ib]["cx"] - crates[ia]["cx"] for ia, ib in shared]
                dys = [crates[ib]["cy"] - crates[ia]["cy"] for ia, ib in shared]
                ws = [crates[ia]["w"] for ia, _ in shared] + [crates[ib]["w"] for _, ib in shared]
                hs = [crates[ia]["h"] for ia, _ in shared] + [crates[ib]["h"] for _, ib in shared]
                dx = float(np.median(dxs))
                dy = float(np.median(dys))
                tol_x = float(np.median(ws)) * self.tolerance_ratio
                tol_y = float(np.median(hs)) * self.tolerance_ratio
                pair_offsets[(a, b)] = {
                    "dx": dx,
                    "dy": dy,
                    "shared": len(shared),
                    "tol_x": tol_x,
                    "tol_y": tol_y,
                }

        # 3) Ofset bilinen her çift için spatial eşleştirme
        spatial_cakisan = 0
        for (a, b), off in pair_offsets.items():
            a_indices = [i for i, c in enumerate(crates) if c["cam"] == a]
            b_indices = [i for i, c in enumerate(crates) if c["cam"] == b]
            used_b = set()
            # b'nin kasalarının a koordinat sistemine düzeltilmiş merkezleri:
            # b'deki (cx_b, cy_b) ↔ a'daki (cx_b - dx, cy_b - dy)
            for ia in a_indices:
                ca = crates[ia]
                best = None
                best_score = None
                for ib in b_indices:
                    if ib in used_b:
                        continue
                    cb = crates[ib]
                    # b'nin a'ya projeksiyonu
                    pcx = cb["cx"] - off["dx"]
                    pcy = cb["cy"] - off["dy"]
                    if abs(pcx - ca["cx"]) > off["tol_x"]:
                        continue
                    if abs(pcy - ca["cy"]) > off["tol_y"]:
                        continue
                    # bbox IoU kontrolü (projekte edilmiş bbox ile)
                    bx1 = cb["bbox"][0] - off["dx"]
                    by1 = cb["bbox"][1] - off["dy"]
                    bx2 = cb["bbox"][2] - off["dx"]
                    by2 = cb["bbox"][3] - off["dy"]
                    iou = self._bbox_iou(ca["bbox"], (bx1, by1, bx2, by2))
                    score_d = abs(pcx - ca["cx"]) + abs(pcy - ca["cy"])
                    if iou < self.bbox_iou:
                        # Barkodu okunmuş ve barkodlar eşleşmiyorsa birleştirme
                        if ca["barkodlar"] and cb["barkodlar"]:
                            if not set(ca["barkodlar"]) & set(cb["barkodlar"]):
                                continue
                        else:
                            continue
                    # En iyi adayı seç
                    if best_score is None or score_d < best_score:
                        best_score = score_d
                        best = ib
                if best is not None:
                    if uf.find(ia) != uf.find(best):
                        uf.union(ia, best)
                        spatial_cakisan += 1
                    used_b.add(best)

        # Gerçek tekil sayı
        roots = {uf.find(i) for i in range(n)}
        unique_total = len(roots)
        raw_total = n
        duplicates = raw_total - unique_total

        # Tekil barkod: farklı kök sayısı barkoda sahip olanlar için
        unique_barcodes = len({bc for bc in barkod_to_idx.keys()})
        # Kameralar arası ortak barkod (en az 2 kamerada geçen)
        shared_bc_total = sum(
            1
            for bc, idxs in barkod_to_idx.items()
            if len({crates[i]["cam"] for i in idxs}) > 1
        )

        return {
            "raw_total": raw_total,
            "unique_total": unique_total,
            "duplicates": duplicates,
            "unique_barcodes": unique_barcodes,
            "per_camera": per_cam,
            "pair_offsets": pair_offsets,
            "shared_barcodes_total": shared_bc_total,
            "barkod_cakisan": barkod_cakisan,
            "spatial_cakisan": spatial_cakisan,
        }


# ----------------------------------------------------------------------------
#  Çoklu kamera yöneticisi
# ----------------------------------------------------------------------------
class MultiCameraManager:
    def __init__(self):
        self.sdk_initialized = False
        self.device_list = None
        self.cameras: list[SingleCamera] = []

    def initialize_sdk(self):
        if not MVS_AVAILABLE:
            raise Exception("MVS SDK import edilemedi — Hikrobot MVS kurulu mu?")
        if not self.sdk_initialized:
            MvCamera.MV_CC_Initialize()
            self.sdk_initialized = True

    def finalize_sdk(self):
        if self.sdk_initialized:
            try:
                MvCamera.MV_CC_Finalize()
            except Exception:
                pass
            self.sdk_initialized = False

    def enum_devices(self):
        """Bağlı tüm kameraları listeler. (dev_info, label, ip, serial) tuple'ları döner."""
        self.device_list = MV_CC_DEVICE_INFO_LIST()
        tlayer_type = (
            MV_GIGE_DEVICE | MV_USB_DEVICE
            | MV_GENTL_CAMERALINK_DEVICE | MV_GENTL_CXP_DEVICE | MV_GENTL_XOF_DEVICE
        )
        ret = MvCamera.MV_CC_EnumDevices(tlayer_type, self.device_list)
        if ret != 0:
            raise Exception(f"Cihaz taraması başarısız (ret={to_hex(ret)})")

        found = []
        for i in range(self.device_list.nDeviceNum):
            dev_info = cast(self.device_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            if dev_info.nTLayerType in (MV_GIGE_DEVICE, MV_GENTL_GIGE_DEVICE):
                model = decoding_char(dev_info.SpecialInfo.stGigEInfo.chModelName)
                serial = decoding_char(dev_info.SpecialInfo.stGigEInfo.chSerialNumber)
                ip_raw = dev_info.SpecialInfo.stGigEInfo.nCurrentIp
                ip_str = "%d.%d.%d.%d" % (
                    (ip_raw & 0xFF000000) >> 24,
                    (ip_raw & 0x00FF0000) >> 16,
                    (ip_raw & 0x0000FF00) >> 8,
                    ip_raw & 0x000000FF,
                )
                label = f"{model} ({serial})"
                found.append({
                    "dev_info": dev_info,
                    "index": i,
                    "label": label,
                    "ip": ip_str,
                    "serial": serial,
                    "model": model,
                })
            elif dev_info.nTLayerType == MV_USB_DEVICE:
                model = decoding_char(dev_info.SpecialInfo.stUsb3VInfo.chModelName)
                serial = decoding_char(dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber)
                found.append({
                    "dev_info": dev_info,
                    "index": i,
                    "label": f"{model} ({serial})",
                    "ip": "",
                    "serial": serial,
                    "model": model,
                })
        return found

    def connect_selected(self, infos):
        """Verilen info listesinden SingleCamera örnekleri oluşturur ve bağlanır."""
        self.disconnect_all()
        cams = []
        for info in infos:
            cam = SingleCamera(
                dev_info=info["dev_info"],
                index=info["index"],
                label=info["label"],
                ip=info.get("ip", ""),
                serial=info.get("serial", ""),
            )
            cam.connect()
            cams.append(cam)
        self.cameras = cams
        return cams

    def disconnect_all(self):
        for c in self.cameras:
            try:
                c.disconnect()
            except Exception:
                pass
        self.cameras = []


# ----------------------------------------------------------------------------
#  Kamera panel widget'ı (her kamera için bir tile)
# ----------------------------------------------------------------------------
class CameraPanel(ctk.CTkFrame):
    def __init__(self, master, title, subtitle, on_capture, **kwargs):
        super().__init__(
            master,
            fg_color=Colors.BG_CARD,
            corner_radius=12,
            border_width=1,
            border_color=Colors.BORDER,
            **kwargs,
        )
        self._on_capture = on_capture
        self._photo_ref = None
        self._last_np = None
        self._last_annotated = None
        self.last_result = None  # process_image son sonucu (agrege için)

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=14, pady=(10, 4))

        ctk.CTkLabel(
            header, text=title,
            font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"),
            text_color=Colors.TEXT_PRIMARY, anchor="w",
        ).pack(side="left")

        self.status_dot = ctk.CTkLabel(
            header, text="●",
            font=ctk.CTkFont(size=14),
            text_color=Colors.TEXT_MUTED, width=18,
        )
        self.status_dot.pack(side="right")

        self.subtitle = ctk.CTkLabel(
            self, text=subtitle,
            font=ctk.CTkFont(family="Consolas", size=11),
            text_color=Colors.TEXT_SECONDARY, anchor="w",
        )
        self.subtitle.pack(fill="x", padx=14, pady=(0, 6))

        self.img_wrap = ctk.CTkFrame(
            self, fg_color=Colors.IMAGE_CANVAS, corner_radius=8,
            border_width=1, border_color=Colors.BORDER,
            width=PREVIEW_MAX_W, height=PREVIEW_MAX_H,
        )
        self.img_wrap.pack(padx=14, pady=6)
        self.img_wrap.pack_propagate(False)

        self.img_label = tk.Label(
            self.img_wrap, bg=Colors.IMAGE_CANVAS, fg=Colors.TEXT_MUTED,
            text="Görüntü bekleniyor…", font=("Segoe UI", 12),
        )
        self.img_label.pack(fill="both", expand=True)

        stats_row = ctk.CTkFrame(self, fg_color="transparent")
        stats_row.pack(fill="x", padx=14, pady=(6, 4))
        self.kasa_var = self._stat_card(stats_row, "KASA", Colors.ACCENT_GREEN)
        self.ok_var = self._stat_card(stats_row, "BARKOD", Colors.ACCENT_CYAN)
        self.fail_var = self._stat_card(stats_row, "HATA", Colors.ACCENT_RED)

        # --- Exposure (Pozlama) paneli ---
        exp_card = ctk.CTkFrame(
            self, fg_color=Colors.BG_SURFACE, corner_radius=8,
            border_width=1, border_color=Colors.BORDER,
        )
        exp_card.pack(fill="x", padx=14, pady=(8, 4))

        exp_header = ctk.CTkFrame(exp_card, fg_color="transparent")
        exp_header.pack(fill="x", padx=10, pady=(8, 4))
        ctk.CTkLabel(
            exp_header, text="⚙  Pozlama (Exposure)",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color=Colors.TEXT_SECONDARY,
        ).pack(side="left")
        self.exp_auto_var = ctk.BooleanVar(value=True)
        self.exp_auto_switch = ctk.CTkSwitch(
            exp_header, text="Otomatik",
            variable=self.exp_auto_var,
            command=self._on_exp_auto_toggle,
            progress_color=Colors.ACCENT_GREEN,
            font=ctk.CTkFont(family="Segoe UI", size=11),
        )
        self.exp_auto_switch.pack(side="right")

        row = ctk.CTkFrame(exp_card, fg_color="transparent")
        row.pack(fill="x", padx=10, pady=(0, 8))
        ctk.CTkLabel(
            row, text="µs:",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color=Colors.TEXT_SECONDARY,
        ).pack(side="left", padx=(0, 6))

        self.exp_entry = ctk.CTkEntry(
            row, width=90,
            font=ctk.CTkFont(family="Consolas", size=11),
            fg_color=Colors.BG_INPUT, border_color=Colors.BORDER,
            text_color=Colors.TEXT_PRIMARY,
            placeholder_text="—",
        )
        self.exp_entry.pack(side="left")

        self.exp_range_lbl = ctk.CTkLabel(
            row, text="aralık: —",
            font=ctk.CTkFont(family="Segoe UI", size=10),
            text_color=Colors.TEXT_MUTED,
        )
        self.exp_range_lbl.pack(side="left", padx=8)

        self.exp_apply_btn = ctk.CTkButton(
            row, text="Uygula", width=68, height=28,
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            fg_color=Colors.BTN_PRIMARY, hover_color=Colors.BTN_HOVER,
            corner_radius=6,
            command=self._on_exp_apply,
        )
        self.exp_apply_btn.pack(side="right")
        self.exp_entry.bind("<Return>", lambda e: self._on_exp_apply())

        self.capture_btn = ctk.CTkButton(
            self, text="📷 ÇEK ve İŞLE",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            fg_color=Colors.BTN_BLUE, hover_color=Colors.BTN_BLUE_HOVER,
            corner_radius=8, height=40,
            command=lambda: self._on_capture(self),
        )
        self.capture_btn.pack(fill="x", padx=14, pady=(4, 10))

    def _stat_card(self, parent, title, color):
        card = ctk.CTkFrame(
            parent, fg_color=Colors.BG_SURFACE, corner_radius=8,
            border_width=1, border_color=Colors.BORDER,
        )
        card.pack(side="left", fill="x", expand=True, padx=3)
        ctk.CTkLabel(
            card, text=title,
            font=ctk.CTkFont(family="Segoe UI", size=10),
            text_color=Colors.TEXT_SECONDARY,
        ).pack(pady=(6, 0))
        val = ctk.CTkLabel(
            card, text="—",
            font=ctk.CTkFont(family="Segoe UI", size=20, weight="bold"),
            text_color=color,
        )
        val.pack(pady=(0, 6))
        return val

    # ---- API ----
    def set_status(self, text, color):
        self.status_dot.configure(text_color=color)

    def set_subtitle(self, text):
        self.subtitle.configure(text=text)

    def set_busy(self, busy: bool):
        self.capture_btn.configure(
            state="disabled" if busy else "normal",
            text="⏳ İşleniyor…" if busy else "📷 ÇEK ve İŞLE",
        )

    def show_image(self, np_bgr):
        """BGR numpy görüntüyü göster (önizleme)."""
        if np_bgr is None:
            return
        self._last_np = np_bgr
        self._render_current()

    def show_annotated(self, np_bgr):
        self._last_annotated = np_bgr
        self._last_np = np_bgr
        self._render_current()

    def _render_current(self):
        img = self._last_np
        if img is None:
            return
        try:
            if len(img.shape) == 2:
                rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            w, h = pil.size
            box_w = self.img_wrap.winfo_width() or PREVIEW_MAX_W
            box_h = self.img_wrap.winfo_height() or PREVIEW_MAX_H
            if box_w < 50 or box_h < 50:
                box_w, box_h = PREVIEW_MAX_W, PREVIEW_MAX_H
            ratio = min(box_w / w, box_h / h, 1.0)
            pil = pil.resize((max(1, int(w * ratio)), max(1, int(h * ratio))), Image.LANCZOS)
            self._photo_ref = ImageTk.PhotoImage(pil)
            self.img_label.configure(image=self._photo_ref, text="")
        except Exception:
            pass

    def set_stats(self, kasa=None, ok=None, fail=None):
        if kasa is not None:
            self.kasa_var.configure(text=str(kasa))
        if ok is not None:
            self.ok_var.configure(text=str(ok))
        if fail is not None:
            color = Colors.ACCENT_RED if fail > 0 else Colors.ACCENT_GREEN
            self.fail_var.configure(text=str(fail), text_color=color)

    # ---- Exposure UI ----
    def _get_cam(self):
        return getattr(self, "cam_ref", None)

    def refresh_exposure(self):
        """Kameradan güncel pozlama bilgilerini okuyup UI'a yaz.
        Okuma başarısız olsa bile kontrolleri AKTİF bırakır; kullanıcı manuel değer girebilir."""
        cam = self._get_cam()
        if cam is None:
            return
        info = cam.read_exposure_info()
        log = getattr(self, "on_log", None)

        # Aralığı göster
        mn, mx, cur = info.get("min"), info.get("max"), info.get("cur")
        if mn is not None and mx is not None and mx > mn:
            self.exp_range_lbl.configure(text=f"aralık: {int(mn)}–{int(mx)}")
        elif info.get("error"):
            self.exp_range_lbl.configure(text=f"aralık: — ({info['error']})")
            if log:
                log(f"[{cam.label}] Pozlama aralığı okunamadı: {info['error']}")
        else:
            self.exp_range_lbl.configure(text="aralık: —")

        # Mevcut değeri göster
        if cur is not None:
            self.exp_entry.delete(0, "end")
            self.exp_entry.insert(0, str(int(cur)))

        # Auto durumunu UI'a yansıt — kullanıcı yine de kapatıp açabilir
        auto = (info.get("auto") == 2) if info.get("auto") is not None else True
        self.exp_auto_var.set(auto)
        # Kontroller HER ZAMAN etkin — sadece entry, auto'ya göre disable olur
        self.exp_auto_switch.configure(state="normal")
        self.exp_apply_btn.configure(state="normal")
        self._update_entry_state(auto)

    def _update_entry_state(self, auto: bool):
        # Auto açıksa entry disabled, apply disabled
        if auto:
            self.exp_entry.configure(state="disabled")
            self.exp_apply_btn.configure(state="disabled")
        else:
            self.exp_entry.configure(state="normal")
            self.exp_apply_btn.configure(state="normal")

    def _on_exp_auto_toggle(self):
        cam = self._get_cam()
        log = getattr(self, "on_log", None)
        if cam is None:
            if log:
                log("Pozlama: kamera yok")
            return
        auto = bool(self.exp_auto_var.get())
        ok, err = cam.set_exposure_auto(continuous=auto)
        self._update_entry_state(auto)
        if log:
            mark = "✔" if ok else f"✘ ({err})"
            log(f"[{cam.label}] Pozlama: {'Otomatik' if auto else 'Manuel'} {mark}")
        if not auto:
            # Manuel'e geçtik → mevcut değeri yeniden oku (auto kapanınca sabitlenir)
            info = cam.read_exposure_info()
            if info and info.get("cur") is not None:
                self.exp_entry.delete(0, "end")
                self.exp_entry.insert(0, str(int(info["cur"])))

    def _on_exp_apply(self):
        cam = self._get_cam()
        log = getattr(self, "on_log", None)
        if cam is None:
            if log:
                log("Pozlama: kamera yok")
            return
        val_str = self.exp_entry.get().strip()
        try:
            val = float(val_str)
        except ValueError:
            if log:
                log(f"[{cam.label}] Geçersiz pozlama değeri: {val_str!r}")
            return
        ok, err = cam.set_exposure_us(val)
        # Değer uygulandıktan sonra auto zorunlu kapalı oldu
        self.exp_auto_var.set(False)
        self._update_entry_state(False)
        if log:
            mark = "✔" if ok else f"✘ ({err})"
            log(f"[{cam.label}] Pozlama = {int(val)} µs {mark}")


# ----------------------------------------------------------------------------
#  Ana GUI
# ----------------------------------------------------------------------------
class MultiCamApp:
    def __init__(self):
        self.manager = MultiCameraManager()
        self.detector = DetectionProcessor()
        self.aggregator = CrateAggregator()
        self.panels: list[CameraPanel] = []
        self.preview_running = False
        self.preview_thread = None
        self._gui_lock = threading.Lock()
        self._summary_widgets = {}  # palet özet kartı değer etiketleri

        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("green")
        self.app = ctk.CTk()
        self.app.title("Çoklu Kamera + YOLO — Hikrobot (3x)")
        self.app.geometry("1700x1000")
        self.app.minsize(1400, 820)
        self.app.configure(fg_color=Colors.BG_DARK)

        self._build_ui()
        self.app.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---- UI ----
    def _build_ui(self):
        # Toolbar
        toolbar = ctk.CTkFrame(self.app, fg_color=Colors.BG_CARD, corner_radius=0, height=70)
        toolbar.pack(fill="x")
        toolbar.pack_propagate(False)

        inner = ctk.CTkFrame(toolbar, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=16, pady=10)

        logo_wrap = ctk.CTkFrame(inner, fg_color="transparent")
        logo_wrap.pack(side="left", padx=(0, 16), pady=2)
        self.app._trento_logo_ctk = None
        logo_path = resolve_trento_logo_path()
        if logo_path:
            try:
                pil_logo = pil_trento_logo_to_rgb(Image.open(logo_path), Colors.BG_CARD)
                max_h = 40
                lw, lh = pil_logo.size
                new_w = max(1, int(lw * max_h / lh))
                pil_logo = pil_logo.resize((new_w, max_h), Image.Resampling.LANCZOS)
                self.app._trento_logo_ctk = ctk.CTkImage(
                    light_image=pil_logo, dark_image=pil_logo, size=(new_w, max_h)
                )
                ctk.CTkLabel(
                    logo_wrap,
                    image=self.app._trento_logo_ctk,
                    text="",
                    fg_color="transparent",
                ).pack()
            except OSError:
                self.app._trento_logo_ctk = None

        ctk.CTkLabel(
            inner, text="Çoklu Kamera Kasa Sayım",
            font=ctk.CTkFont(family="Segoe UI", size=18, weight="bold"),
            text_color=Colors.BRAND_GREEN_SOFT,
        ).pack(side="left")

        self.status_label = ctk.CTkLabel(
            inner, text="● Başlatılmadı",
            font=ctk.CTkFont(family="Segoe UI", size=13),
            text_color=Colors.TEXT_MUTED,
        )
        self.status_label.pack(side="left", padx=20)

        self.connect_btn = ctk.CTkButton(
            inner, text="🔌 Kameraları Tara & Bağlan",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            fg_color=Colors.BTN_PRIMARY, hover_color=Colors.BTN_HOVER,
            corner_radius=8, height=36, width=220,
            command=self._on_connect,
        )
        self.connect_btn.pack(side="right", padx=(8, 0))

        self.capture_all_btn = ctk.CTkButton(
            inner, text="📸 TÜMÜNÜ ÇEK",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            fg_color=Colors.BTN_BLUE, hover_color=Colors.BTN_BLUE_HOVER,
            corner_radius=8, height=36, width=180,
            command=self._on_capture_all, state="disabled",
        )
        self.capture_all_btn.pack(side="right", padx=8)

        self.preview_switch = ctk.CTkSwitch(
            inner, text="Canlı Önizleme",
            command=self._toggle_preview,
            progress_color=Colors.ACCENT_CYAN,
        )
        self.preview_switch.pack(side="right", padx=8)

        # Kamera panelleri için alan
        self.panels_frame = ctk.CTkFrame(self.app, fg_color=Colors.BG_DARK)
        self.panels_frame.pack(fill="both", expand=True, padx=14, pady=(10, 6))

        # Palet özet kartı (tekilleştirilmiş gerçek kasa sayısı)
        self._build_summary_card()

        # Log
        log_wrap = ctk.CTkFrame(self.app, fg_color=Colors.BG_CARD, corner_radius=8,
                                border_width=1, border_color=Colors.BORDER, height=180)
        log_wrap.pack(fill="x", padx=14, pady=(0, 12))
        log_wrap.pack_propagate(False)

        header = ctk.CTkFrame(log_wrap, fg_color="transparent")
        header.pack(fill="x", padx=10, pady=(6, 2))
        ctk.CTkLabel(header, text="📝 Log",
                     font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
                     text_color=Colors.TEXT_SECONDARY).pack(side="left")

        self.log_text = tk.Text(
            log_wrap, bg=Colors.TERMINAL_BG, fg=Colors.TERMINAL_FG,
            font=("Consolas", 10), bd=0, relief="flat",
            insertbackground=Colors.TERMINAL_FG, wrap="word",
        )
        self.log_text.pack(fill="both", expand=True, padx=10, pady=(0, 8))
        self.log_text.configure(state="disabled")

    # ---- palet özet kartı ----
    def _build_summary_card(self):
        wrap = ctk.CTkFrame(
            self.app, fg_color=Colors.BG_CARD, corner_radius=10,
            border_width=1, border_color=Colors.BORDER, height=110,
        )
        wrap.pack(fill="x", padx=14, pady=(0, 6))
        wrap.pack_propagate(False)

        header = ctk.CTkFrame(wrap, fg_color="transparent")
        header.pack(fill="x", padx=12, pady=(8, 2))
        ctk.CTkLabel(
            header, text="📦 Palet Özeti — Tekilleştirilmiş Kasa Sayımı",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            text_color=Colors.TEXT_PRIMARY,
        ).pack(side="left")
        self.summary_info_lbl = ctk.CTkLabel(
            header, text="—",
            font=ctk.CTkFont(family="Consolas", size=10),
            text_color=Colors.TEXT_MUTED,
        )
        self.summary_info_lbl.pack(side="right")

        row = ctk.CTkFrame(wrap, fg_color="transparent")
        row.pack(fill="both", expand=True, padx=12, pady=(0, 8))

        def _card(parent, title, color):
            card = ctk.CTkFrame(
                parent, fg_color=Colors.BG_SURFACE, corner_radius=8,
                border_width=1, border_color=Colors.BORDER,
            )
            card.pack(side="left", fill="both", expand=True, padx=4)
            ctk.CTkLabel(
                card, text=title,
                font=ctk.CTkFont(family="Segoe UI", size=10),
                text_color=Colors.TEXT_SECONDARY,
            ).pack(pady=(6, 0))
            val = ctk.CTkLabel(
                card, text="—",
                font=ctk.CTkFont(family="Segoe UI", size=24, weight="bold"),
                text_color=color,
            )
            val.pack(pady=(0, 6))
            return val

        self._summary_widgets["raw"] = _card(row, "HAM TOPLAM (çift sayımlı)", Colors.TEXT_SECONDARY)
        self._summary_widgets["dup"] = _card(row, "ÇAKIŞAN (düşülen)", Colors.ACCENT_YELLOW)
        self._summary_widgets["unique"] = _card(row, "GERÇEK KASA", Colors.ACCENT_GREEN)
        self._summary_widgets["ubc"] = _card(row, "TEKİL BARKOD", Colors.ACCENT_CYAN)
        self._summary_widgets["shared"] = _card(row, "KAMERALAR ARASI ORTAK BARKOD", Colors.ACCENT_BLUE)

    def _update_summary(self, agg):
        def _set():
            w = self._summary_widgets
            if not w:
                return
            w["raw"].configure(text=str(agg.get("raw_total", 0)))
            w["dup"].configure(text=str(agg.get("duplicates", 0)))
            w["unique"].configure(text=str(agg.get("unique_total", 0)))
            w["ubc"].configure(text=str(agg.get("unique_barcodes", 0)))
            w["shared"].configure(text=str(agg.get("shared_barcodes_total", 0)))

            # Açıklama satırı
            per = agg.get("per_camera", [])
            parts = []
            for p in per:
                parts.append(
                    f"Kam{p['cam']+1}: {p['kasa']} kasa / {p['barkod']} barkod"
                )
            info = "  |  ".join(parts) if parts else "—"
            pair_off = agg.get("pair_offsets", {})
            if pair_off:
                off_parts = []
                for (a, b), o in pair_off.items():
                    off_parts.append(
                        f"K{a+1}↔K{b+1}: {o['shared']} ortak, Δ=({int(o['dx'])},{int(o['dy'])})"
                    )
                info += "   ||   " + "  ;  ".join(off_parts)
            self.summary_info_lbl.configure(text=info)
        try:
            self.app.after(0, _set)
        except Exception:
            pass

    def _reset_summary(self):
        def _set():
            for k in self._summary_widgets:
                self._summary_widgets[k].configure(text="—")
            if hasattr(self, "summary_info_lbl"):
                self.summary_info_lbl.configure(text="—")
        try:
            self.app.after(0, _set)
        except Exception:
            pass

    # ---- log ----
    def log(self, msg):
        def _append():
            self.log_text.configure(state="normal")
            self.log_text.insert("end", f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
            self.log_text.see("end")
            self.log_text.configure(state="disabled")
        try:
            self.app.after(0, _append)
        except Exception:
            print(msg)

    def set_status(self, text, color=Colors.TEXT_SECONDARY):
        def _set():
            self.status_label.configure(text=f"● {text}", text_color=color)
        try:
            self.app.after(0, _set)
        except Exception:
            pass

    # ---- bağlan ----
    def _on_connect(self):
        threading.Thread(target=self._connect_worker, daemon=True).start()

    def _connect_worker(self):
        try:
            self.set_status("Bağlanıyor…", Colors.ACCENT_YELLOW)
            self.log("MVS SDK başlatılıyor…")
            self.manager.initialize_sdk()
            self.log("Kameralar taranıyor…")
            infos = self.manager.enum_devices()
            self.log(f"{len(infos)} cihaz bulundu.")
            for info in infos:
                ip_str = f" IP={info['ip']}" if info.get("ip") else ""
                self.log(f"  • {info['label']}{ip_str}")

            gige = [i for i in infos if i.get("ip")]
            selected = gige[:MAX_CAMERAS] if gige else infos[:MAX_CAMERAS]
            if not selected:
                raise Exception("Bağlanacak kamera bulunamadı!")

            self.log(f"{len(selected)} kameraya bağlanılıyor…")
            self.manager.connect_selected(selected)
            self.log("✔ Tüm kameralar bağlandı.")

            self.app.after(0, lambda: self._build_panels(self.manager.cameras))
            self.set_status(f"{len(self.manager.cameras)} kamera bağlı", Colors.ACCENT_GREEN)
            self.app.after(0, lambda: self.capture_all_btn.configure(state="normal"))

            # YOLO modelini arka planda yükle
            self.log("YOLO modeli yükleniyor…")
            self.detector.load_model(log_fn=self.log)
            self.log("✔ Model hazır.")
        except Exception as e:
            self.log(f"HATA: {e}")
            self.log(traceback.format_exc())
            self.set_status("Hata", Colors.ACCENT_RED)

    # ---- paneller ----
    def _build_panels(self, cameras):
        for w in self.panels_frame.winfo_children():
            w.destroy()
        self.panels = []

        n = max(1, len(cameras))
        for i in range(n):
            self.panels_frame.grid_columnconfigure(i, weight=1, uniform="cols")
        self.panels_frame.grid_rowconfigure(0, weight=1)

        for i, cam in enumerate(cameras):
            subtitle = f"IP: {cam.ip or '—'}   |   SN: {cam.serial or '—'}"
            panel = CameraPanel(
                self.panels_frame,
                title=f"Kamera {i+1} — {cam.label}",
                subtitle=subtitle,
                on_capture=self._on_single_capture,
            )
            panel.grid(row=0, column=i, sticky="nsew", padx=6, pady=4)
            panel.cam_ref = cam  # paneli → kamera bağla
            panel.on_log = self.log
            panel.set_status("bağlı", Colors.ACCENT_GREEN)
            panel.refresh_exposure()
            self.panels.append(panel)

    # ---- önizleme ----
    def _toggle_preview(self):
        if self.preview_switch.get():
            if not self.manager.cameras:
                self.log("Önizleme için önce kameralara bağlan.")
                self.preview_switch.deselect()
                return
            self._start_preview()
        else:
            self._stop_preview()

    def _start_preview(self):
        if self.preview_running:
            return
        self.preview_running = True
        self.preview_thread = threading.Thread(target=self._preview_loop, daemon=True)
        self.preview_thread.start()
        self.log("Canlı önizleme başlatıldı.")

    def _stop_preview(self):
        self.preview_running = False
        self.log("Canlı önizleme durduruldu.")

    def _preview_loop(self):
        import time
        target_dt = 1.0 / max(1, PREVIEW_FPS)
        while self.preview_running:
            t0 = time.time()
            for panel in list(self.panels):
                if not self.preview_running:
                    break
                cam = getattr(panel, "cam_ref", None)
                if cam is None or not cam.connected:
                    continue
                # Panel meşgulse (YOLO işlerken) atla
                if panel.capture_btn.cget("state") == "disabled":
                    continue
                try:
                    frame = cam.capture(timeout_ms=800)
                    self.app.after(0, lambda p=panel, f=frame: p.show_image(f))
                except Exception:
                    pass
            dt = time.time() - t0
            if dt < target_dt:
                time.sleep(target_dt - dt)

    # ---- tek kamera çek ----
    def _on_single_capture(self, panel: CameraPanel):
        cam = getattr(panel, "cam_ref", None)
        if cam is None:
            self.log("Panel için kamera yok.")
            return
        panel.set_busy(True)
        threading.Thread(
            target=self._capture_and_process, args=(panel, cam), daemon=True
        ).start()

    def _on_capture_all(self):
        if not self.panels:
            return
        self.capture_all_btn.configure(state="disabled")
        self.log("=== TÜM KAMERALARDAN EŞZAMANLI ÇEKİM ===")
        self._reset_summary()
        # Eski sonuçları temizle ki agrege eski veriyle karışmasın
        for p in self.panels:
            p.last_result = None

        threads = []
        for panel in self.panels:
            cam = getattr(panel, "cam_ref", None)
            if cam is None:
                continue
            panel.set_busy(True)
            t = threading.Thread(
                target=self._capture_and_process, args=(panel, cam), daemon=True
            )
            t.start()
            threads.append(t)

        def _wait_and_restore():
            for t in threads:
                t.join()
            self.app.after(0, lambda: self.capture_all_btn.configure(state="normal"))
            self.log("=== Tüm kameralar tamamlandı ===")
            # Tekilleştirme (dedup) çalıştır
            try:
                self._run_aggregation()
            except Exception as e:
                self.log(f"Özet hesaplama hatası: {e}")
                self.log(traceback.format_exc())

        threading.Thread(target=_wait_and_restore, daemon=True).start()

    def _run_aggregation(self):
        """Panellerdeki son sonuçları birleştir, palet özet kartını güncelle."""
        results = [getattr(p, "last_result", None) for p in self.panels]
        if not any(r for r in results):
            self.log("Özet: henüz sonuç yok.")
            return
        agg = self.aggregator.aggregate(results)
        self._update_summary(agg)

        # Özet log
        self.log("─" * 50)
        self.log("PALET ÖZETİ (tekilleştirme sonrası)")
        for p in agg["per_camera"]:
            self.log(
                f"  Kamera {p['cam']+1}: {p['kasa']} kasa "
                f"(barkod={p['barkod']}, okunamayan={p['okunamayan']})"
            )
        self.log(f"  Ham toplam (çift sayımlı) : {agg['raw_total']}")
        self.log(f"  Kameralar arası ortak bc  : {agg['shared_barcodes_total']}")
        self.log(f"  Barkoddan tekilleşen      : {agg['barkod_cakisan']}")
        self.log(f"  Spatial tekilleşen        : {agg['spatial_cakisan']}")
        self.log(f"  Çakışan toplam (düşülen)  : {agg['duplicates']}")
        self.log(f"  >>> GERÇEK KASA SAYISI    : {agg['unique_total']} <<<")
        if agg["pair_offsets"]:
            for (a, b), off in agg["pair_offsets"].items():
                self.log(
                    f"  Kam{a+1}↔Kam{b+1} hizalama: "
                    f"Δ=({int(off['dx'])},{int(off['dy'])}) "
                    f"ortak={off['shared']} tol=({int(off['tol_x'])},{int(off['tol_y'])})"
                )
        self.log("─" * 50)

    def _capture_and_process(self, panel: CameraPanel, cam: SingleCamera):
        try:
            self.log(f"[{cam.label}] Kare yakalanıyor…")
            frame = cam.capture(timeout_ms=5000)
            self.log(f"[{cam.label}] Kare alındı {frame.shape}. YOLO işleniyor…")
            self.app.after(0, lambda p=panel, f=frame: p.show_image(f))

            # Her kamera için ayrı klasör
            cam_dir = os.path.join(SAVE_ROOT, (cam.serial or cam.label).replace(" ", "_"))
            os.makedirs(cam_dir, exist_ok=True)

            # Orijinal kareyi kaydet
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            raw_path = os.path.join(cam_dir, f"raw_{ts}.jpg")
            cv2.imwrite(raw_path, frame)

            # Global detector, ama tek tek çağrıldığı için sorun olmaz
            # (YOLO predict thread-safe değil → kilit ile serileştir)
            with self._gui_lock:
                result = self.detector.process_image(
                    frame, cam_dir,
                    log_fn=lambda m, lbl=cam.label: self.log(f"[{lbl}] {m}"),
                )

            # Panelde son sonucu sakla — palet özeti (dedup) için kullanılacak
            panel.last_result = result

            panel.set_stats(
                kasa=result["toplam_kasa"],
                ok=result["okunan_barkod"],
                fail=result["okunamayan_barkod"],
            )
            annotated = result.get("annotated_image")
            if annotated is not None:
                self.app.after(0, lambda p=panel, a=annotated: p.show_annotated(a))

            self.log(
                f"[{cam.label}] ✔ Kasa={result['toplam_kasa']}  "
                f"Barkod={result['okunan_barkod']}  "
                f"Okunamayan={result['okunamayan_barkod']}"
            )
        except Exception as e:
            self.log(f"[{cam.label}] HATA: {e}")
        finally:
            self.app.after(0, lambda: panel.set_busy(False))

    # ---- kapanış ----
    def _on_close(self):
        self.preview_running = False
        try:
            self.manager.disconnect_all()
        except Exception:
            pass
        try:
            self.manager.finalize_sdk()
        except Exception:
            pass
        self.app.destroy()

    def run(self):
        self.app.mainloop()


def main():
    if not MVS_AVAILABLE:
        print("[HATA] MVS SDK bulunamadı. Hikrobot MVS'i kurun ve"
              " MVCAM_COMMON_RUNENV ortam değişkeninin set olduğundan emin olun.")
        sys.exit(1)
    MultiCamApp().run()


if __name__ == "__main__":
    main()
