# -*- coding: utf-8 -*-
"""TTO Hızlandırılmış — barkod okuma yolları ve sonuç kurma.

TTO'daki AremakDetectionProcessor.process_image tek fonksiyonda YOLO + kasa
kasa Aremak + kesit kaydı + işaretli görüntü yapar. Burada aynı iş üç parçaya
ayrılır ki barkod adımı seçilen yolla (kamera kamera / ortak havuz / tam kare)
ve paralel çalıştırılabilsin:

  tespit(frame)                 -> YOLO kutuları (TTO ile aynı eşik)
  barkod_oku(kalemler, ayar)    -> her kutuya "barkodlar" yazar
  sonuc_kur(kalem, ...)         -> process_image ile AYNI biçimde sonuç sözlüğü
                                   (kesit dosyaları, işaretli görüntü, ön işleme
                                   görüntüsü, OKUNAMADI_ adları)

Tüm kutular kapalıyken app.py bu modülü hiç kullanmaz; bugünkü TTO yolu
(process_image) çalışır. Karşılaştırma bu yüzden adildir.
"""
from __future__ import annotations

import json
import os
import queue
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from camera_gui_v2 import CONFIDENCE_THRESHOLD, CROP_EXT, preprocess_crop_for_barcode

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

import sys

_AREMAK_PKG = Path(__file__).resolve().parent.parent / "barkod_aremak"
if str(_AREMAK_PKG) not in sys.path:
    sys.path.insert(0, str(_AREMAK_PKG))
try:
    from aremak_barkod import AremakBarkodReader
except Exception:  # pragma: no cover
    AremakBarkodReader = None

AREMAK_BIN = Path(r"C:\AremakCodeReaderCN\bin")
MAX_READERS = 6

AYAR_ADLARI = ("paralel", "havuz", "tamkare", "hizli_kayit", "akis")
AYAR_ETIKET = {
    "paralel": "6 kamera aynı anda",
    "havuz": "Kesit havuzu",
    "tamkare": "Tam kare",
    "hizli_kayit": "Hızlı kayıt",
    "akis": "Çekerken oku (akış)",
}


class HizliAyarlar:
    """Aç/kapa kutuları; hizli_ayarlar.json'a yazılır, açılışta okunur."""

    def __init__(self, path: Path):
        self.path = path
        self.values = {name: False for name in AYAR_ADLARI}
        try:
            data = json.loads(path.read_text("utf-8"))
            for name in AYAR_ADLARI:
                self.values[name] = bool(data.get(name, False))
        except Exception:
            pass

    def __getitem__(self, name):
        return self.values[name]

    def set(self, name, value):
        self.values[name] = bool(value)
        # havuz ile tam kare birbirini dışlar (barkod yöntemi tek olmalı)
        if value and name == "havuz":
            self.values["tamkare"] = False
        if value and name == "tamkare":
            self.values["havuz"] = False
        # havuz / tam kare paralel okuyucu ister
        if value and name in ("havuz", "tamkare"):
            self.values["paralel"] = True
        try:
            self.path.write_text(json.dumps(self.values, ensure_ascii=False, indent=1), "utf-8")
        except Exception:
            pass

    @property
    def any_on(self):
        return any(self.values.values())

    def ozet(self):
        acik = [AYAR_ETIKET[n] for n in AYAR_ADLARI if self.values[n]]
        return " + ".join(acik) if acik else "TTO gibi (hepsi kapalı)"


class HizliBarkod:
    """Aremak okuyucu havuzu (kamera başına bir) ve okuma yolları."""

    def __init__(self, detector, log_fn=None):
        self.detector = detector          # AremakDetectionProcessor (YOLO + 1 okuyucu)
        self.log = log_fn or (lambda m: None)
        self.readers: list = []
        self._tmp = [os.path.join(tempfile.gettempdir(), f"tto_hizli_kasa_{i}.bmp")
                     for i in range(MAX_READERS)]
        self._lock = threading.Lock()
        self._warm_full = False

    # ------------------------------------------------------------ hazırlık
    def hazirla(self, n=MAX_READERS, tam_kare_isit=False, ornek_frame=None):
        """n okuyucu kurar ve ısıtır (kesit; istenirse tam kare)."""
        if AremakBarkodReader is None:
            raise RuntimeError("aremak_barkod paketi yok")
        with self._lock:
            if not self.readers and getattr(self.detector, "_reader", None) is not None:
                self.readers.append(self.detector._reader)   # slot 0 = TTO'nun okuyucusu
            while len(self.readers) < n:
                self.readers.append(AremakBarkodReader())
            # ısınma: motor gördüğü ilk görüntü BOYUTU için hazırlık yapıyor
            if ornek_frame is not None:
                h, w = ornek_frame.shape[:2]
                crop = preprocess_crop_for_barcode(ornek_frame[h // 3: h // 3 + 200, : w // 2])
                gray = cv2.cvtColor(ornek_frame, cv2.COLOR_BGR2GRAY)
                started = time.perf_counter()

                def warm(slot):
                    cv2.imwrite(self._tmp[slot], crop)
                    self.readers[slot].scan(self._tmp[slot])
                    if tam_kare_isit:
                        cv2.imwrite(self._tmp[slot], gray)
                        self.readers[slot].scan(self._tmp[slot])

                with self._cwd_bin():
                    with ThreadPoolExecutor(max_workers=n) as pool:
                        list(pool.map(warm, range(n)))
                self._warm_full = self._warm_full or tam_kare_isit
                self.log(f"Aremak {n} okuyucu ısıtıldı"
                         f"{' (tam kare dahil)' if tam_kare_isit else ''} "
                         f"· {time.perf_counter() - started:.1f} sn")

    class _cwd_bin:
        """Aremak sarmalayıcısı her taramada cwd'yi bin'e alıp eskisine döndürür;
        paralel taramalarda biri diğerinin cwd'sini bozar. Aşama boyunca bin'de
        sabitlenir."""

        def __enter__(self):
            self.prev = os.getcwd()
            if AREMAK_BIN.is_dir():
                os.chdir(AREMAK_BIN)

        def __exit__(self, *exc):
            os.chdir(self.prev)

    # --------------------------------------------------------------- YOLO
    def tespit(self, frame):
        """YOLO kutuları [(x1,y1,x2,y2)], TTO ile aynı model/eşik/cihaz."""
        self.detector.load_model(None)
        device = 0 if (torch is not None and torch.cuda.is_available()) else "cpu"
        results = self.detector.model.predict(
            source=frame, save=False, conf=CONFIDENCE_THRESHOLD, device=device, verbose=False
        )
        boxes = []
        if results:
            r = results[0]
            b = getattr(r, "boxes", None)
            if b is not None and hasattr(b, "xyxy"):
                h, w = frame.shape[:2]
                for x1, y1, x2, y2 in b.xyxy.cpu().numpy():
                    x1, y1 = max(0, int(x1)), max(0, int(y1))
                    x2, y2 = min(w, int(x2)), min(h, int(y2))
                    if x2 > x1 and y2 > y1:
                        boxes.append((x1, y1, x2, y2))
        return boxes

    # -------------------------------------------------------------- tarama
    def _scan_crop(self, crop_bgr, slot):
        """TTO ile aynı: ön işleme → BMP → Aremak. (kodlar, ön işlenmiş gri)"""
        proc = preprocess_crop_for_barcode(crop_bgr)
        src = proc if proc is not None else cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        codes = []
        try:
            if cv2.imwrite(self._tmp[slot], src):
                codes = [str(c.content) for c in self.readers[slot].scan(self._tmp[slot])]
        except Exception:
            codes = []
        return codes, proc

    def _scan_frame(self, frame_bgr, slot):
        """Tam kare gri → [(kod, cx, cy)]."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        try:
            if not cv2.imwrite(self._tmp[slot], gray):
                return []
            return [(str(c.content), float(c.center_x), float(c.center_y))
                    for c in self.readers[slot].scan(self._tmp[slot])]
        except Exception:
            return []

    @staticmethod
    def _assign(codes, crates):
        """Kod merkezini içeren kutuya yazar (birden çoksa merkeze en yakın).
        Kutuya düşmeyen kod sayısını döner."""
        orphan = 0
        for code, cx, cy in codes:
            best, best_d = None, None
            for k in crates:
                x1, y1, x2, y2 = k["bbox"]
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    d = abs(cx - (x1 + x2) / 2) + abs(cy - (y1 + y2) / 2)
                    if best is None or d < best_d:
                        best, best_d = k, d
            if best is None:
                orphan += 1
            elif code not in best["barkodlar"]:
                best["barkodlar"].append(code)
        return orphan

    # --------------------------------------------------------- barkod oku
    def barkod_oku(self, kalemler, ayar):
        """kalemler: [{index, frame, crates:[{kasa_no,bbox,barkodlar,...}]}].
        Her kasaya barkodlar + proc (ön işlenmiş kesit, görselleştirme için)."""
        workers = MAX_READERS if ayar["paralel"] else 1
        n = min(workers, len(self.readers))
        if n == 0:
            raise RuntimeError("Aremak okuyucu yok")
        for item in kalemler:
            item["barkod_sn"] = 0.0
            for k in item["crates"]:
                k["barkodlar"] = []
                k["proc"] = None
        with self._cwd_bin():
            if ayar["tamkare"]:
                self._oku_tam_kare(kalemler, n)
            elif ayar["havuz"]:
                self._oku_havuz(kalemler, n)
            else:
                self._oku_kamera_kamera(kalemler, n)
        for item in kalemler:
            for k in item["crates"]:
                k["barkod_okundu"] = bool(k["barkodlar"])

    def _oku_kamera_kamera(self, kalemler, n):
        def cam(item, slot):
            t = time.perf_counter()
            f = item["frame"]
            for k in item["crates"]:
                x1, y1, x2, y2 = k["bbox"]
                k["barkodlar"], k["proc"] = self._scan_crop(f[y1:y2, x1:x2], slot)
            item["barkod_sn"] = time.perf_counter() - t

        if n == 1:
            for item in kalemler:
                cam(item, 0)
            return
        with ThreadPoolExecutor(max_workers=n) as pool:
            list(pool.map(cam, kalemler, [i % n for i in range(len(kalemler))]))

    def _oku_havuz(self, kalemler, n):
        jobs = [(item, k) for item in kalemler for k in item["crates"]]
        jobs.sort(key=lambda j: -(j[1]["bbox"][2] - j[1]["bbox"][0]) * (j[1]["bbox"][3] - j[1]["bbox"][1]))
        lock = threading.Lock()
        t0 = time.perf_counter()

        def worker(slot):
            while True:
                with lock:
                    if not jobs:
                        return
                    item, k = jobs.pop(0)
                x1, y1, x2, y2 = k["bbox"]
                codes, proc = self._scan_crop(item["frame"][y1:y2, x1:x2], slot)
                done = time.perf_counter() - t0
                with lock:
                    k["barkodlar"], k["proc"] = codes, proc
                    item["barkod_sn"] = max(item["barkod_sn"], done)

        with ThreadPoolExecutor(max_workers=n) as pool:
            list(pool.map(worker, range(n)))

    def _oku_tam_kare(self, kalemler, n):
        """Önce tam kareler; tam karesi biten okuyucu eşleşmeyen kutuları
        havuza atar, boşa çıkan her okuyucu havuzdan kesit çeker."""
        jobs = [("frame", item, None) for item in kalemler]
        lock = threading.Lock()
        t0 = time.perf_counter()
        for item in kalemler:
            item.update(ff_found=0, ff_matched=0, ff_orphan=0, ff_extra=0)

        def worker(slot):
            while True:
                with lock:
                    if not jobs:
                        return
                    kind, item, k = jobs.pop(0)
                if kind == "frame":
                    found = self._scan_frame(item["frame"], slot)
                    with lock:
                        item["ff_orphan"] = self._assign(found, item["crates"])
                        item["ff_found"] = len(found)
                        item["ff_matched"] = sum(1 for c in item["crates"] if c["barkodlar"])
                        jobs.extend(("crop", item, c) for c in item["crates"] if not c["barkodlar"])
                        item["barkod_sn"] = max(item["barkod_sn"], time.perf_counter() - t0)
                else:
                    x1, y1, x2, y2 = k["bbox"]
                    codes, proc = self._scan_crop(item["frame"][y1:y2, x1:x2], slot)
                    done = time.perf_counter() - t0
                    with lock:
                        k["barkodlar"], k["proc"] = codes, proc
                        item["ff_extra"] += bool(codes)
                        item["barkod_sn"] = max(item["barkod_sn"], done)

        with ThreadPoolExecutor(max_workers=n) as pool:
            list(pool.map(worker, range(n)))
        for item in kalemler:
            self.log(f"K{item['index'] + 1}: tam kare {item['ff_found']} kod → "
                     f"{item['ff_matched']} kasa ({item['ff_orphan']} kutu dışı), "
                     f"{len(item['crates']) - item['ff_matched']} kesit tekrar okundu, "
                     f"+{item['ff_extra']}")

    # ------------------------------------------------------------ sonuç
    def sonuc_kur(self, item, save_dir, hizli_kayit, io_pool=None):
        """process_image ile aynı dosyalar ve aynı sonuç sözlüğü."""
        frame = item["frame"]
        crates = item["crates"]
        save_dir = str(save_dir)
        crops_dir = os.path.join(save_dir, "crops")
        os.makedirs(crops_dir, exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        h, w = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        barcode_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        annotated = frame.copy()
        writes = []
        unread_names, unread_paths, contents = [], [], []
        total_codes = 0
        for k in crates:
            x1, y1, x2, y2 = k["bbox"]
            proc = k.pop("proc", None)
            if proc is not None:  # ön işleme görüntüsü (Okunamayan sayfası)
                fit = cv2.resize(proc, (x2 - x1, y2 - y1), interpolation=cv2.INTER_AREA)
                barcode_vis[y1:y2, x1:x2] = cv2.cvtColor(fit, cv2.COLOR_GRAY2BGR)
            codes = k["barkodlar"]
            if codes:
                total_codes += len(codes)
                contents.extend(codes)
                name = f"{timestamp_str}_kasa_{k['kasa_no']}{CROP_EXT}"
            else:
                name = f"OKUNAMADI_{timestamp_str}_kasa_{k['kasa_no']}{CROP_EXT}"
                unread_names.append(name)
                unread_paths.append(os.path.join(crops_dir, name))
            writes.append((os.path.join(crops_dir, name), frame[y1:y2, x1:x2]))
            color = (255, 255, 0) if codes else (0, 0, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            numara = str(k["kasa_no"])
            font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            (tw, tth), _ = cv2.getTextSize(numara, font, fs, th)
            tx, ty = x2 - tw - 4, y1 + tth + 4
            cv2.rectangle(annotated, (tx - 2, ty - tth - 2), (tx + tw + 2, ty + 2), (255, 255, 255), -1)
            cv2.putText(annotated, numara, (tx, ty), font, fs, (0, 0, 0), th, cv2.LINE_AA)

        annotated_path = os.path.join(save_dir, f"annotated_{timestamp_str}.jpg")
        prep_path = os.path.join(save_dir, f"barcode_preprocess_{timestamp_str}.jpg")
        writes.append((annotated_path, annotated))
        writes.append((prep_path, barcode_vis))
        if hizli_kayit and io_pool is not None:
            list(io_pool.map(lambda job: cv2.imwrite(*job), writes))
        else:
            for job in writes:
                cv2.imwrite(*job)

        return {
            "toplam_kasa": len(crates),
            "okunan_barkod": total_codes,
            "okunamayan_barkod": len(unread_names),
            "annotated_path": annotated_path,
            "annotated_image": annotated,
            "barcode_preprocess_image": barcode_vis,
            "barcode_preprocess_path": prep_path,
            "barkod_bulunamayan_isimler": unread_names,
            "barkod_bulunamayan_yollar": unread_paths,
            "kasalar": [{"kasa_no": k["kasa_no"], "bbox": k["bbox"],
                         "barkodlar": list(k["barkodlar"]), "barkod_okundu": k["barkod_okundu"]}
                        for k in crates],
            "image_width": int(w),
            "image_height": int(h),
            "barkod_icerikler": contents,
        }


class BarkodAkisi:
    """Çekim sürerken barkod okuma (akış / pipeline).

    Okuyucu iş parçacıkları sayım başında açılır ve bir kuyruğu dinler:
      kare_geldi(item)    fotoğraf gelir gelmez (tam kare modunda tam kare
                          okuması YOLO'yu beklemeden başlar)
      kutular_hazir(item) YOLO bitince (kesit / kamera işleri kuyruğa girer)
      bitir()             tüm kameralar eklendikten sonra; kuyruk boşalınca döner
    Bir kameranın tüm okumaları bitince on_done(item) çağrılır (kayıt vb.).
    Okuma yöntemi ayarlardan: tam kare / kesit havuzu / kamera kamera.
    """

    def __init__(self, hb, ayar, on_done, t0):
        self.hb, self.on_done, self.t0 = hb, on_done, t0
        self.mode = "tamkare" if ayar["tamkare"] else ("havuz" if ayar["havuz"] else "kamera")
        n = min(MAX_READERS if ayar["paralel"] else 1, len(hb.readers))
        if n == 0:
            raise RuntimeError("Aremak okuyucu yok")
        self.q = queue.Queue()
        self.cv = threading.Condition()
        self.pending = 0
        self._cwd = hb._cwd_bin()
        self._cwd.__enter__()   # okuma boyunca cwd = Aremak bin (bkz. _cwd_bin)
        self.threads = [threading.Thread(target=self._worker, args=(slot,), daemon=True)
                        for slot in range(n)]
        for t in self.threads:
            t.start()

    # --------------------------------------------------------- dış arayüz
    def kare_geldi(self, item):
        item.update(_frame_done=False, _boxes_ready=False, _b_first=None, _b_last=None,
                    ff_found=0, ff_matched=0, ff_orphan=0, ff_extra=0, _found=[])
        if self.mode == "tamkare":
            self._put(("frame", item, None))

    def kutular_hazir(self, item):
        for k in item["crates"]:
            k["barkodlar"], k["proc"] = [], None
        if self.mode == "tamkare":
            with self.cv:
                item["_boxes_ready"] = True
                ready = item["_frame_done"]
            if ready:
                self._after_frame(item)
        elif self.mode == "havuz":
            item["_remaining"] = len(item["crates"])
            if not item["crates"]:
                self._item_done(item)
            for k in sorted(item["crates"], key=lambda c: -(c["bbox"][2] - c["bbox"][0]) * (c["bbox"][3] - c["bbox"][1])):
                self._put(("crop", item, k))
        else:
            self._put(("camera", item, None))

    def bitir(self):
        with self.cv:
            while self.pending > 0:
                self.cv.wait()
        for _ in self.threads:
            self.q.put(None)
        for t in self.threads:
            t.join()
        self._cwd.__exit__(None, None, None)

    # ------------------------------------------------------------- iç işler
    def _put(self, job):
        with self.cv:
            self.pending += 1
        self.q.put(job)

    def _mark(self, item, start, end):
        with self.cv:
            if item["_b_first"] is None or start < item["_b_first"]:
                item["_b_first"] = start
            if item["_b_last"] is None or end > item["_b_last"]:
                item["_b_last"] = end

    def _after_frame(self, item):
        self.hb._assign(item["_found"], item["crates"])
        item["ff_found"] = len(item["_found"])
        item["ff_matched"] = sum(1 for c in item["crates"] if c["barkodlar"])
        unread = [c for c in item["crates"] if not c["barkodlar"]]
        item["_remaining"] = len(unread)
        if not unread:
            self._item_done(item)
        for c in unread:
            self._put(("crop", item, c))

    def _item_done(self, item):
        for k in item["crates"]:
            k["barkod_okundu"] = bool(k["barkodlar"])
        first = item["_b_first"] if item["_b_first"] is not None else time.perf_counter()
        last = item["_b_last"] if item["_b_last"] is not None else first
        item["barkod_sn"] = last - first
        item["barkod_bitti"] = last - self.t0
        if self.mode == "tamkare":
            self.hb.log(f"K{item['index'] + 1}: tam kare {item['ff_found']} kod → "
                        f"{item['ff_matched']} kasa, {len(item['crates']) - item['ff_matched']} "
                        f"kesit tekrar okundu, +{item['ff_extra']}")
        try:
            self.on_done(item)
        except Exception as exc:
            self.hb.log(f"[HIZLI] Akış sonuç HATA: {exc}")

    def _worker(self, slot):
        hb = self.hb
        while True:
            job = self.q.get()
            if job is None:
                return
            kind, item, k = job
            try:
                start = time.perf_counter()
                if kind == "frame":
                    found = hb._scan_frame(item["frame"], slot)
                    self._mark(item, start, time.perf_counter())
                    with self.cv:
                        item["_found"] = found
                        item["_frame_done"] = True
                        ready = item["_boxes_ready"]
                    if ready:
                        self._after_frame(item)
                elif kind == "crop":
                    x1, y1, x2, y2 = k["bbox"]
                    codes, proc = hb._scan_crop(item["frame"][y1:y2, x1:x2], slot)
                    self._mark(item, start, time.perf_counter())
                    with self.cv:
                        k["barkodlar"], k["proc"] = codes, proc
                        item["ff_extra"] += bool(codes)
                        item["_remaining"] -= 1
                        done = item["_remaining"] == 0
                    if done:
                        self._item_done(item)
                else:  # camera: bu kameranın tüm kesitleri bu okuyucuda
                    for c in item["crates"]:
                        x1, y1, x2, y2 = c["bbox"]
                        c["barkodlar"], c["proc"] = hb._scan_crop(item["frame"][y1:y2, x1:x2], slot)
                    self._mark(item, start, time.perf_counter())
                    self._item_done(item)
            except Exception as exc:
                hb.log(f"[HIZLI] Akış okuma HATA: {exc}")
            finally:
                with self.cv:
                    self.pending -= 1
                    self.cv.notify_all()
