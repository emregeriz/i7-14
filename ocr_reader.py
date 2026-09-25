# -*- coding: utf-8 -*-
"""TTO — okunamayan kasaların üstündeki basılı numarayı PaddleOCR ile okur.

Repo kökündeki `ocr_engine.py` / `app_ui.py` PP-OCRv5 sarmalayıcılarını
kullanır (kutahyaya_gidiyoruz.py ile aynı altyapı). PaddleX bir process'te
yalnız bir kez init edilebildiği için OCR örneği tek cache'ten paylaşılır.
"""

from __future__ import annotations

# PaddlePaddle 3.x içindeki bir hatayı baypas eden bayraklar.
# paddle/paddleocr importlarından ÖNCE set edilmek zorunda.
import os
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_pir_in_executor", "0")

import re
import sys
import threading
import traceback
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR  # paket: app_ui ve ocr_engine ayni klasorde
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

try:
    import app_ui as _paddle_ui          # yalnızca _get_ocr için (gradio lazy)
    import ocr_engine as _paddle_engine
    PADDLE_AVAILABLE = True
    _IMPORT_ERR: str | None = None
except Exception as _ex:
    _paddle_ui = None          # type: ignore[assignment]
    _paddle_engine = None      # type: ignore[assignment]
    PADDLE_AVAILABLE = False
    _IMPORT_ERR = f"{type(_ex).__name__}: {_ex}"

# PP-OCRv5 server modelinin iç `max_side_limit` değeri 4000; aynı davranış.
OCR_MAX_SIDE = 4000
# Aday sayılması için gereken en düşük OCR güveni.
MIN_SCORE = 0.5

# Kasanın KENDİ seri bilgisi kesitte yalnız iki biçimde yazar:
#   a) büyük kabartma iki hane          → "12"          (→ 6412)
#   b) tam seri kodu 64xx/43xx ile BAŞLAR → "6412A", "6412A00000000000001249348"
#
# Kasanın sağ kenarındaki kalıp/üretim damgası ("TR-77-K-005403") her kasada
# bulunur ve OCR onu 0.99 güvenle okur. Eskiden metindeki HER 2-4 haneli rakam
# grubu aday sayıldığı için bu damgadan "77" çıkıp "6477" gibi olmayan bir seri
# üretiliyordu; kasanın kendi numarası okunamadığında bu sahte aday kazanıyordu.
# Aşağıdaki iki kalıp damgayı hiç aday etmez.
_BARE_TWO_DIGIT = re.compile(r"^\D{0,2}(\d{2})\D{0,2}$")
_PREFIXED_SERIAL = re.compile(r"^(64|43)(\d{2})")

# Aday ağırlıkları: tam seri kodu çıplak iki haneden güçlü sayılır; böylece
# beraberlik OCR'ın satır sırasına göre değil kanıtın gücüne göre çözülür.
WEIGHT_PREFIXED = 3
WEIGHT_BARE = 2

_CACHE: dict = {"ocr": None, "err": None, "device": None}
# PaddleX aynı process'te iki kez init edilirse çöker; yüklemeyi kilitle.
_LOAD_LOCK = threading.Lock()


def active_device() -> str | None:
    """Yüklü OCR'ın gerçekte çalıştığı cihaz ('gpu'/'cpu'); yüklenmediyse None."""
    return _CACHE.get("device")


def availability_error() -> str | None:
    """OCR bu ortamda çalışamayacaksa nedenini döner; çalışabilirse None.

    Modeli yüklemeden, ucuz bir kontrolle cevap verir — arayüz açılışta
    operatörü uyarabilsin diye.
    """
    if not PADDLE_AVAILABLE:
        return _IMPORT_ERR or "ocr_engine/app_ui import edilemedi"
    if _CACHE["err"] is not None:
        return str(_CACHE["err"])
    import importlib.util
    if importlib.util.find_spec("paddleocr") is None:
        return (
            "paddleocr bu Python ortamında kurulu değil — uygulamayı "
            "baslat.bat ile ya da 'py -3.11 app.py' ile başlatın"
        )
    return None


def _device_candidates() -> list[str]:
    """Denenecek cihaz sırası. GPU derlemesi varsa önce gpu, olmazsa cpu.

    TTO_OCR_DEVICE ortam değişkeni (`gpu`/`cpu`) sırayı ezer. GPU derlemeli
    paddle'da `device_count` anlık 0 dönse bile gpu yine de denenir; başarısız
    olursa cpu'ya düşülür — böylece geçici algılama hataları OCR'ı CPU'ya
    mahkûm etmez.
    """
    forced = os.environ.get("TTO_OCR_DEVICE", "").strip().lower()
    if forced in ("gpu", "cpu"):
        return [forced]
    try:
        import paddle  # noqa: WPS433
        if paddle.is_compiled_with_cuda():
            return ["gpu", "cpu"]
    except Exception:
        pass
    return ["cpu"]


def _warmup(ocr, log_fn=None) -> None:
    """Küçük bir görüntüyle ilk tahmini yapar; kernel/pipeline ısınır.

    Böylece gerçek ilk kasanın OCR'ı saniyeler değil, salise mertebesinde olur.
    """
    import tempfile

    from PIL import Image, ImageDraw

    fd, path = tempfile.mkstemp(suffix=".png", prefix="ocr_warmup_")
    os.close(fd)
    try:
        image = Image.new("RGB", (320, 96), "white")
        ImageDraw.Draw(image).text((40, 30), "6412", fill="black")
        image.save(path)
        _paddle_engine.predict_with_details(ocr, path, max_input_side=OCR_MAX_SIDE)
        if log_fn:
            log_fn("[OCR] Isınma tahmini tamam.")
    except Exception as ex:
        if log_fn:
            log_fn(f"[OCR] Isınma atlandı: {type(ex).__name__}: {ex}")
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def get_ocr(log_fn=None):
    """PaddleOCR singleton'ını döner; ilk çağrıda modeli yükler.

    Hata durumunda None döner. Yarım kalan init sonrası paddle
    "PDX has already been initialized" verdiğinden başarısız denemeden
    sonra tekrar denemiyoruz — programın yeniden başlatılması gerekir.
    """
    with _LOAD_LOCK:
        return _get_ocr_locked(log_fn)


def _get_ocr_locked(log_fn=None):
    if _CACHE["ocr"] is not None:
        return _CACHE["ocr"]
    if _CACHE["err"] is not None:
        return None
    if not PADDLE_AVAILABLE or _paddle_ui is None:
        _CACHE["err"] = _IMPORT_ERR or "paddle bulunamadı"
        if log_fn:
            log_fn(f"[OCR] PaddleOCR import edilemedi: {_CACHE['err']}")
        return None
    # Torch, paddle'dan önce yüklenmeli — Windows'ta DLL sırası bozulunca
    # shm.dll (WinError 127) hatası çıkıyor.
    try:
        import torch  # noqa: F401
    except Exception:
        pass
    last_err = None
    for device in _device_candidates():
        try:
            if log_fn:
                log_fn(f"[OCR] PaddleOCR (PP-OCRv5) yükleniyor · cihaz={device}…")
            # _get_ocr(device, mobile, use_v4, lang): mobile=False → server modeli.
            ocr = _paddle_ui._get_ocr(device, False, False, "")
            _warmup(ocr, log_fn)
            _CACHE["ocr"] = ocr
            _CACHE["device"] = device
            if log_fn:
                log_fn(f"[OCR] ✔ PaddleOCR hazır (cihaz={device}).")
            return ocr
        except Exception as ex:
            last_err = f"{type(ex).__name__}: {ex}"
            if log_fn:
                log_fn(f"[OCR] {device} ile yükleme başarısız: {last_err}")
                for line in traceback.format_exc().splitlines():
                    log_fn("  " + line)
    _CACHE["err"] = last_err or "OCR yüklenemedi"
    return None


def serial_from_number(number, prefix: str = "64") -> str | None:
    """OCR'da okunan numarayı 4 haneli kasa serisine çevirir.

    "12"   → "6412"   (prefix + kasa numarası)
    "6412" → "6412"   (zaten tam seri)
    Diğer uzunluklar güvenilmez sayılır → None (operatöre sorulur).
    """
    s = str(number or "").strip()
    if not s.isdigit():
        return None
    if len(s) == 4 and s[:2] in ("64", "43"):
        return s
    if len(s) == 2:
        # "64" / "43" kasanın basılı numarası DEĞİL, serinin ön ekidir; kasa
        # köşesindeki "6412A" kodundan OCR yalnız "64" yakalayınca buraya
        # düşüp "6464" gibi OLMAYAN bir seri üretiyordu. Ön eki kasa numarası
        # sayma — kanıt buysa seri çözülemez (operatöre sorulur).
        if s in ("64", "43"):
            return None
        return prefix + s
    return None


def serial_candidates(texts, scores=None, prefix: str = "64"):
    """OCR satırlarından (seri, ağırlık, güven) adaylarını çıkarır.

    Yalnız kasanın kendi numarası kabul edilir; kalıp damgası, uyarı yazısı
    vb. metinlerin içindeki rakamlar aday sayılmaz. Ayrı fonksiyon olması
    kuralın OCR çalıştırmadan test edilebilmesi içindir.
    """
    candidates: list[tuple[str, int, float]] = []
    for index, text in enumerate(texts or []):
        try:
            score = float(scores[index]) if scores and index < len(scores) else 1.0
        except (TypeError, ValueError, IndexError):
            score = 1.0
        if score < MIN_SCORE:
            continue
        raw = str(text).strip()
        # a) Tam seri kodu: harf/boşluk atıldığında 64xx/43xx ile başlıyorsa.
        digits = re.sub(r"\D", "", raw)
        match = _PREFIXED_SERIAL.match(digits)
        if match and len(digits) >= 4:
            candidates.append((match.group(1) + match.group(2), WEIGHT_PREFIXED, score))
            continue
        # b) Satırın TAMAMI iki haneli sayıysa (kasanın büyük kabartma numarası).
        match = _BARE_TWO_DIGIT.match(raw)
        if match:
            serial = serial_from_number(match.group(1), prefix)
            if serial:
                candidates.append((serial, WEIGHT_BARE, score))
    return candidates


def read_crate_serial(crop_path, prefix: str = "64", log_fn=None, tag: str = "") -> str | None:
    """Kasa kesit görüntüsünden seriyi okumaya çalışır ("6412" gibi).

    Kesitteki tüm metin satırları PaddleOCR ile okunur, rakam grupları
    seriye çevrilir ve en sık görülen seri döner. Okunamazsa None —
    karar operatöre kalır.
    """
    ocr = get_ocr(log_fn)
    if ocr is None or not crop_path or not os.path.isfile(str(crop_path)):
        return None
    try:
        out = _paddle_engine.predict_with_details(
            ocr, str(crop_path), max_input_side=OCR_MAX_SIDE
        )
    except Exception as ex:
        if log_fn:
            log_fn(f"[OCR] {tag}: tarama hatası {type(ex).__name__}: {ex}")
        return None
    texts = list(out.get("texts") or [])
    scores = list(out.get("scores") or [])
    candidates = serial_candidates(texts, scores, prefix)
    if log_fn:
        found = sorted({serial for serial, _, _ in candidates})
        log_fn(
            f"[OCR] {tag}: {len(texts)} satır okundu, "
            f"{len(candidates)} seri adayı" + (f" → {found}" if found else "")
        )
    if not candidates:
        return None
    # Ağırlık toplamı kazanır; eşitlikte toplam OCR güveni ayırır. (Eskiden
    # beraberliği satır sırası çözüyordu ve damgadan gelen sahte aday
    # kazanabiliyordu.)
    tally: dict[str, list[float]] = {}
    for serial, weight, score in candidates:
        entry = tally.setdefault(serial, [0.0, 0.0])
        entry[0] += weight
        entry[1] += score
    return max(tally.items(), key=lambda item: (item[1][0], item[1][1]))[0]
