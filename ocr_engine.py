"""
PaddleOCR (PP-OCRv5 varsayılan) — CLI ve arayüz için ortak yardımcılar.
"""

from __future__ import annotations

import json
import os
import re
import sys
import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_IMAGE_URL = (
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png"
)

# Paddle metin tespiti varsayılan ``max_side_limit`` (aşılırsa uyarı + içte yeniden boyutlandırma).
DEFAULT_MAX_OCR_INPUT_SIDE = 4000


def prepare_image_for_ocr(
    image_path: str,
    max_side: int = DEFAULT_MAX_OCR_INPUT_SIDE,
) -> tuple[str, str | None]:
    """
    Uzun kenarı ``max_side`` üzerindeyse orantılı küçültüp geçici dosyaya yazar.
    Dönüş: ``(ocr_girdi_yolu, geçici_dosya_yolu_veya_None)`` — geçici varsa iş bitince silin.
    """
    path = os.path.abspath(image_path)
    if not os.path.isfile(path) or max_side <= 0:
        return path, None

    try:
        from PIL import Image

        with Image.open(path) as im:
            w, h = im.size
            long_side = max(w, h)
            if long_side <= max_side:
                return path, None
            scale = max_side / float(long_side)
            nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
            resized = im.convert("RGB").resize((nw, nh), Image.Resampling.LANCZOS)
            resized.load()

        suffix = Path(path).suffix.lower()
        if suffix not in (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"):
            suffix = ".png"
        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        save_kw: dict[str, Any] = {}
        if suffix in (".jpg", ".jpeg"):
            save_kw["quality"] = 95
            resized.save(tmp_path, format="JPEG", **save_kw)
        else:
            resized.save(tmp_path)
        print(f"OCR input resized: {w}x{h} -> {nw}x{nh} (max side {max_side})")
        return tmp_path, tmp_path
    except Exception as e:
        print(f"prepare_image_for_ocr: skipped ({e})", file=sys.stderr)
        return path, None


def paddleocr_major_version() -> int:
    """PaddleOCR 3.x = PP-OCRv5 resmi boru hattı (paddleocr.ai ile aynı aile)."""
    try:
        import paddleocr as _po

        return int(str(getattr(_po, "__version__", "0")).split(".", 1)[0])
    except Exception:
        return 0


def download_if_url(path_or_url: str, dest_dir: str) -> str:
    if path_or_url.startswith(("http://", "https://")):
        os.makedirs(dest_dir, exist_ok=True)
        name = path_or_url.rstrip("/").split("/")[-1] or "input.png"
        local = os.path.join(dest_dir, name)
        if not os.path.isfile(local):
            print(f"İndiriliyor: {path_or_url} -> {local}")
            urllib.request.urlretrieve(path_or_url, local)
        return local
    return path_or_url


def ocr_kwargs(
    device: str = "cpu",
    lang: str | None = None,
    ocr_version: str | None = None,
    mobile: bool = False,
    *,
    full_pipeline: bool = False,
) -> dict[str, Any]:
    """
    PaddleOCR 3.x: varsayılan olarak ``ocr_version=PP-OCRv5`` (paddleocr.ai «PP-OCRv5» ile aynı seri).
    ``lang='en'`` vermek çoğu kurulumda ``en_PP-OCRv5_mobile_rec`` seçer; webdeki çok sahneli
    ``PP-OCRv5_server_rec`` için ``lang`` boş bırakın.

    ``full_pipeline=True``: belge yönü / düzeltme / satır yönü modüllerini Paddle varsayılanına
    bırakır (daha yavaş, eğik ekran görüntüleri için web davranışına yakın).
    """
    kwargs: dict[str, Any] = {"device": device}

    if not full_pipeline:
        kwargs.update(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    if lang:
        cleaned = lang.strip()
        if cleaned:
            kwargs["lang"] = cleaned

    maj = paddleocr_major_version()
    if maj >= 3:
        if ocr_version:
            kwargs["ocr_version"] = ocr_version
        else:
            kwargs["ocr_version"] = "PP-OCRv5"
    elif ocr_version:
        # 2.x: yalnızca PP-OCRv4 anlamlı; PP-OCRv5 yerel pakette yok
        if ocr_version == "PP-OCRv5":
            pass
        else:
            kwargs["ocr_version"] = ocr_version

    if mobile:
        kwargs["text_detection_model_name"] = "PP-OCRv5_mobile_det"
        kwargs["text_recognition_model_name"] = "PP-OCRv5_mobile_rec"

    return kwargs


def build_ocr(**kwargs: Any) -> Any:
    import inspect

    from paddleocr import PaddleOCR

    sig = inspect.signature(PaddleOCR.__init__)
    params = list(sig.parameters.values())
    param_names = {p.name for p in params if p.name != "self"}

    # PaddleOCR 2.10+ çoğu sürüm: __init__(self, **kwargs) — imzada yalnızca self + kwargs olur;
    # eski filtre tüm argümanları atıp PaddleOCR() boş çağrılırdı (lang/device yok sayılırdı).
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
        filtered = dict(kwargs)
    else:
        filtered = {k: v for k, v in kwargs.items() if k in param_names}

    if (
        "device" in kwargs
        and "device" not in filtered
        and "use_gpu" in param_names
        and "use_gpu" not in filtered
    ):
        filtered["use_gpu"] = str(kwargs["device"]).lower().startswith("gpu")

    return PaddleOCR(**filtered)


def _scores_to_list(scores: Any) -> list[float]:
    if scores is None:
        return []
    try:
        import numpy as np

        if isinstance(scores, np.ndarray):
            return [float(x) for x in scores.tolist()]
    except ImportError:
        pass
    if isinstance(scores, (list, tuple)):
        return [float(x) for x in scores]
    return []


def _extract_inner_dict(res: Any) -> dict[str, Any] | None:
    if isinstance(res, dict):
        return res.get("res") if "res" in res else res
    inner = getattr(res, "res", None)
    if isinstance(inner, dict):
        return inner
    for name in ("json", "to_dict"):
        fn = getattr(res, name, None)
        if callable(fn):
            try:
                d = fn()
                if isinstance(d, dict):
                    return d.get("res", d)
            except Exception:
                pass
    return None


def _texts_scores_from_inner(inner: dict[str, Any]) -> tuple[list[str], list[float]] | None:
    texts = inner.get("rec_texts")
    if texts is None:
        return None
    texts = [str(t) for t in texts]
    scores = _scores_to_list(inner.get("rec_scores"))
    if len(scores) < len(texts):
        scores.extend([0.0] * (len(texts) - len(scores)))
    elif len(scores) > len(texts):
        scores = scores[: len(texts)]
    return texts, scores


def extract_texts_scores(pred_item: Any) -> tuple[list[str], list[float]]:
    inner = _extract_inner_dict(pred_item)
    if inner:
        got = _texts_scores_from_inner(inner)
        if got:
            return got

    td = tempfile.mkdtemp(prefix="ocr_json_")
    try:
        pred_item.save_to_json(td)
        for p in Path(td).rglob("*.json"):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            inner2 = data.get("res", data) if isinstance(data, dict) else None
            if isinstance(inner2, dict):
                got = _texts_scores_from_inner(inner2)
                if got:
                    return got
    except Exception:
        pass
    finally:
        shutil.rmtree(td, ignore_errors=True)

    return [], []


def _legacy_ocr_to_texts_scores(ocr_out: Any) -> tuple[list[str], list[float]]:
    """PaddleOCR 2.x `ocr.ocr()` çıktısı: [ [ [kutu, (metin, skor)], ... ] ] veya benzeri."""
    texts: list[str] = []
    scores: list[float] = []
    if not ocr_out:
        return texts, scores
    for page in ocr_out:
        if not page:
            continue
        for item in page:
            if not item or len(item) < 2:
                continue
            pair = item[1]
            if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                texts.append(str(pair[0]))
                try:
                    scores.append(float(pair[1]))
                except (TypeError, ValueError):
                    scores.append(0.0)
    return texts, scores


def _line_kind(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return "boş"
    if re.fullmatch(r"[0-9\s.,:+\-/%]+", stripped):
        return "rakam / sayı"
    if re.fullmatch(r"[A-Za-z0-9\s.,:+\-/%]+", stripped) and re.search(r"\d", stripped):
        if re.search(r"[A-Za-z\u0080-\uFFFF]", stripped):
            return "karışık (harf+rakam)"
        return "rakam / sayı"
    if re.search(r"\d", stripped):
        return "metin + rakam"
    return "kelime / metin"


def predict_with_details(
    ocr: Any,
    image_path: str,
    *,
    max_input_side: int = DEFAULT_MAX_OCR_INPUT_SIDE,
) -> dict[str, Any]:
    """Tahmin + görselleştirme yolu (varsa) + tablo satırları."""
    tmp_img = tempfile.mkdtemp(prefix="ocr_viz_")
    all_texts: list[str] = []
    all_scores: list[float] = []
    viz_path: str | None = None

    use_path, tmp_downscale = prepare_image_for_ocr(image_path, max_input_side)
    try:
        if hasattr(ocr, "predict"):
            result = ocr.predict(use_path)
            for res in result:
                t, s = extract_texts_scores(res)
                all_texts.extend(t)
                all_scores.extend(s)
                try:
                    res.save_to_img(tmp_img)
                except Exception:
                    pass
        else:
            ocr_out = ocr.ocr(use_path, cls=False)
            all_texts, all_scores = _legacy_ocr_to_texts_scores(ocr_out)
    finally:
        if tmp_downscale and os.path.isfile(tmp_downscale):
            try:
                os.unlink(tmp_downscale)
            except OSError:
                pass

    for root, _, files in os.walk(tmp_img):
        for f in sorted(files):
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                viz_path = os.path.join(root, f)
                break
        if viz_path:
            break

    rows: list[dict[str, Any]] = []
    for i, (txt, sc) in enumerate(zip(all_texts, all_scores), start=1):
        rows.append(
            {
                "#": i,
                "metin": txt,
                "güven": round(float(sc), 4),
                "tip": _line_kind(txt),
            }
        )

    full_text = "\n".join(all_texts)
    md_lines = "\n".join(
        f"| {r['#']} | {r['tip']} | `{r['metin']}` | {r['güven']} |" for r in rows
    )
    md_table = (
        "| # | Tip | Metin | Güven |\n|---|-----|-------|-------|\n"
        + (md_lines if md_lines else "| — | — | Sonuç yok | — |")
    )

    return {
        "texts": all_texts,
        "scores": all_scores,
        "rows": rows,
        "viz_path": viz_path,
        "full_text": full_text,
        "markdown_table": md_table,
        "_viz_dir": tmp_img,
    }


def export_cli(
    ocr: Any,
    image_path: str,
    output_dir: str,
    *,
    max_input_side: int = DEFAULT_MAX_OCR_INPUT_SIDE,
) -> None:
    """Komut satırı: konsola yazdır ve mümkünse diske kaydet."""
    os.makedirs(output_dir, exist_ok=True)
    use_path, tmp_downscale = prepare_image_for_ocr(image_path, max_input_side)
    try:
        if hasattr(ocr, "predict"):
            for res in ocr.predict(use_path):
                res.print()
                try:
                    res.save_to_img(output_dir)
                    res.save_to_json(output_dir)
                except Exception:
                    pass
            return

        out = ocr.ocr(use_path, cls=False)
    finally:
        if tmp_downscale and os.path.isfile(tmp_downscale):
            try:
                os.unlink(tmp_downscale)
            except OSError:
                pass

    print(out)
    try:
        p = Path(output_dir) / "ocr_legacy_result.txt"
        p.write_text(repr(out), encoding="utf-8")
        print(f"(Eski API) ham çıktı yazıldı: {p}")
    except OSError as e:
        print(f"(Eski API) dosyaya yazılamadı: {e}")
