# -*- coding: utf-8 -*-
"""PaddleOCR yükleyici — app_ui.py'deki _get_ocr'ın kopyası (gradio'suz)."""
from typing import Any

import ocr_engine

_ocr_cache: dict[str, Any] = {"key": None, "ocr": None}


def _get_ocr(device: str, mobile: bool, use_v4: bool, lang: str) -> Any:
    if ocr_engine.paddleocr_major_version() >= 3:
        ver: str | None = "PP-OCRv4" if use_v4 else "PP-OCRv5"
    else:
        ver = "PP-OCRv4" if use_v4 else None
    lang_clean = (lang or "").strip() or None
    key = (device, mobile, ver, lang_clean)
    if _ocr_cache["key"] != key:
        kw = ocr_engine.ocr_kwargs(
            device=device,
            mobile=mobile,
            ocr_version=ver,
            lang=lang_clean,
        )
        _ocr_cache["ocr"] = ocr_engine.build_ocr(**kw)
        _ocr_cache["key"] = key
    return _ocr_cache["ocr"]

