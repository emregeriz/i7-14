"""Aremak DataMatrix/QR barkod okuyucu - bagimsiz Python sarmalayici.

Kullanim:

    from aremak_barkod import AremakBarkodReader

    with AremakBarkodReader() as reader:
        for code in reader.scan("foto.jpg"):
            print(code.type, code.content, code.center_x, code.center_y)
"""

from .reader import AremakBarkodReader, BarkodKodu, AremakBarkodHatasi

__all__ = ["AremakBarkodReader", "BarkodKodu", "AremakBarkodHatasi"]
__version__ = "1.0.0"
