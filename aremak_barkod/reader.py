"""Aremak DataMatrix/QR barkod okuyucu - .NET DLL'ini pythonnet ile cagirir.

Bu modul, Aremak Kod Okuyucu kurulumu (`C:\\AremakCodeReaderCN`) gerektirir.
USB lisans dongle (SenseShield/Virbox) takili olmalidir.

Asil ALGO motoruna `Aremak.CodeReader.Algo.dll` icindeki `AremakReader.Scan(path)`
metodu ile dogrudan baglanir; SDK'nin TSV server modu kullanilmaz.
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

DEFAULT_INSTALL_DIR = Path(r"C:\AremakCodeReaderCN")

PathLike = Union[str, os.PathLike]


class AremakBarkodHatasi(RuntimeError):
    """Aremak okuyucu yuklenemediginde veya tarama sirasinda hata alindiginda firlatilir."""


@dataclass
class BarkodKodu:
    """Tek bir cozulmus barkod sonucu."""

    index: int
    type: str
    content: str
    center_x: int
    center_y: int
    angle: float
    ppm: float
    corners: List[tuple] = field(default_factory=list)


_DOTNET_LOCK = threading.Lock()
_DOTNET_LOADED = False
_AremakReaderClass = None


def _load_dotnet(install_dir: Path) -> type:
    """Aremak .NET tipini yukler ve sinifi dondurur. Process basina sadece bir kez yapilir."""
    global _DOTNET_LOADED, _AremakReaderClass

    with _DOTNET_LOCK:
        if _DOTNET_LOADED and _AremakReaderClass is not None:
            return _AremakReaderClass

        bin_dir = install_dir / "bin"
        runtime_dir = install_dir / "runtime"
        algo_dll = bin_dir / "Aremak.CodeReader.Algo.dll"

        if not algo_dll.is_file():
            raise AremakBarkodHatasi(
                f"Aremak DLL bulunamadi: {algo_dll}\n"
                f"Aremak Kod Okuyucu kurulu mu? (Beklenen klasor: {install_dir})"
            )

        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(bin_dir))
            if runtime_dir.is_dir():
                os.add_dll_directory(str(runtime_dir))
        os.environ["PATH"] = (
            str(bin_dir) + os.pathsep + str(runtime_dir) + os.pathsep + os.environ.get("PATH", "")
        )

        try:
            import clr  # type: ignore
        except ImportError as ex:
            raise AremakBarkodHatasi(
                "pythonnet yuklu degil. Once `pip install pythonnet` calistirin."
            ) from ex

        sys.path.insert(0, str(bin_dir))
        clr.AddReference(str(algo_dll))

        from Aremak.CodeReader.Algo import AremakReader  # type: ignore

        _AremakReaderClass = AremakReader
        _DOTNET_LOADED = True
        return AremakReader


class AremakBarkodReader:
    """Aremak Kod Okuyucu icin yuksek seviye Python arayuzu.

    Parameters
    ----------
    install_dir
        Aremak Kod Okuyucu kurulum klasoru (varsayilan: ``C:\\AremakCodeReaderCN``).
    auto_convert_bmp
        JPG/PNG/TIFF gibi formatlari ALGO motoru destekleyemez; True ise tarama
        oncesi gecici bir BMP olusturulur (varsayilan: True).
    """

    def __init__(
        self,
        install_dir: PathLike = DEFAULT_INSTALL_DIR,
        *,
        auto_convert_bmp: bool = True,
    ) -> None:
        self.install_dir = Path(install_dir)
        self.auto_convert_bmp = auto_convert_bmp

        reader_class = _load_dotnet(self.install_dir)

        # AremakReader ve icindeki MVD bilesenleri yan DLL'leri current
        # working directory uzerinden cozer; baska bir cwd'den
        # calistirildiginda "Wrong with running environment" hatasi verir.
        # Instance olusumunu ve her tarama cagrisini bin klasoru cwd'sinde
        # yapiyoruz.
        self._bin_dir = self.install_dir / "bin"
        with self._bin_cwd():
            self._reader = reader_class()

        self._tmp_bmp: Optional[Path] = None

    def scan(self, image_path: PathLike) -> List[BarkodKodu]:
        """Verilen goruntu dosyasini tarar ve bulunan barkod listesini dondurur.

        JPG/PNG gibi destek disi formatlar varsayilan olarak otomatik BMP'ye
        cevrilir (``auto_convert_bmp=False`` ise hata firlatir).
        """
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Goruntu bulunamadi: {path}")

        scan_path = self._prepare_path(path)

        try:
            with self._bin_cwd():
                result = self._reader.Scan(str(scan_path))
        except Exception as ex:
            raise AremakBarkodHatasi(f"Tarama hatasi: {ex}") from ex

        codes: List[BarkodKodu] = []
        if hasattr(result, "Codes"):
            for i, c in enumerate(list(result.Codes), start=1):
                codes.append(self._convert(i, c))
        return codes

    @property
    def son_sdk_suresi_ms(self) -> float:
        """Son taramanin SDK suresi (ms). Tarama yapilmadan once 0.0."""
        return getattr(self, "_son_elapsed", 0.0)

    def scan_with_timing(self, image_path: PathLike) -> tuple[List[BarkodKodu], float]:
        """`scan()` ile ayni; ek olarak SDK suresini (ms) dondurur."""
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Goruntu bulunamadi: {path}")

        scan_path = self._prepare_path(path)
        try:
            with self._bin_cwd():
                result = self._reader.Scan(str(scan_path))
        except Exception as ex:
            raise AremakBarkodHatasi(f"Tarama hatasi: {ex}") from ex

        elapsed_ms = float(getattr(result, "SdkMs", 0.0))
        self._son_elapsed = elapsed_ms

        codes: List[BarkodKodu] = []
        if hasattr(result, "Codes"):
            for i, c in enumerate(list(result.Codes), start=1):
                codes.append(self._convert(i, c))
        return codes, elapsed_ms

    def close(self) -> None:
        """Okuyucuyu serbest birakir. Cagrildiktan sonra kullanim disi."""
        if self._reader is not None:
            try:
                self._reader.Dispose()
            except Exception:
                pass
            self._reader = None
        if self._tmp_bmp is not None:
            try:
                self._tmp_bmp.unlink(missing_ok=True)
            except Exception:
                pass
            self._tmp_bmp = None

    def __enter__(self) -> "AremakBarkodReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _bin_cwd(self):
        """`bin` klasorunu cwd yapan context manager. DLL kesfi icin gerekli."""

        class _BinCwd:
            def __init__(_self, target: Path):
                _self.target = target
                _self.previous: Optional[Path] = None

            def __enter__(_self):
                _self.previous = Path.cwd()
                if _self.target.is_dir():
                    try:
                        os.chdir(_self.target)
                    except Exception:
                        pass
                return _self

            def __exit__(_self, *exc):
                if _self.previous is not None:
                    try:
                        os.chdir(_self.previous)
                    except Exception:
                        pass

        return _BinCwd(self._bin_dir)

    def _prepare_path(self, path: Path) -> Path:
        if path.suffix.lower() == ".bmp":
            return path
        if not self.auto_convert_bmp:
            raise AremakBarkodHatasi(
                f"Aremak ALGO motoru sadece BMP destekler. Once .bmp'ye donusturun: {path}"
            )

        try:
            from PIL import Image  # type: ignore
        except ImportError as ex:
            raise AremakBarkodHatasi(
                "JPG/PNG icin Pillow gerekli. `pip install Pillow` calistirin."
            ) from ex

        if self._tmp_bmp is None:
            fd, tmp = tempfile.mkstemp(prefix="aremak_barkod_", suffix=".bmp")
            os.close(fd)
            self._tmp_bmp = Path(tmp)

        with Image.open(path) as img:
            img.convert("RGB").save(self._tmp_bmp, format="BMP")
        return self._tmp_bmp

    @staticmethod
    def _convert(index: int, c) -> BarkodKodu:
        type_name = str(getattr(c, "Type", "")).replace("MVD_CNN_CODETYPE_", "").replace("MVD_CODETYPE_", "")

        corners: List[tuple] = []
        try:
            xs = list(getattr(c, "CornerXs", []) or [])
            ys = list(getattr(c, "CornerYs", []) or [])
            for x, y in zip(xs, ys):
                corners.append((int(x), int(y)))
        except Exception:
            corners = []

        return BarkodKodu(
            index=index,
            type=type_name or "?",
            content=str(getattr(c, "Content", "") or ""),
            center_x=int(getattr(c, "CenterX", 0)),
            center_y=int(getattr(c, "CenterY", 0)),
            angle=float(getattr(c, "Angle", 0.0)),
            ppm=float(getattr(c, "Ppm", 0.0)),
            corners=corners,
        )
