@echo off
title TTO HIZLANDIRILMIS - Trento Toplu Okuma
cd /d "%~dp0"

set "PYTHON_CMD="
if exist ".venv\Scripts\python.exe" set "PYTHON_CMD=.venv\Scripts\python.exe"
if not defined PYTHON_CMD where python >nul 2>nul && set "PYTHON_CMD=python"

if not defined PYTHON_CMD (
    echo [HATA] Python bulunamadi. Once ilk_kurulum.bat dosyasini calistirin.
    pause
    exit /b 1
)

%PYTHON_CMD% -c "import customtkinter, cv2, PIL" >nul 2>nul
if errorlevel 1 (
    echo [HATA] Gerekli paketler bulunamadi. Once ilk_kurulum.bat dosyasini calistirin.
    pause
    exit /b 1
)

REM Okunamayan kasalar icin OCR: paddleocr secili Python'da yoksa,
REM kurulu oldugu sistem Python 3.11'e gec (kutahya kurulumu ile ayni).
%PYTHON_CMD% -c "import paddleocr" >nul 2>nul
if errorlevel 1 (
    py -3.11 -c "import paddleocr, customtkinter, cv2, PIL" >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_CMD=py -3.11"
        echo [BILGI] OCR icin sistem Python 3.11 kullaniliyor.
    ) else (
        echo [UYARI] paddleocr bulunamadi - okunamayan kasalarda OCR devre disi.
    )
)

%PYTHON_CMD% app.py

if errorlevel 1 (
    echo.
    echo Uygulama bir hatayla kapandi.
    pause
)
