@echo off
title i7-14 - Kurulum
cd /d "%~dp0"
echo.
echo === TTO Sayim Sure Testi - kurulum ===
echo Python 3.11 gerekli (python.org, "py launcher" ile). Kontrol ediliyor...
py -3.11 --version || (echo [HATA] Python 3.11 bulunamadi. https://www.python.org/downloads/ adresinden 3.11.x kurun ^(Add to PATH + py launcher^). & pause & exit /b 1)

echo.
echo [1/3] Temel paketler (torch CPU derlemesi; YOLO icin yeterli)...
py -3.11 -m pip install --upgrade pip
py -3.11 -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
py -3.11 -m pip install -r requirements.txt

echo.
echo [2/3] PaddlePaddle GPU (CUDA 12.6 derlemesi; OCR ekran kartinda calisir)...
py -3.11 -m pip install paddlepaddle-gpu==3.2.2 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

echo.
echo [3/3] Kontrol...
py -3.11 -c "import paddle, torch, ultralytics, customtkinter, cv2; print('paddle CUDA:', paddle.is_compiled_with_cuda(), '| GPU sayisi:', paddle.device.cuda.device_count())"
py -3.11 -c "import halcon" 2>nul && (echo HALCON: hazir) || (echo HALCON: yok - istege bagli, README'ye bakin)
py -3.11 -c "import clr" 2>nul && (echo pythonnet: hazir) || (echo pythonnet: yok - Aremak icin gerekli)
echo.
echo Kurulum bitti. baslat.bat ile acin. Ilk acilista PaddleOCR modelleri (~200 MB) internetten indirilir.
pause
