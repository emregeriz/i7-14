@echo off
title TTO - Sayim Sure Testi
cd /d "%~dp0"
py -3.11 sure_testi.py
if errorlevel 1 (
    echo.
    echo Uygulama bir hatayla kapandi. Once kurulum.bat calistirildi mi?
    pause
)
