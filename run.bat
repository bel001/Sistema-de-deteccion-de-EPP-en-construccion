@echo off
title Sistema de Deteccion de EPP - Construccion
cd /d "%~dp0"

echo === Sistema de Deteccion de EPP ===

IF NOT EXIST ".venv" (
    echo [INFO] Creando entorno virtual Python...
    py -3 -m venv .venv
    call .venv\Scripts\activate.bat
    python -m pip install --upgrade pip
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    python -m pip install ultralytics opencv-python numpy pandas pillow
) ELSE (
    call .venv\Scripts\activate.bat
)

IF NOT EXIST "weights\best.pt" (
    IF EXIST "weights\best.pt.zip" (
        echo [INFO] Preparando modelo weights\best.pt...
        copy /y "weights\best.pt.zip" "weights\best.pt"
    )
)

echo [INFO] Iniciando aplicacion...
python main.py
pause
