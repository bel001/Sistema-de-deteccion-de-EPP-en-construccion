#!/usr/bin/env bash
cd "$(dirname "$0")"

echo "=== Sistema de Detección de EPP (macOS) ==="

if [ ! -d ".venv" ]; then
    echo "🐍 Creando entorno virtual .venv..."
    python3 -m venv .venv
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/pip install ultralytics opencv-python numpy pandas pillow torch torchvision torchaudio
fi

if [ ! -f "weights/best.pt" ] && [ -f "weights/best.pt.zip" ]; then
    echo "📦 Preparando modelo weights/best.pt..."
    cp weights/best.pt.zip weights/best.pt
fi

echo "🚀 Iniciando Sistema de Detección de EPP..."
.venv/bin/python main.py
