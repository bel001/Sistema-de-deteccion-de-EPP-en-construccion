#!/usr/bin/env bash
set -e

echo "=== Configuración de Aceleración por GPU (Linux - NVIDIA) ==="
echo "Directorio de trabajo: $(pwd)"

# 1. Verificar GPU NVIDIA y Tkinter
if command -v nvidia-smi &> /dev/null; then
    echo "✔ GPU NVIDIA detectada:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    echo "⚠️ ADVERTENCIA: nvidia-smi no fue encontrado. Asegúrate de tener los drivers NVIDIA instalados."
fi

if ! python3 -c "import tkinter" &> /dev/null; then
    echo "⚠️ ATENCIÓN: No se detectó 'python3-tk'. Necesitas instalarlo con: sudo apt install -y python3-tk"
fi

# 2. Copiar pesos si es necesario
if [ ! -f "weights/best.pt" ] && [ -f "weights/best.pt.zip" ]; then
    echo "📦 Preparando weights/best.pt desde weights/best.pt.zip..."
    cp weights/best.pt.zip weights/best.pt
    echo "✔ Weights preparados en weights/best.pt"
fi

# 3. Crear Entorno Virtual
if [ ! -d ".venv" ]; then
    echo "🐍 Creando entorno virtual .venv..."
    python3 -m venv .venv
fi

# 4. Instalar PyTorch con CUDA 12.1 y dependencias usando directorio temporal local
echo "🚀 Instalando/actualizando PyTorch con soporte CUDA..."
mkdir -p .venv/tmp
export TMPDIR="$(pwd)/.venv/tmp"
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
.venv/bin/pip install --no-cache-dir ultralytics opencv-python numpy pandas pillow

# 5. Verificación de PyTorch CUDA
echo "🔍 Diagnosticando estado de PyTorch y CUDA..."
.venv/bin/python -c "
import torch
print('-'*50)
print('PyTorch Version:', torch.__version__)
print('CUDA Disponible:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Dispositivo GPU:', torch.cuda.get_device_name(0))
    print('VRAM Total:', round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2), 'GB')
else:
    print('⚠️ ALERTA: CUDA no está disponible para PyTorch.')
print('-'*50)
"

echo "✅ Configuración de GPU completada exitosamente."
