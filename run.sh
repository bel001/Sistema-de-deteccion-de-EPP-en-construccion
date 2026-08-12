#!/usr/bin/env bash
cd "$(dirname "$0")"

# 1. Comprobar soporte de Tkinter en Linux
if ! python3 -c "import tkinter" &> /dev/null; then
    echo "❌ ERROR: Falta el paquete 'python3-tk' necesario para la interfaz gráfica en Linux."
    echo ""
    echo "Para solucionarlo, ejecuta una vez este comando en tu terminal e introduce tu contraseña:"
    echo "👉  sudo apt update && sudo apt install -y python3-tk"
    echo ""
    exit 1
fi

# 2. Ejecutar setup automático si no está creado el entorno o faltan los pesos
if [ ! -d ".venv" ] || [ ! -f "weights/best.pt" ]; then
    echo "⚙️ Configurando entorno por primera vez..."
    bash setup_gpu_linux.sh
fi

echo "🚀 Iniciando Sistema de Detección de EPP..."
.venv/bin/python main.py
