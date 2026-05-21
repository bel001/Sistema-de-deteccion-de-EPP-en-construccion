# 🏗️ Sistema de Detección de EPP en Construcción

Sistema de visión artificial para la detección y verificación del uso de **Equipos de Protección Personal (EPP)** en entornos de construcción, utilizando un modelo **YOLOv8** entrenado con el dataset SH17.

El sistema procesa imágenes, videos y transmisiones de cámara web en tiempo real, detectando personas, cascos, manos, guantes, calzado y gafas, y generando alertas visuales cuando se detecta incumplimiento de seguridad.

---

## 📋 Tabla de Contenidos

- [Características](#-características)
- [Clases Detectadas](#-clases-detectadas)
- [Alertas de Seguridad](#-alertas-de-seguridad)
- [Requisitos del Sistema](#-requisitos-del-sistema)
- [Instalación Paso a Paso](#-instalación-paso-a-paso)
- [Uso del Sistema](#-uso-del-sistema)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Explicación Técnica](#-explicación-técnica)

---

## ✨ Características

- 🎥 **Procesamiento de video** en tiempo real con panel de telemetría
- 📷 **Cámara web en vivo** con detección instantánea
- 🖼️ **Análisis de imagen individual** con exportación anotada
- 📁 **Procesamiento por lotes** de carpetas completas con reporte CSV
- 🧠 **Asociación inteligente cabeza-casco** para evitar falsos positivos
- ⚡ **Filtros geométricos avanzados** que eliminan detecciones ruidosas
- 🔴 **Alertas visuales en pantalla** cuando se detecta incumplimiento

---

## 🎯 Clases Detectadas

| Clase | Color en Pantalla | Descripción |
|---|---|---|
| Persona | 🔵 Azul | Detecta personas completas |
| Casco | 🟢 Verde | Cascos de seguridad |
| Sin casco | 🔴 Rojo | Cabezas expuestas sin casco |
| Manos | 🟣 Magenta | Manos visibles |
| Guantes | 🟠 Naranja | Guantes de protección |
| Calzado | 🔵 Celeste | Zapatos/botas |
| Gafas | 🟡 Cyan | Gafas de seguridad |

---

## 🚨 Alertas de Seguridad

El sistema genera las siguientes alertas automáticas:

| Alerta | Condición | Estado |
|---|---|---|
| `X persona(s) sin casco` | Se detecta una cabeza sin casco superpuesto | ✅ Activa |
| `Posible falta de guantes` | Se detectan manos sin guantes | ✅ Activa |
| `Posible falta de chaleco` | — | ⚠️ Desactivada (ver nota) |

> **Nota sobre el chaleco**: El modelo actual (`best.pt`) no detecta chalecos de seguridad debido a un error de indexación durante el entrenamiento original en Google Colab. Para habilitarlo, se requiere reentrenar el modelo incluyendo la clase `Safety-vest`.

---

## 💻 Requisitos del Sistema

- **Python**: 3.10 o superior
- **Sistema Operativo**: Linux (probado en Ubuntu), Windows o macOS
- **Cámara web** (opcional, para detección en vivo)
- **Espacio en disco**: ~500 MB (modelo + dependencias)

---

## 🚀 Instalación Paso a Paso

### 1. Clonar el repositorio

```bash
git clone https://github.com/bel001/Sistema-de-deteccion-de-EPP-en-construccion.git
cd Sistema-de-deteccion-de-EPP-en-construccion
```

### 2. Crear el entorno virtual de Python

```bash
python3 -m venv .venv
```

### 3. Activar el entorno virtual

**Linux / macOS:**
```bash
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
.venv\Scripts\activate.bat
```

### 4. Instalar las dependencias

```bash
pip install -r requirements.txt
```

### 5. Colocar el modelo entrenado

Copia el archivo de pesos del modelo (`best.pt`) dentro de la carpeta `weights/`:

```
weights/
└── best.pt
```

> **Importante**: El archivo `best.pt` no se incluye en el repositorio por su tamaño (~50 MB). Debe obtenerse del entrenamiento en Google Colab o descargarse por separado.

### 6. Colocar datos de prueba (opcional)

- Coloca imágenes de prueba en `inputs/images/`
- Coloca videos de prueba en `inputs/videos/`

---

## 🎮 Uso del Sistema

### 📹 Opción 1: Detección en Video

Procesa un archivo de video y muestra los resultados en pantalla:

```bash
python src/realtime_video.py --video inputs/videos/tu_video.mp4
```

**Guardar el video procesado:**
```bash
python src/realtime_video.py --video inputs/videos/tu_video.mp4 --save outputs/video_resultado.mp4
```

**Procesar sin abrir ventana (servidores sin GUI):**
```bash
python src/realtime_video.py --video inputs/videos/tu_video.mp4 --no-display --save outputs/video_resultado.mp4
```

**Parámetros opcionales:**
| Parámetro | Valor por defecto | Descripción |
|---|---|---|
| `--model` | `weights/best.pt` | Ruta al modelo YOLO |
| `--video` | *(requerido)* | Ruta del video de entrada |
| `--conf` | `0.25` | Umbral de confianza general |
| `--ppe-conf` | `0.25` | Umbral para guantes y otros EPP |
| `--helmet-conf` | `0.15` | Umbral especial para cascos |
| `--save` | `outputs/video_resultado.mp4` | Ruta del video de salida |
| `--no-display` | `false` | No abrir ventana de visualización |

> **Controles**: Presiona `q` para salir de la visualización.

---

### 📷 Opción 2: Detección con Cámara Web

Ejecuta la detección en tiempo real usando tu cámara web:

```bash
python src/realtime_webcam.py --camera 0
```

**Si tu cámara no es la 0, prueba con otros índices:**
```bash
python src/realtime_webcam.py --camera 2
```

**Guardar la grabación procesada:**
```bash
python src/realtime_webcam.py --camera 0 --save outputs/webcam_resultado.mp4
```

**Parámetros opcionales:**
| Parámetro | Valor por defecto | Descripción |
|---|---|---|
| `--model` | `weights/best.pt` | Ruta al modelo YOLO |
| `--camera` | `0` | Índice de la cámara |
| `--conf` | `0.25` | Umbral de confianza general |
| `--ppe-conf` | `0.25` | Umbral para guantes y otros EPP |
| `--helmet-conf` | `0.15` | Umbral especial para cascos |
| `--save` | *(vacío)* | Ruta opcional para guardar video |
| `--no-display` | `false` | No abrir ventana de visualización |

> **Controles**: Presiona `q` para salir.

---

### 🖼️ Opción 3: Detección en Imagen Individual

Analiza una imagen y guarda una copia anotada:

```bash
python src/detect_image.py --image inputs/images/tu_imagen.jpg
```

**Especificar la ruta de salida:**
```bash
python src/detect_image.py --image inputs/images/tu_imagen.jpg --save outputs/imagen_resultado.jpg
```

**Salida en consola:**
```
Imagen guardada en: outputs/imagen_resultado.jpg
Clases detectadas: ['Helmet', 'Hands', 'Person']
Cabezas sin casco: 0
Alertas: ['Posible falta de guantes']
```

---

### 📁 Opción 4: Procesamiento por Lotes (Batch)

Procesa todas las imágenes de una carpeta y genera un reporte CSV:

```bash
python src/batch_images.py --input inputs/images --output outputs/batch
```

**Resultado:**
- Imágenes anotadas individuales en `outputs/batch/`
- Archivo CSV con resumen: `outputs/batch/resultados_batch.csv`

---

## 📂 Estructura del Proyecto

```
Sistema-de-deteccion-de-EPP-en-construccion/
├── src/                        # Código fuente principal
│   ├── epp_utils.py            # Biblioteca central: filtros, alertas, renderizado
│   ├── realtime_video.py       # Procesador de videos
│   ├── realtime_webcam.py      # Procesador de cámara web en vivo
│   ├── detect_image.py         # Analizador de imagen individual
│   └── batch_images.py         # Procesador por lotes con reporte CSV
├── weights/                    # Modelo entrenado (no incluido, ver instalación)
│   └── best.pt                 # Pesos del modelo YOLOv8
├── inputs/                     # Datos de entrada
│   ├── images/                 # Imágenes para analizar
│   └── videos/                 # Videos para procesar
├── outputs/                    # Resultados generados (se crea automáticamente)
├── EXPLICACION.md              # Documentación técnica detallada del código
├── requirements.txt            # Dependencias de Python
├── .gitignore                  # Archivos excluidos de Git
└── README.md                   # Este archivo
```

---

## 📖 Explicación Técnica

Para una explicación exhaustiva de la arquitectura, la lógica de cada función, el mapeo de clases del modelo y los filtros geométricos, consulta el archivo [EXPLICACION.md](EXPLICACION.md).

---

## 🛠️ Tecnologías Utilizadas

- **[YOLOv8](https://docs.ultralytics.com/)** — Modelo de detección de objetos en tiempo real
- **[OpenCV](https://opencv.org/)** — Procesamiento de imágenes y video
- **[Python 3.10+](https://www.python.org/)** — Lenguaje de programación principal
- **Dataset [SH17](https://universe.roboflow.com/sh17)** — Dataset de seguridad industrial para entrenamiento
