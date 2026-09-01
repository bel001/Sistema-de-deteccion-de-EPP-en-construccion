# Explicación Detallada del Sistema de Detección de EPP (Equipos de Protección Personal)

Este documento detalla exhaustivamente el propósito, la lógica de negocio, las modificaciones críticas realizadas y el análisis detallado del código fuente para el sistema de detección y cumplimiento de EPP en entornos de construcción.

---

## 1. Arquitectura y Propósito General

El propósito de este proyecto es procesar imágenes, videos y transmisiones de cámara web en tiempo real para verificar si el personal en una obra de construcción cumple con el uso de los Equipos de Protección Personal (EPP). 

El sistema utiliza un modelo de inteligencia artificial **YOLOv8** entrenado con una versión filtrada del dataset SH17 (específico para construcción). El flujo de procesamiento general es el siguiente:
1. **Entrada de Datos**: Carga de imagen, video o stream de cámara web.
2. **Inferencia de IA**: YOLOv8 predice la localización y clase de los objetos.
3. **Mapeo y Corrección de Clases**: Traducción de clases internas del modelo a la realidad semántica real.
4. **Filtrado Geométrico**: Eliminación de detecciones ruidosas o incoherentes por tamaño/proporción de caja.
5. **Asociación Espacial**: Relación entre partes del cuerpo y EPP (por ejemplo, verificar si una cabeza detectada tiene un casco encima).
6. **Cálculo de Cumplimiento**: Evaluación de reglas de negocio para determinar si hay infracciones.
7. **Renderizado Visual**: Anotación de cajas delimitadoras con colores específicos y un panel de alertas.
8. **Salida**: Guardado de archivos o visualización interactiva.

---

## 2. El Problema Crítico del Mapeo de Clases (Solucionado)

Durante el entrenamiento en Colab del dataset SH17, las 17 clases originales se redujeron a 9 clases de construcción y se reordenaron. Sin embargo, los metadatos internos del modelo guardados en `model.names` no se actualizaron con el orden real del dataset de salida. 

### Mapeo Interno Incorrecto (Metadatos) vs Semántica Real (Corregido)
Esto causaba que el modelo retornara predicciones que el código interpretaba incorrectamente (por ejemplo, detectar un casco pero pintarlo como "Foot", o detectar una cabeza expuesta y pintarla como "Safety-vest").

Para solucionar esto de raíz, implementamos `ACTUAL_CLASS_MAP` en `src/epp_utils.py:11`:
```python
ACTUAL_CLASS_MAP = {
    0: "Person",  1: "Ear", 2: "Glasses", 3: "Helmet", 4: "Face",
    5: "Gloves", 6: "Hands", 7: "Head", 8: "Shoes", 9: "Safety-vest",
}
```
Gracias a este diccionario, interceptamos cualquier clase devuelta por el modelo y la traducimos a su valor físico real. El modelo actual reentrenado (`best.pt`) ya expone 10 clases (0-9) y el mapeo es 1:1 con `model.names`.

### 2.1 Detección de Chaleco (Safety-vest) — Activa

> El modelo reentrenado (`best.pt`, 10 clases, `CONSTRUCTION_NAMES` en Colab) incluye `Safety-vest` como ID 9.
> * **Activación automática**: `EPPDetectionEngine` verifica `model_supports_class_id(model, 9)` (`src/engine.py:88`). Si el modelo no tiene la clase 9 (modelos antiguos de 9 clases), `check_vest=False` y no se generan alertas falsas.
> * **Filtro anti-falso-positivo**: `vest_is_valid_inside_person` exige que el chaleco esté dentro del torso de una `Person` y no solapado con `Head` (`src/epp_utils.py:402`), eliminando confusión cabello/rostro→chaleco.
> * **Colores**: `Safety-vest` cian `(0,255,255)` distinto de `Head` oro `(0,215,255)` y `Glasses` cian claro `(255,255,0)`.

---

## 3. Lógica de Asociación Inteligente: Cabeza vs Casco

El modelo YOLOv8 detecta de forma independiente la clase `Head` (Cabeza sin casco expuesta) y la clase `Helmet` (Casco). 
* **El Problema**: Cuando un trabajador lleva casco, el modelo suele detectar tanto el casco como la cabeza debajo del casco. Sin lógica asociativa, el sistema pintaba un recuadro verde de *"Casco"* y un recuadro rojo de *"Cabeza sin casco"* sobre la misma persona, lo cual es contradictorio.
* **La Solución**: Implementamos la función `head_has_helmet()`. Esta función toma la caja delimitadora de cada cabeza detectada y calcula su solapamiento geométrico respecto a todos los cascos de la escena.
  * Si hay una coincidencia (un casco encima de la cabeza con un solapamiento mayor al 15% o muy cerca verticalmente), el sistema considera la **Cabeza como Protegida** y omite el recuadro rojo de peligro.
  * Si una cabeza no tiene ningún casco asociado espacialmente, se considera **Cabeza Desprotegida** ("Sin casco"), se dibuja el recuadro rojo y se incrementa el contador de alertas.

---

## 4. Explicación Detallada del Código por Archivo

### A. [epp_utils.py](file:///home/rodrigo/Escritorio/Codigos/PROYECTO%20PERCEPCION/src/epp_utils.py) (La Biblioteca Central de Utilidades)

Este archivo contiene la lógica matemática, los filtros geométricos, las constantes del sistema y la lógica de renderizado gráfico de OpenCV.

#### Funciones y Lógica Interna:
1. **`box_area(box)`**:
   - *Propósito*: Calcula el área en píxeles de una caja delimitadora `(x1, y1, x2, y2)`.
   - *Lógica*: Multiplica el ancho `(x2 - x1)` por el alto `(y2 - y1)`.

2. **`intersection_area(box_a, box_b)`**:
   - *Propósito*: Calcula el área de solape (intersección) entre dos cajas.
   - *Lógica*: Encuentra las coordenadas máximas de inicio y mínimas de fin de ambas cajas y calcula el área del rectángulo resultante.

3. **`head_has_helmet(head_box, helmet_boxes, min_overlap)`**:
   - *Propósito*: Determina si una cabeza está protegida por un casco.
   - *Lógica*:
     - Mide el porcentaje de solapamiento de la cabeza con cada casco.
     - Si el solapamiento supera el umbral `min_overlap` (15%), retorna `True`.
     - Si no hay suficiente solapamiento directo pero el casco está verticalmente encima de la cabeza a corta distancia (dentro del 50% de la altura de la cabeza), también retorna `True` para prevenir fallos cuando las cajas no se tocan perfectamente.

4. **`passes_class_filters(...)`**:
   - *Propósito*: Reduce falsos positivos mediante filtros geométricos de tamaño relativo y relación de aspecto (aspect ratio).
   - *Lógica*:
     - **Personas**: Deben ocupar al menos el 2% del área total del frame para evitar personas en el fondo lejano que no se pueden evaluar bien.
     - **Cabezas**: Deben tener un aspecto ratio entre 0.25 y 2.40 (para evitar líneas verticales u horizontales erróneas) y no ocupar más del 18% del frame.
     - **Cascos**: Aspect ratio entre 0.35 y 2.80, área menor al 20% del frame.
     - **Guantes**: Área entre 0.1% y 12% del frame.

5. **`analyze_frame_level_compliance(detected_names, unprotected_heads)`** y **`analyze_video_compliance(...)`**:
   - *Propósito*: Evaluar el cumplimiento de EPP del frame.
   - *Lógica*:
     - Si `unprotected_heads` es mayor a 0, genera la alerta *"X persona(s) sin casco"*.
     - Si se detecta la clase `"Hands"` pero no la clase `"Gloves"`, genera la alerta *"Posible falta de guantes"*.

6. **`draw_detection_boxes(...)`**:
   - *Propósito*: Recorre las predicciones de YOLO, filtra las clases no deseadas, aplica la lógica Cabeza-Casco y dibuja las cajas en el frame.
   - *Lógica*:
     - **Fase 1**: Clasifica y recolecta por separado las cajas válidas de personas, cabezas y cascos.
     - **Fase 2**: Vuelve a iterar las cajas. Si procesa una cabeza y `head_has_helmet` es verdadero, no hace nada (está protegida). Si no tiene casco, la dibuja con un recuadro rojo de *"Sin casco"*. Para el resto de objetos (Personas, Cascos, Guantes, Calzado, etc.), dibuja el color correspondiente asignado en `CLASS_COLORS`.
     - Retorna el frame anotado, la lista de nombres detectados y el número de cabezas desprotegidas.

7. **`draw_status_panel(...)`** y **`draw_status_panel_big(...)`**:
   - *Propósito*: Renderizar en la esquina superior izquierda un menú semi-transparente negro que indica el FPS actual, las clases detectadas activamente y la lista de alertas activas en color rojo llamativo.

---

### B. [realtime_video.py](file:///home/rodrigo/Escritorio/Codigos/PROYECTO%20PERCEPCION/src/realtime_video.py) (Procesador de Videos)

* **Propósito**: Ejecutar la detección sobre archivos de video local (`.mp4`, `.avi`, etc.), mostrando el procesamiento en pantalla o guardándolo en disco de forma optimizada.
* **Lógica**:
  1. Carga el video con `cv2.VideoCapture` y obtiene parámetros como el ancho, alto y FPS original.
  2. Configura un archivo de salida usando el códec `mp4v` si el usuario solicita guardar el resultado (`--save`).
  3. Ejecuta la inferencia frame por frame llamando a `model.predict(frame, conf=inference_conf)`.
  4. Redimensiona los frames a una resolución estándar de procesamiento para acelerar la inferencia en equipos sin GPU dedicada.
  5. Llama a `draw_detection_boxes` para pintar los recuadros y a `draw_status_panel_big` para dibujar la telemetría.
  6. Escribe el frame procesado en el video de salida y opcionalmente lo muestra en pantalla.

---

### C. [realtime_webcam.py](file:///home/rodrigo/Escritorio/Codigos/PROYECTO%20PERCEPCION/src/realtime_webcam.py) (Procesador de Cámara en Vivo)

* **Propósito**: Procesar el flujo de video en vivo de una cámara web conectada al equipo bajo las mismas reglas y clases que el video.
* **Lógica**:
  1. Inicializa la captura en el índice de cámara especificado (`--camera 0` por defecto).
  2. Ajusta la resolución de la cámara a HD (1280x720) para equilibrar calidad y tasa de refresco.
  3. Ejecuta un bucle infinito que lee frames de la cámara de manera asíncrona.
  4. Ejecuta inferencia optimizada usando una resolución de imagen menor (`imgsz=416`) para garantizar alta velocidad (FPS) en tiempo real.
  5. Se habilitaron en la cámara las mismas clases detectadas en video (Persona, Cabeza, Casco, Manos, Guantes, Calzado y Gafas).
  6. Aplica la lógica de renderizado de EPP (incluyendo la detección de cabeza vs casco y la comprobación de guantes) y dibuja el panel de alertas.
  7. Cierra de forma segura todos los descriptores de la cámara y las ventanas de OpenCV al presionar la tecla `'q'`.

---

### D. [detect_image.py](file:///home/rodrigo/Escritorio/Codigos/PROYECTO%20PERCEPCION/src/detect_image.py) (Analizador de Imagen Única)

* **Propósito**: Procesar una única imagen fija de forma rápida y guardar una versión anotada en disco.
* **Lógica**:
  1. Lee la imagen de entrada desde la ruta especificada mediante `--image`.
  2. Ejecuta una sola inferencia de YOLO.
  3. Ejecuta el renderizado de cajas y el panel de alertas estático.
  4. Guarda la imagen en la ruta indicada por `--save`.
  5. Imprime en consola un resumen estructurado indicando las clases detectadas, las alertas generadas y si se encontraron personas sin casco.

---

### E. [batch_images.py](file:///home/rodrigo/Escritorio/Codigos/PROYECTO%20PERCEPCION/src/batch_images.py) (Procesador en Lote)

* **Propósito**: Procesar toda una carpeta de imágenes y generar un reporte unificado en formato CSV.
* **Lógica**:
  1. Escanea recursivamente el directorio de entrada `--input` buscando extensiones válidas de imágenes (`.jpg`, `.png`, `.webp`, etc.).
  2. Crea un archivo CSV de salida en el directorio `--output`.
  3. Procesa secuencialmente cada imagen aplicando el modelo, filtrado geométrico y telemetría de alertas.
  4. Guarda cada imagen anotada con el sufijo `_detected.jpg`.
  5. Escribe una fila en el CSV indicando:
     * Nombre de la imagen original.
     * Lista de clases detectadas separadas por punto y coma.
     * Alertas activas generadas.
     * Ruta de la imagen procesada guardada.

---

## 5. Parámetros de Configuración del Modelo

Cada script de ejecución te permite ajustar el comportamiento de la detección mediante argumentos en línea de comando:

* **`--conf`** (Confianza General, por defecto `0.25`):
  - Umbral mínimo para la detección de personas. Confianzas más bajas detectan personas más lejanas pero pueden generar falsos positivos.
* **`--ppe-conf`** (Confianza EPP, por defecto `0.25`):
  - Umbral de confianza aplicado a clases de tamaño mediano/pequeño como los guantes.
* **`--helmet-conf`** (Confianza Casco, por defecto `0.15`):
  - Umbral especial de confianza para cascos. Es intencionalmente más bajo (`0.15`) debido a que los cascos a menudo se ven pequeños en el frame o sufren de oclusión (sombras de la obra), permitiendo capturarlos con mayor sensibilidad.
