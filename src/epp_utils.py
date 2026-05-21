from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import cv2


# Mapeo corregido de lo que el modelo realmente aprendió a detectar debido al desorden en el entrenamiento original
ACTUAL_CLASS_MAP = {
    0: "Person",       # Persona (0)
    1: "Ear",          # Oreja (1)
    2: "Glasses",      # Gafas (2)
    3: "Helmet",       # Casco (3) - En el metadata del modelo dice "Foot"
    4: "Face",         # Cara (4)
    5: "Gloves",       # Guantes (5)
    6: "Hands",        # Manos (6) - En el metadata del modelo dice "Shoes"
    7: "Head",         # Cabeza sin casco (7) - En el metadata del modelo dice "Safety-vest"
    8: "Shoes",        # Calzado (8) - En el metadata del modelo dice "Helmet"
}

CONSTRUCTION_CLASSES = list(ACTUAL_CLASS_MAP.values())

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_IOU = 0.50
MIN_HEAD_AREA_RATIO = 0.006

PERSON_CLASS_ID = 0
HEAD_CLASS_ID = 7
HANDS_CLASS_ID = 6
GLOVES_CLASS_ID = 5
SAFETY_VEST_CLASS_ID = -1  # El modelo NO detecta chalecos (no fue entrenado debido a un error de orden original)
HELMET_CLASS_ID = 3
SHOES_CLASS_ID = 8
GLASSES_CLASS_ID = 2

# Clases activas en realtime_video.py.
VIDEO_DISPLAY_CLASS_IDS = [
    PERSON_CLASS_ID,
    HEAD_CLASS_ID,
    HANDS_CLASS_ID,
    GLOVES_CLASS_ID,
    HELMET_CLASS_ID,
    SHOES_CLASS_ID,
    GLASSES_CLASS_ID,
]

# Clases usadas en webcam.
WEBCAM_CLASS_IDS = [
    PERSON_CLASS_ID,
    HEAD_CLASS_ID,
    HANDS_CLASS_ID,
    GLOVES_CLASS_ID,
    HELMET_CLASS_ID,
    SHOES_CLASS_ID,
    GLASSES_CLASS_ID,
]

CLASS_COLORS = {
    "Person": (255, 0, 0),        # Persona (Azul en BGR)
    "Head": (255, 255, 0),        # Cabeza (Amarillo)
    "Hands": (255, 0, 255),       # Manos (Magenta)
    "Gloves": (0, 165, 255),      # Guantes (Naranja)
    "Helmet": (0, 255, 0),        # Casco (Verde)
    "Shoes": (255, 128, 0),       # Calzado (Celeste)
    "Glasses": (255, 255, 0),     # Gafas (Cyan)
}

DISPLAY_NAMES = {
    "Person": "Persona",
    "Head": "Cabeza",
    "Hands": "Manos",
    "Gloves": "Guantes",
    "Helmet": "Casco",
    "Shoes": "Calzado",
    "Glasses": "Gafas",
}



def require_existing_path(path: str | Path, description: str) -> Path:
    """Devuelve la ruta validada o falla con un mensaje claro."""
    resolved_path = Path(path)

    if not resolved_path.exists():
        raise FileNotFoundError(f"No se encontro {description}: {resolved_path}")

    return resolved_path


def require_existing_file(path: str | Path, description: str) -> Path:
    """Valida que la ruta exista y sea un archivo."""
    resolved_path = require_existing_path(path, description)

    if not resolved_path.is_file():
        raise FileNotFoundError(f"La ruta de {description} no es un archivo: {resolved_path}")

    return resolved_path


def require_existing_directory(path: str | Path, description: str) -> Path:
    """Valida que la ruta exista y sea una carpeta."""
    resolved_path = require_existing_path(path, description)

    if not resolved_path.is_dir():
        raise NotADirectoryError(f"La ruta de {description} no es una carpeta: {resolved_path}")

    return resolved_path


def collect_detected_names(result, class_names: Mapping[int, str]) -> list[str]:
    """Extrae los nombres de clases detectadas desde un resultado de YOLO."""
    if result.boxes is None:
        return []

    return [class_names[int(box.cls[0])] for box in result.boxes]


def get_display_name(class_name: str) -> str:
    """Convierte el nombre tecnico del modelo a una etiqueta legible."""
    return DISPLAY_NAMES.get(class_name, class_name)


def format_detected_names(detected_names: Iterable[str], limit: int | None = None) -> str:
    """Formatea clases detectadas para mostrarlas en el panel."""
    display_names = sorted({get_display_name(name) for name in detected_names})

    if limit is not None:
        display_names = display_names[:limit]

    return ", ".join(display_names) if display_names else "Nada"


def head_has_helmet(
    head_box: tuple[float, float, float, float],
    helmet_boxes: Sequence[tuple[float, float, float, float]],
    min_overlap: float = 0.15,
) -> bool:
    """
    Verifica si una cabeza tiene un casco superpuesto.
    Retorna True si la cabeza esta protegida por un casco.
    """
    if not helmet_boxes:
        return False

    head_area = box_area(head_box)
    if head_area <= 0:
        return False

    hx1, hy1, hx2, hy2 = head_box
    head_cx = (hx1 + hx2) / 2
    head_cy = (hy1 + hy2) / 2

    for helmet_box in helmet_boxes:
        inter = intersection_area(head_box, helmet_box)
        helmet_area = box_area(helmet_box)

        # Solapamiento relativo al area de la cabeza o del casco
        overlap_head = inter / head_area if head_area > 0 else 0
        overlap_helmet = inter / helmet_area if helmet_area > 0 else 0

        if overlap_head >= min_overlap or overlap_helmet >= min_overlap:
            return True

        # Tambien verificamos proximidad vertical: si el casco esta justo encima de la cabeza
        kx1, ky1, kx2, ky2 = helmet_box
        helmet_cx = (kx1 + kx2) / 2
        head_h = max(1, hy2 - hy1)

        horizontal_close = abs(head_cx - helmet_cx) < (hx2 - hx1) * 0.8
        vertical_close = (hy1 - ky2) < head_h * 0.5 and (hy1 - ky2) > -head_h * 0.8

        if horizontal_close and vertical_close:
            return True

    return False


def analyze_frame_level_compliance(
    detected_names: Iterable[str],
    unprotected_heads: int = 0,
) -> tuple[bool, list[str]]:
    """
    Reglas simples para imagen, batch y webcam.
    Usa conteo de cabezas sin casco si esta disponible.
    """
    names = set(detected_names)
    alerts: list[str] = []

    if unprotected_heads > 0:
        alerts.append(f"{unprotected_heads} persona(s) sin casco")
    elif ("Person" in names or "Head" in names) and "Helmet" not in names:
        alerts.append("Posible falta de casco")

    if "Hands" in names and "Gloves" not in names:
        alerts.append("Posible falta de guantes")

    return bool(alerts), alerts


def analyze_video_compliance(
    detected_names: Iterable[str],
    unprotected_heads: int = 0,
) -> list[str]:
    """
    Reglas usadas por realtime_video.py.
    Usa conteo de cabezas sin casco si esta disponible.
    """
    names = set(detected_names)
    alerts: list[str] = []

    if unprotected_heads > 0:
        alerts.append(f"{unprotected_heads} persona(s) sin casco")
    elif ("Person" in names or "Head" in names) and "Helmet" not in names:
        alerts.append("Posible falta de casco")

    if "Hands" in names and "Gloves" not in names:
        alerts.append("Posible falta de guantes")

    return alerts


def draw_status_panel(
    frame,
    fps: float,
    detected_names: Iterable[str],
    alerts: Sequence[str],
) -> None:
    """Dibuja el panel compacto usado por imagen, batch y webcam."""
    y = 35

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    y += 35
    detected_text = "Detectado: " + format_detected_names(detected_names, limit=7)

    cv2.putText(
        frame,
        detected_text,
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
    )

    y += 35

    if alerts:
        cv2.putText(
            frame,
            "ALERTA EPP",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            3,
        )

        y += 35

        for alert in alerts[:4]:
            cv2.putText(
                frame,
                f"- {alert}",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 0, 255),
                2,
            )
            y += 28
    else:
        cv2.putText(
            frame,
            "Sin alerta evidente",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )


def box_area(box: tuple[float, float, float, float]) -> float:
    """Calcula el area de una caja delimitadora."""
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection_area(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float]
) -> float:
    """Calcula el area de interseccion entre dos cajas delimitadoras."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def vest_overlaps_head(
    vest_box: tuple[float, float, float, float],
    head_boxes: Sequence[tuple[float, float, float, float]],
) -> bool:
    """
    Rechaza un chaleco si se solapa significativamente con cualquier
    caja de Cabeza detectada. Esto elimina el falso positivo comun
    donde el modelo confunde cabello/rostro con un chaleco.
    """
    if not head_boxes:
        return False

    vest_area = box_area(vest_box)
    if vest_area <= 0:
        return False

    for head_box in head_boxes:
        inter = intersection_area(vest_box, head_box)
        head_area = box_area(head_box)

        # Si el chaleco cubre mas del 25% de la cabeza, es sospechoso.
        if head_area > 0 and inter / head_area > 0.25:
            return True

        # Si mas del 30% del chaleco se solapa con la cabeza, es sospechoso.
        if inter / vest_area > 0.30:
            return True

    return False


def vest_is_valid_inside_person(
    vest_box: tuple[float, float, float, float],
    person_boxes: Sequence[tuple[float, float, float, float]],
    head_boxes: Sequence[tuple[float, float, float, float]] = (),
) -> bool:
    """
    Valida que el chaleco este dentro de una persona y en la zona del torso.
    Esto evita falsos positivos donde el modelo marca la cabeza/casco o cabello
    como Safety-vest.
    """
    # Si se solapa con una cabeza detectada, rechazar directamente.
    if vest_overlaps_head(vest_box, head_boxes):
        return False

    if not person_boxes:
        return True

    vx1, vy1, vx2, vy2 = vest_box
    vest_area = box_area(vest_box)

    if vest_area <= 0:
        return False

    vest_center_y = (vy1 + vy2) / 2

    for person_box in person_boxes:
        px1, py1, px2, py2 = person_box
        person_h = max(1.0, py2 - py1)

        inter = intersection_area(vest_box, person_box)
        overlap_ratio = inter / vest_area

        relative_y = (vest_center_y - py1) / person_h

        # El chaleco debe estar mayormente dentro de la persona.
        if overlap_ratio < 0.55:
            continue

        # El chaleco no deberia estar en la cabeza (zona superior del 23%).
        if relative_y < 0.23:
            continue

        # Tampoco deberia estar muy abajo (zona inferior del 88%).
        if relative_y > 0.88:
            continue

        return True

    return False


def passes_class_filters(
    name: str,
    conf: float,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    frame_width: int,
    frame_height: int,
    person_boxes: Sequence[tuple[int, int, int, int]],
    head_boxes: Sequence[tuple[int, int, int, int]] = (),
    conf_thresh: float = 0.25,
    ppe_conf_thresh: float = 0.25,
    helmet_conf_thresh: float = 0.15,
) -> bool:
    """
    Aplica filtros de calidad y tamano segun la clase detectada.
    """
    box_w = x2 - x1
    box_h = y2 - y1
    frame_area = frame_width * frame_height
    current_area = box_w * box_h

    if box_w <= 0 or box_h <= 0:
        return False

    # 1. Validar umbral de confianza especifico
    if name in {"Person", "Head", "Hands"}:
        if conf < conf_thresh:
            return False
    elif name in {"Safety-vest", "Gloves"}:
        if conf < ppe_conf_thresh:
            return False
    elif name == "Helmet":
        if conf < helmet_conf_thresh:
            return False

    # 2. Filtros geometricos de tamano y relacion de aspecto
    if name == "Person":
        if current_area < frame_area * 0.02:
            return False

    elif name == "Head":
        if current_area < frame_area * 0.0015:
            return False
        if current_area > frame_area * 0.18:
            return False
        aspect_ratio = box_w / max(box_h, 1)
        if aspect_ratio < 0.25 or aspect_ratio > 2.40:
            return False

    elif name == "Hands":
        if current_area > frame_area * 0.15:
            return False

    elif name == "Gloves":
        if current_area < frame_area * 0.001:
            return False
        if current_area > frame_area * 0.12:
            return False

    elif name == "Safety-vest":
        if current_area < frame_area * 0.015:
            return False
        if current_area > frame_area * 0.45:
            return False
        aspect_ratio = box_w / max(box_h, 1)
        if aspect_ratio < 0.35 or aspect_ratio > 3.20:
            return False
        # Validar torso y descartar solapamiento con cabeza
        if not vest_is_valid_inside_person(
            (float(x1), float(y1), float(x2), float(y2)),
            [(float(px1), float(py1), float(px2), float(py2)) for px1, py1, px2, py2 in person_boxes],
            [(float(hx1), float(hy1), float(hx2), float(hy2)) for hx1, hy1, hx2, hy2 in head_boxes],
        ):
            return False

    elif name == "Helmet":
        if current_area < frame_area * 0.001:
            return False
        if current_area > frame_area * 0.20:
            return False
        aspect_ratio = box_w / max(box_h, 1)
        if aspect_ratio < 0.35 or aspect_ratio > 2.80:
            return False

    return True


def draw_big_label(frame, text: str, x1: int, y1: int, color: tuple[int, int, int]) -> None:
    """Dibuja una etiqueta grande y legible sobre una caja de deteccion."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.68
    thickness = 2
    text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size

    label_y1 = max(y1 - text_h - baseline - 8, 0)
    label_y2 = label_y1 + text_h + baseline + 8
    label_x1 = max(x1, 0)
    label_x2 = min(x1 + text_w + 12, frame.shape[1] - 1)

    cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), color, -1)
    cv2.putText(
        frame,
        text,
        (label_x1 + 6, label_y2 - baseline - 4),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )


def draw_detection_boxes(
    frame,
    result,
    model,
    scale_x: float,
    scale_y: float,
    class_ids: Sequence[int] = VIDEO_DISPLAY_CLASS_IDS,
    conf_thresh: float = 0.25,
    ppe_conf_thresh: float = 0.25,
    helmet_conf_thresh: float = 0.15,
) -> tuple[object, list[str], int]:
    """
    Dibuja cajas filtradas en espanol aplicando filtros geometricos y de confianza.
    Retorna (frame, detected_names, unprotected_heads).
    """
    detected_names: list[str] = []
    unprotected_heads: int = 0
    enabled_class_ids = set(class_ids)

    if result.boxes is None:
        return frame, detected_names, 0

    frame_height, frame_width = frame.shape[:2]

    # Primero recolectamos todas las cajas de Personas, Cabezas y Cascos validas
    person_boxes: list[tuple[int, int, int, int]] = []
    head_boxes: list[tuple[int, int, int, int]] = []
    helmet_boxes: list[tuple[int, int, int, int]] = []

    for box in result.boxes:
        cls_id = int(box.cls[0])
        name = ACTUAL_CLASS_MAP.get(cls_id, model.names[cls_id])
        conf = float(box.conf[0])

        if cls_id not in enabled_class_ids:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)

        if name == "Person" and conf >= conf_thresh:
            if (x2 - x1) * (y2 - y1) >= (frame_width * frame_height * 0.02):
                person_boxes.append((x1, y1, x2, y2))
        elif name == "Head" and conf >= conf_thresh:
            head_boxes.append((x1, y1, x2, y2))
        elif name == "Helmet" and conf >= helmet_conf_thresh:
            helmet_boxes.append((x1, y1, x2, y2))

    # Ahora procesamos todas las detecciones y las dibujamos
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        name = ACTUAL_CLASS_MAP.get(cls_id, model.names[cls_id])

        if cls_id not in enabled_class_ids:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)

        if not passes_class_filters(
            name,
            conf,
            x1,
            y1,
            x2,
            y2,
            frame_width,
            frame_height,
            person_boxes,
            head_boxes,
            conf_thresh,
            ppe_conf_thresh,
            helmet_conf_thresh,
        ):
            continue

        # Para Head: verificar si tiene casco superpuesto
        if name == "Head":
            current_head = (float(x1), float(y1), float(x2), float(y2))
            float_helmet_boxes = [
                (float(hx1), float(hy1), float(hx2), float(hy2))
                for hx1, hy1, hx2, hy2 in helmet_boxes
            ]
            has_helmet = head_has_helmet(current_head, float_helmet_boxes)

            if has_helmet:
                # Cabeza protegida: no dibujar (el casco ya se dibuja)
                detected_names.append(name)
                continue
            else:
                # Cabeza sin casco: dibujar en rojo con alerta
                color = (0, 0, 255)  # Rojo
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                draw_big_label(frame, f"Sin casco {conf:.2f}", x1, y1, color)
                detected_names.append("Head_unprotected")
                unprotected_heads += 1
                continue

        color = CLASS_COLORS.get(name, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        draw_big_label(frame, f"{get_display_name(name)} {conf:.2f}", x1, y1, color)
        detected_names.append(name)

    return frame, detected_names, unprotected_heads


def draw_status_panel_big(
    frame,
    fps: float,
    detected_names: Iterable[str],
    alerts: Sequence[str],
) -> None:
    """Dibuja el panel grande usado por la demo de video."""
    panel_x1 = 15
    panel_y1 = 15
    panel_x2 = min(720, frame.shape[1] - 1)
    panel_y2 = min(215, frame.shape[0] - 1)

    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x1, panel_y1), (panel_x2, panel_y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    y = 45

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (30, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    y += 33
    detected_text = "Detectado: " + format_detected_names(detected_names)

    if len(detected_text) > 70:
        detected_text = detected_text[:67] + "..."

    cv2.putText(
        frame,
        detected_text,
        (30, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    y += 35

    if alerts:
        cv2.putText(
            frame,
            "ALERTA EPP",
            (30, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )

        y += 30

        for alert in alerts[:4]:
            cv2.putText(
                frame,
                f"- {alert}",
                (30, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            y += 25
    else:
        cv2.putText(
            frame,
            "Sin alerta evidente",
            (30, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

