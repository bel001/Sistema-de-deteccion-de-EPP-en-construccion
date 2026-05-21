from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

from epp_utils import (
    VIDEO_DISPLAY_CLASS_IDS,
    analyze_frame_level_compliance,
    draw_detection_boxes,
    draw_status_panel,
    require_existing_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detecta EPP en una imagen y guarda una copia anotada."
    )
    parser.add_argument("--model", default="weights/best.pt", help="Ruta del modelo YOLO.")
    parser.add_argument("--image", required=True, help="Ruta de la imagen de entrada.")
    parser.add_argument("--conf", type=float, default=0.25, help="Umbral de confianza general.")
    parser.add_argument("--ppe-conf", type=float, default=0.25, help="Umbral para chaleco y guantes.")
    parser.add_argument("--helmet-conf", type=float, default=0.15, help="Umbral especial para casco.")
    parser.add_argument(
        "--save",
        default="outputs/imagen_resultado.jpg",
        help="Ruta donde se guardara la imagen procesada.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = require_existing_file(args.model, "el modelo")
    image_path = require_existing_file(args.image, "la imagen")
    output_path = Path(args.save)

    model = YOLO(str(model_path))
    frame = cv2.imread(str(image_path))

    if frame is None:
        raise RuntimeError(f"No se pudo leer la imagen: {image_path}")

    inference_conf = min(args.conf, args.ppe_conf, args.helmet_conf)
    result = model.predict(frame, conf=inference_conf, verbose=False)[0]

    annotated = frame.copy()
    annotated, detected_names, unprotected_heads = draw_detection_boxes(
        annotated,
        result,
        model,
        scale_x=1.0,
        scale_y=1.0,
        class_ids=VIDEO_DISPLAY_CLASS_IDS,
        conf_thresh=args.conf,
        ppe_conf_thresh=args.ppe_conf,
        helmet_conf_thresh=args.helmet_conf,
    )

    _, alerts = analyze_frame_level_compliance(detected_names, unprotected_heads)
    draw_status_panel(annotated, 0.0, detected_names, alerts)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(output_path), annotated):
        raise RuntimeError(f"No se pudo guardar la imagen: {output_path}")

    print(f"Imagen guardada en: {output_path}")
    print("Clases detectadas:", sorted(set(detected_names)))
    print("Cabezas sin casco:", unprotected_heads)
    print("Alertas:", alerts)


if __name__ == "__main__":
    main()

