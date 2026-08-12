from __future__ import annotations

import argparse
from pathlib import Path

import cv2
try:
    from src.engine import EPPDetectionEngine
except ImportError:
    from engine import EPPDetectionEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detecta EPP en una imagen fija usando aceleración por GPU."
    )
    parser.add_argument("--model", default="weights/best.pt", help="Ruta del modelo YOLO.")
    parser.add_argument("--image", required=True, help="Ruta de la imagen de entrada.")
    parser.add_argument(
        "--save",
        default="outputs/imagen_resultado.jpg",
        help="Ruta donde se guardará la imagen procesada.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = EPPDetectionEngine(args.model)
    print(f"Cargando motor de inferencia — Backend: {engine.runtime_label}")

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"No se encontró la imagen: {image_path}")

    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"No se pudo leer la imagen: {image_path}")

    result = engine.detect_image_frame(frame)

    output_path = Path(args.save)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(output_path), result.annotated):
        raise RuntimeError(f"No se pudo guardar la imagen: {output_path}")

    print(f"Imagen guardada en: {output_path}")
    print("Clases detectadas:", sorted(set(result.names)))
    print("Cabezas sin casco:", result.unprotected_heads)
    print("Alertas:", result.alerts)


if __name__ == "__main__":
    main()
