from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
try:
    from src.engine import EPPDetectionEngine
    from src.epp_utils import IMAGE_EXTENSIONS, require_existing_directory
except ImportError:
    from engine import EPPDetectionEngine
    from epp_utils import IMAGE_EXTENSIONS, require_existing_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Procesa una carpeta de imágenes en lote usando aceleración GPU."
    )
    parser.add_argument("--model", default="weights/best.pt", help="Ruta del modelo YOLO.")
    parser.add_argument("--input", default="inputs/images", help="Carpeta de imágenes.")
    parser.add_argument("--output", default="outputs/batch", help="Carpeta de salida.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = EPPDetectionEngine(args.model)
    print(f"Cargando motor de inferencia — Backend: {engine.runtime_label}")

    input_dir = require_existing_directory(args.input, "la carpeta de imágenes")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(
        path for path in input_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    csv_path = output_dir / "resultados_batch.csv"
    processed_count = 0

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image", "detected_classes", "alerts", "output_image"])

        for image_path in images:
            frame = cv2.imread(str(image_path))
            if frame is None:
                print(f"Imagen omitida porque no se pudo leer: {image_path}")
                continue

            result = engine.detect_batch_frame(frame)
            output_image_path = output_dir / f"{image_path.stem}_detected.jpg"

            if not cv2.imwrite(str(output_image_path), result.annotated):
                raise RuntimeError(f"No se pudo guardar la imagen: {output_image_path}")

            writer.writerow(
                [
                    str(image_path),
                    ";".join(sorted(set(result.names))),
                    ";".join(result.alerts),
                    str(output_image_path),
                ]
            )
            processed_count += 1

    print(f"Imágenes encontradas: {len(images)}")
    print(f"Imágenes procesadas: {processed_count}")
    print(f"CSV guardado en: {csv_path}")


if __name__ == "__main__":
    main()
