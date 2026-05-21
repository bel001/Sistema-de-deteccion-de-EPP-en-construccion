from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
from ultralytics import YOLO

from epp_utils import (
    IMAGE_EXTENSIONS,
    VIDEO_DISPLAY_CLASS_IDS,
    analyze_frame_level_compliance,
    draw_detection_boxes,
    draw_status_panel,
    require_existing_directory,
    require_existing_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Procesa una carpeta de imagenes y genera un CSV con resultados."
    )
    parser.add_argument("--model", default="weights/best.pt", help="Ruta del modelo YOLO.")
    parser.add_argument("--input", default="inputs/images", help="Carpeta de imagenes.")
    parser.add_argument("--output", default="outputs/batch", help="Carpeta de salida.")
    parser.add_argument("--conf", type=float, default=0.25, help="Umbral de confianza general.")
    parser.add_argument("--ppe-conf", type=float, default=0.25, help="Umbral para chaleco y guantes.")
    parser.add_argument("--helmet-conf", type=float, default=0.15, help="Umbral especial para casco.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = require_existing_file(args.model, "el modelo")
    input_dir = require_existing_directory(args.input, "la carpeta de imagenes")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    images = sorted(
        path for path in input_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    csv_path = output_dir / "resultados_batch.csv"
    processed_count = 0

    inference_conf = min(args.conf, args.ppe_conf, args.helmet_conf)

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image", "detected_classes", "alerts", "output_image"])

        for image_path in images:
            frame = cv2.imread(str(image_path))

            if frame is None:
                print(f"Imagen omitida porque no se pudo leer: {image_path}")
                continue

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

            output_image_path = output_dir / f"{image_path.stem}_detected.jpg"

            if not cv2.imwrite(str(output_image_path), annotated):
                raise RuntimeError(f"No se pudo guardar la imagen: {output_image_path}")

            writer.writerow(
                [
                    str(image_path),
                    ";".join(sorted(set(detected_names))),
                    ";".join(alerts),
                    str(output_image_path),
                ]
            )
            processed_count += 1

    print(f"Imagenes encontradas: {len(images)}")
    print(f"Imagenes procesadas: {processed_count}")
    print(f"CSV guardado en: {csv_path}")


if __name__ == "__main__":
    main()

