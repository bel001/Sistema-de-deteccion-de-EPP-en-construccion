from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from epp_utils import (
    DEFAULT_IOU,
    SAFETY_VEST_CLASS_ID,
    WEBCAM_CLASS_IDS,
    analyze_frame_level_compliance,
    draw_detection_boxes,
    draw_status_panel,
    filter_supported_class_ids,
    model_supports_class_id,
    require_existing_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ejecuta deteccion de EPP usando una camara local."
    )
    parser.add_argument("--model", default="weights/best.pt", help="Ruta del modelo YOLO.")
    parser.add_argument("--camera", type=int, default=0, help="Indice de camara.")
    parser.add_argument("--conf", type=float, default=0.25, help="Umbral de confianza general.")
    parser.add_argument("--ppe-conf", type=float, default=0.25, help="Umbral para chaleco y guantes.")
    parser.add_argument("--helmet-conf", type=float, default=0.15, help="Umbral especial para casco.")
    parser.add_argument("--save", default="", help="Ruta opcional para guardar el video.")
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Procesa sin abrir ventana. Util para servidores o entornos sin GUI.",
    )
    return parser.parse_args()


def create_writer(save_path: str, width: int, height: int) -> cv2.VideoWriter | None:
    if not save_path:
        return None

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter.fourcc("m", "p", "4", "v")
    writer = cv2.VideoWriter(str(output_path), fourcc, 20, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"No se pudo crear el video de salida: {output_path}")

    return writer


def main() -> None:
    args = parse_args()
    model_path = require_existing_file(args.model, "el modelo")

    model = YOLO(str(model_path))
    webcam_class_ids: list[int] = filter_supported_class_ids(model, WEBCAM_CLASS_IDS)
    check_vest: bool = model_supports_class_id(model, SAFETY_VEST_CLASS_ID)
    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la camara. Prueba --camera 1 o --camera 2.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    width: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer: cv2.VideoWriter | None = create_writer(args.save, width, height)
    prev_time: float = time.time()

    if args.no_display:
        print("Ejecutando deteccion sin ventana. Usa Ctrl+C para salir.")
    else:
        print("Ejecutando deteccion en tiempo real. Presiona q para salir.")

    inference_conf: float | np.float32 = min(args.conf, args.ppe_conf, args.helmet_conf)

    try:
        while True:
            ok: bool
            ok, frame = cap.read()

            if not ok:
                break

            result = model.predict(
                frame,
                conf=inference_conf,
                iou=DEFAULT_IOU,
                imgsz=416,
                classes=webcam_class_ids,
                verbose=False,
            )[0]

            annotated, detected_names, unprotected_heads = draw_detection_boxes(
                frame,
                result,
                model,
                scale_x=1.0,
                scale_y=1.0,
                class_ids=webcam_class_ids,
                conf_thresh=args.conf,
                ppe_conf_thresh=args.ppe_conf,
                helmet_conf_thresh=args.helmet_conf,
            )

            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            _, alerts = analyze_frame_level_compliance(
                detected_names,
                unprotected_heads,
                check_vest=check_vest,
            )
            draw_status_panel(annotated, fps, detected_names, alerts)

            if writer is not None:
                writer.write(annotated)

            if args.no_display:
                continue

            cv2.imshow("EPP Construccion - YOLOv8", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()

        if writer is not None:
            writer.release()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
