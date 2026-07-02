from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from epp_utils import (
    SAFETY_VEST_CLASS_ID,
    VIDEO_DISPLAY_CLASS_IDS,
    analyze_compliance,
    draw_detection_boxes,
    draw_status_panel_big,
    filter_supported_class_ids,
    model_supports_class_id,
)

# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="weights/best.pt",
        help="Ruta al modelo entrenado best.pt"
    )

    parser.add_argument(
        "--video",
        required=True,
        help="Ruta al video de entrada"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.50,
        help="Umbral general para Persona, Cabeza y Manos"
    )

    parser.add_argument(
        "--ppe-conf",
        type=float,
        default=0.45,
        help="Umbral para Chaleco y Guantes"
    )

    parser.add_argument(
        "--helmet-conf",
        type=float,
        default=0.15,
        help="Umbral especial para Casco"
    )

    parser.add_argument(
        "--save",
        default="outputs/video_resultado.mp4",
        help="Ruta donde se guardara el video procesado"
    )

    parser.add_argument(
        "--display-width",
        type=int,
        default=854,
        help="Ancho de la ventana de visualizacion"
    )

    parser.add_argument(
        "--display-height",
        type=int,
        default=480,
        help="Alto de la ventana de visualizacion"
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Tamano de inferencia. 640 detecta mejor; 416 es mas rapido."
    )

    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Procesa y guarda el video sin abrir ventana"
    )

    args = parser.parse_args()

    model_path = Path(args.model)
    video_path = Path(args.video)

    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontro el modelo: {model_path}. "
            "Copia best.pt dentro de weights/best.pt"
        )

    if not video_path.exists():
        raise FileNotFoundError(
            f"No se encontro el video: {video_path}"
        )

    print("Cargando modelo...")
    model = YOLO(str(model_path))
    display_class_ids: list[int] = filter_supported_class_ids(model, VIDEO_DISPLAY_CLASS_IDS)
    check_vest: bool = model_supports_class_id(model, SAFETY_VEST_CLASS_ID)

    print("Abriendo video...")
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"No se pudo abrir el video: {video_path}"
        )

    original_fps: float | np.float32 = cap.get(cv2.CAP_PROP_FPS)

    if original_fps is None or original_fps <= 0:
        original_fps = 20

    original_width: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(
        f"Video original: "
        f"{original_width}x{original_height} @ {original_fps:.2f} FPS"
    )

    output_width: int = args.display_width
    output_height: int = args.display_height

    output_path = Path(args.save)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter.fourcc("m", "p", "4", "v"),
        original_fps,
        (output_width, output_height)
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"No se pudo crear el video de salida: {output_path}"
        )

    window_name = "EPP Construccion - Video"

    if not args.no_display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, output_width, output_height)

    prev_time: float = time.time()
    frame_count: int = 0

    inference_conf: float | np.float32 = min(args.conf, args.ppe_conf, args.helmet_conf)

    print("Procesando video...")

    if not args.no_display:
        print("Presiona 'q' para salir.")

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        frame_count += 1

        result = model.predict(
            frame,
            conf=inference_conf,
            iou=0.50,
            imgsz=args.imgsz,
            classes=display_class_ids,
            verbose=False
        )[0]

        display_frame = cv2.resize(
            frame,
            (output_width, output_height)
        )

        scale_x = output_width / original_width
        scale_y = output_height / original_height

        display_frame, detected_names, unprotected_heads = draw_detection_boxes(
            display_frame,
            result,
            model,
            scale_x,
            scale_y,
            class_ids=display_class_ids,
            conf_thresh=args.conf,
            ppe_conf_thresh=args.ppe_conf,
            helmet_conf_thresh=args.helmet_conf,
        )

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        alerts = analyze_compliance(
            detected_names,
            unprotected_heads,
            check_vest=check_vest,
        )

        draw_status_panel_big(
            display_frame,
            fps,
            detected_names,
            alerts
        )

        writer.write(display_frame)

        if not args.no_display:
            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("Proceso detenido por el usuario.")
                break

    cap.release()
    writer.release()

    if not args.no_display:
        cv2.destroyAllWindows()

    print("=" * 60)
    print("Video procesado correctamente.")
    print(f"Frames procesados: {frame_count}")
    print(f"Video guardado en: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
