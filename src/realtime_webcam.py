from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
try:
    from src.engine import EPPDetectionEngine
except ImportError:
    from engine import EPPDetectionEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ejecuta detección de EPP usando una cámara local acelerada por GPU."
    )
    parser.add_argument("--model", default="weights/best.pt", help="Ruta del modelo YOLO.")
    parser.add_argument("--camera", type=int, default=0, help="Índice de cámara.")
    parser.add_argument("--save", default="", help="Ruta opcional para guardar el video.")
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Procesa sin abrir ventana. Útil para servidores o entornos sin GUI.",
    )
    return parser.parse_args()


def create_writer(save_path: str, width: int, height: int) -> cv2.VideoWriter | None:
    if not save_path:
        return None

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, 20, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"No se pudo crear el video de salida: {output_path}")

    return writer


def main() -> None:
    args = parse_args()
    engine = EPPDetectionEngine(args.model)
    print(f"Iniciando cámara — Backend: {engine.runtime_label}")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara. Prueba --camera 1 o --camera 2.")

    # Intentar MJPG para HD, fallback silencioso si no soportado
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    for _ in range(5):
        cap.read()

    width: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer: cv2.VideoWriter | None = create_writer(args.save, width, height)
    prev_time: float = time.time()

    if args.no_display:
        print("Ejecutando detección sin ventana. Usa Ctrl+C para salir.")
    else:
        print("Ejecutando detección en tiempo real. Presiona 'q' para salir.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            result = engine.detect_webcam_frame(frame)

            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            if writer is not None:
                writer.write(result.annotated)

            if args.no_display:
                continue

            cv2.imshow("EPP Construccion - Tiempo Real (GPU)", result.annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
