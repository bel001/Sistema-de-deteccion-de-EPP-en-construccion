from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
try:
    from src.engine import EPPDetectionEngine
except ImportError:
    from engine import EPPDetectionEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Procesamiento acelerado por GPU de archivos de video.")
    parser.add_argument("--model", default="weights/best.pt", help="Ruta al modelo entrenado.")
    parser.add_argument("--video", required=True, help="Ruta al video de entrada.")
    parser.add_argument("--save", default="outputs/video_resultado.mp4", help="Ruta de guardado.")
    parser.add_argument("--display-width", type=int, default=854, help="Ancho de ventana.")
    parser.add_argument("--display-height", type=int, default=480, help="Alto de ventana.")
    parser.add_argument("--no-display", action="store_true", help="Procesa sin ventana gráfica.")
    args = parser.parse_args()

    engine = EPPDetectionEngine(args.model)
    print(f"Cargando motor de inferencia — Backend: {engine.runtime_label}")

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"No se encontró el video: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    display_w, display_h = args.display_width, args.display_height

    out_path = Path(args.save)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter.fourcc("m", "p", "4", "v"),
        original_fps,
        (display_w, display_h),
    )

    window_name = "EPP Construcción - Video GPU"
    if not args.no_display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, display_w, display_h)

    prev_time = time.time()
    frame_count = 0
    print("Procesando video con aceleración por GPU... Presiona 'q' para cancelar.")

    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            frame_count += 1
            resized = cv2.resize(frame, (display_w, display_h))
            
            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            result = engine.detect_video_frame(resized, fps)
            writer.write(result.annotated)

            if not args.no_display:
                cv2.imshow(window_name, result.annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Detenido por el usuario.")
                    break
    finally:
        cap.release()
        writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    print("=" * 60)
    print("Video procesado correctamente.")
    print(f"Frames procesados: {frame_count}")
    print(f"Video guardado en: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
