from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

try:
    from src.epp_utils import (
        DEFAULT_IOU,
        SAFETY_VEST_CLASS_ID,
        VIDEO_DISPLAY_CLASS_IDS,
        WEBCAM_CLASS_IDS,
        analyze_compliance,
        draw_detection_boxes,
        draw_status_panel,
        draw_status_panel_big,
        filter_supported_class_ids,
        model_supports_class_id,
    )
except ImportError:
    from epp_utils import (
        DEFAULT_IOU,
        SAFETY_VEST_CLASS_ID,
        VIDEO_DISPLAY_CLASS_IDS,
        WEBCAM_CLASS_IDS,
        analyze_compliance,
        draw_detection_boxes,
        draw_status_panel,
        draw_status_panel_big,
        filter_supported_class_ids,
        model_supports_class_id,
    )


def _project_root() -> Path:
    """Resuelve la raíz del proyecto (carpeta que contiene weights/)."""
    # src/engine.py -> src -> root
    return Path(__file__).resolve().parent.parent


def ensure_weights_exist(weights_dir: str | Path | None = None) -> Path:
    """Garantiza que best.pt exista. Si solo existe best.pt.zip (copia idéntica), lo duplica."""
    wdir = Path(weights_dir) if weights_dir is not None else _project_root() / "weights"
    # Si es ruta relativa, resolver respecto a la raíz del proyecto
    if not wdir.is_absolute():
        wdir = (_project_root() / wdir).resolve()
    target_pt = wdir / "best.pt"
    target_zip = wdir / "best.pt.zip"

    if target_pt.exists():
        return target_pt

    if target_zip.exists():
        # best.pt y best.pt.zip son el mismo archivo zip de torch.save; basta con copiar
        shutil.copyfile(target_zip, target_pt)
        return target_pt

    raise FileNotFoundError(
        f"No se encontró el modelo {target_pt}. Coloca best.pt en {wdir} "
        f"(o best.pt.zip que se copiará automáticamente)."
    )


@dataclass
class DetectionResult:
    annotated: np.ndarray
    names: list[str]
    unprotected_heads: int
    alerts: list[str]
    persons: int = 0


class EPPDetectionEngine:
    """Motor unificado de inferencia de EPP con aceleración GPU CUDA."""

    def __init__(self, model_path: str | Path | None = None) -> None:
        # Resolver ruta por defecto relativa a la raíz del proyecto, no al CWD
        if model_path is None:
            default = _project_root() / "weights" / "best.pt"
            path = default
        else:
            path = Path(model_path)
            if not path.is_absolute():
                # Permitir rutas relativas tanto al CWD como a la raíz del proyecto
                candidate = Path.cwd() / path
                project_candidate = _project_root() / path
                if candidate.exists():
                    path = candidate
                elif project_candidate.exists():
                    path = project_candidate

        if not path.exists():
            # Intentar recuperar desde weights/ (zip copia) antes de fallar
            parent = path.parent if str(path.parent) != "" else _project_root() / "weights"
            path = ensure_weights_exist(parent)

        self.model_path = path
        self.uses_cuda = torch.cuda.is_available()
        self.predict_device: int | str = 0 if self.uses_cuda else "cpu"
        self.runtime_label = self._runtime_label()

        if self.uses_cuda:
            torch.backends.cudnn.benchmark = True

        self.model = YOLO(str(self.model_path))
        self.webcam_ids = filter_supported_class_ids(self.model, WEBCAM_CLASS_IDS)
        self.video_ids = filter_supported_class_ids(self.model, VIDEO_DISPLAY_CLASS_IDS)
        self.check_vest = model_supports_class_id(self.model, SAFETY_VEST_CLASS_ID)
        self._last_yolo_result: object = None
        self._last_webcam_res: DetectionResult | None = None

    def _runtime_label(self) -> str:
        if self.uses_cuda:
            gpu_name = torch.cuda.get_device_name(0)
            return f"GPU: {gpu_name}"
        return "CPU (Sin GPU CUDA)"

    def _predict(
        self,
        frame: np.ndarray,
        conf: float,
        class_ids: list[int],
        imgsz: int,
    ) -> object:
        return self.model.predict(
            frame,
            conf=conf,
            iou=DEFAULT_IOU,
            imgsz=imgsz,
            classes=class_ids,
            device=self.predict_device,
            verbose=False,
        )[0]

    def detect_frame(
        self,
        frame: np.ndarray,
        class_ids: list[int],
        conf_thresh: float = 0.25,
        ppe_conf_thresh: float = 0.25,
        helmet_conf_thresh: float = 0.15,
        imgsz: int = 640,
    ) -> tuple[DetectionResult, object]:
        inference_conf = min(conf_thresh, ppe_conf_thresh, helmet_conf_thresh)
        result = self._predict(frame, inference_conf, class_ids, imgsz)

        annotated, names, heads = draw_detection_boxes(
            frame.copy(),
            result,
            self.model,
            scale_x=1.0,
            scale_y=1.0,
            class_ids=class_ids,
            conf_thresh=conf_thresh,
            ppe_conf_thresh=ppe_conf_thresh,
            helmet_conf_thresh=helmet_conf_thresh,
        )
        alerts = analyze_compliance(names, heads, check_vest=self.check_vest)
        persons = sum(1 for name in names if name == "Person")
        det_res = DetectionResult(annotated, names, int(heads), alerts, persons)
        return det_res, result

    def detect_webcam_frame(self, frame: np.ndarray) -> DetectionResult:
        res, self._last_yolo_result = self.detect_frame(
            frame,
            class_ids=self.webcam_ids,
            conf_thresh=0.25,
            ppe_conf_thresh=0.25,
            helmet_conf_thresh=0.15,
            imgsz=416,
        )
        self._last_webcam_res = res
        return res

    def detect_webcam_frame_skipped(
        self,
        frame: np.ndarray,
        frame_idx: int,
        skip_interval: int = 2
    ) -> DetectionResult:
        """
        Modo Turbo: ejecuta inferencia cada `skip_interval` frames.
        En frames intermedios redibuja la última detección sobre el frame actual
        (cajas stale semi-visibles) para mantener feedback continuo mientras se
        ahorra GPU. Corrige bug previo que retornaba metadatos viejos desfasados.
        """
        if skip_interval <= 1 or frame_idx % skip_interval == 0 or self._last_yolo_result is None:
            return self.detect_webcam_frame(frame)

        # Redibujar última inferencia sobre frame actual (mantiene cajas visibles)
        annotated, names, heads = draw_detection_boxes(
            frame.copy(),
            self._last_yolo_result,
            self.model,
            scale_x=1.0,
            scale_y=1.0,
            class_ids=self.webcam_ids,
            conf_thresh=0.25,
            ppe_conf_thresh=0.25,
            helmet_conf_thresh=0.15,
        )
        alerts = analyze_compliance(names, heads, check_vest=self.check_vest)
        persons = sum(1 for n in names if n == "Person")
        # Marca visual de interpolación sin ocultar cajas
        cv2.putText(
            annotated, "Turbo: cache", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA
        )
        # Actualizar cache para que siguiente frame stale sea consistente
        cached_res = DetectionResult(annotated, names, int(heads), alerts, persons)
        # No sobre-escribir _last_yolo_result (mantiene origen), solo metadatos visuales
        self._last_webcam_res = cached_res
        return cached_res

    def detect_image_frame(self, frame: np.ndarray) -> DetectionResult:
        result, _ = self.detect_frame(
            frame,
            class_ids=self.video_ids,
            conf_thresh=0.25,
            ppe_conf_thresh=0.25,
            helmet_conf_thresh=0.15,
            imgsz=640,
        )
        draw_status_panel(result.annotated, 0.0, result.names, result.alerts)
        return result

    def detect_batch_frame(self, frame: np.ndarray) -> DetectionResult:
        return self.detect_image_frame(frame)

    def detect_video_frame(
        self,
        frame: np.ndarray,
        fps: float,
        conf_thresh: float = 0.25,
        ppe_conf_thresh: float = 0.25,
        helmet_conf_thresh: float = 0.15,
    ) -> DetectionResult:
        result, _ = self.detect_frame(
            frame,
            class_ids=self.video_ids,
            conf_thresh=conf_thresh,
            ppe_conf_thresh=ppe_conf_thresh,
            helmet_conf_thresh=helmet_conf_thresh,
            imgsz=640,
        )
        draw_status_panel_big(result.annotated, fps, result.names, result.alerts)
        return result
