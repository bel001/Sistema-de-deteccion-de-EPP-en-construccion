from __future__ import annotations

import os
import zipfile
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


def ensure_weights_exist(weights_dir: str | Path = "weights") -> Path:
    """Garantiza que el archivo de pesos best.pt exista, copiándolo desde best.pt.zip si es necesario."""
    wdir = Path(weights_dir)
    target_pt = wdir / "best.pt"
    target_zip = wdir / "best.pt.zip"

    if target_pt.exists():
        return target_pt

    if target_zip.exists():
        import shutil
        shutil.copyfile(target_zip, target_pt)
        return target_pt

    raise FileNotFoundError(
        f"No se encontró el archivo de modelo {target_pt}. Por favor asegúrate de tener best.pt en {wdir}."
    )


@dataclass
class DetectionResult:
    annotated: np.ndarray
    names: list[str]
    unprotected_heads: int
    alerts: list[str]
    persons: int = 0


class EPPDetectionEngine:
    """Motor unificado de inferencia de EPP con aceleración GPU (CUDA/Half-precision)."""

    def __init__(self, model_path: str | Path = "weights/best.pt") -> None:
        path = Path(model_path)
        if not path.exists():
            path = ensure_weights_exist(path.parent)

        self.model_path = path
        self.uses_cuda = torch.cuda.is_available()
        self.predict_device = 0 if self.uses_cuda else "cpu"
        self.runtime_label = self._runtime_label()

        if self.uses_cuda:
            torch.backends.cudnn.benchmark = True

        self.model = YOLO(str(self.model_path))
        self.webcam_ids = filter_supported_class_ids(self.model, WEBCAM_CLASS_IDS)
        self.video_ids = filter_supported_class_ids(self.model, VIDEO_DISPLAY_CLASS_IDS)
        self.check_vest = model_supports_class_id(self.model, SAFETY_VEST_CLASS_ID)

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
    ) -> DetectionResult:
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
        return DetectionResult(annotated, names, int(heads), alerts, persons)

    def detect_webcam_frame(self, frame: np.ndarray) -> DetectionResult:
        return self.detect_frame(
            frame,
            class_ids=self.webcam_ids,
            conf_thresh=0.25,
            ppe_conf_thresh=0.25,
            helmet_conf_thresh=0.15,
            imgsz=416,
        )

    def detect_image_frame(self, frame: np.ndarray) -> DetectionResult:
        result = self.detect_frame(
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

    def detect_video_frame(self, frame: np.ndarray, fps: float) -> DetectionResult:
        result = self.detect_frame(
            frame,
            class_ids=self.video_ids,
            conf_thresh=0.50,
            ppe_conf_thresh=0.45,
            helmet_conf_thresh=0.15,
            imgsz=640,
        )
        draw_status_panel_big(result.annotated, fps, result.names, result.alerts)
        return result
