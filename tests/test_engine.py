from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from engine import EPPDetectionEngine, _project_root, ensure_weights_exist


def test_project_root_resolves():
    root = _project_root()
    assert (root / "weights").exists() or (root / "src").exists()


def test_ensure_weights_exist():
    p = ensure_weights_exist()
    assert p.exists()
    assert p.name == "best.pt"


def test_engine_loads_and_detects_dummy():
    engine = EPPDetectionEngine()
    # Runtime label debe ser string no vacío
    assert isinstance(engine.runtime_label, str) and len(engine.runtime_label) > 0
    # Detect en frame negro dummy (sin crash)
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    res = engine.detect_image_frame(dummy)
    assert hasattr(res, "annotated")
    assert res.annotated.shape[0] > 0
    assert isinstance(res.names, list)
    assert isinstance(res.alerts, list)


def test_engine_turbo_cache_no_stale_boxes():
    engine = EPPDetectionEngine()
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    # Primer frame hace inferencia
    r1 = engine.detect_webcam_frame(dummy)
    # Segundo frame con skip_interval=2 debe reusar metadatos sin dibujar cajas viejas
    r2 = engine.detect_webcam_frame_skipped(dummy, frame_idx=1, skip_interval=2)
    assert r2.names == r1.names
    assert r2.alerts == r1.alerts
    # annotated no debe ser idéntico a r1 si el frame es distinto, pero en este caso dummy igual
    assert r2.annotated.shape == dummy.shape


def test_engine_video_thresholds_parametrizable():
    engine = EPPDetectionEngine()
    dummy = np.zeros((480, 854, 3), dtype=np.uint8)
    # Debe aceptar thresholds custom sin error
    r = engine.detect_video_frame(dummy, fps=30.0, conf_thresh=0.5, ppe_conf_thresh=0.45, helmet_conf_thresh=0.15)
    assert r.annotated.shape[0] > 0


def test_engine_batch_no_crash():
    engine = EPPDetectionEngine()
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    r = engine.detect_batch_frame(dummy)
    assert r.annotated is not None
