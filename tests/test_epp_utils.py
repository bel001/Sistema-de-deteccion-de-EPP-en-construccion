from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Permitir import desde src/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from epp_utils import (
    ACTUAL_CLASS_MAP,
    CLASS_COLORS,
    DISPLAY_NAMES,
    IMAGE_EXTENSIONS,
    analyze_compliance,
    box_area,
    filter_supported_class_ids,
    head_has_helmet,
    intersection_area,
    model_supports_class_id,
    passes_class_filters,
    vest_is_valid_inside_person,
    vest_overlaps_head,
)


def test_actual_class_map_completeness():
    assert 9 in ACTUAL_CLASS_MAP
    assert ACTUAL_CLASS_MAP[9] == "Safety-vest"
    assert len(ACTUAL_CLASS_MAP) == 10
    assert ACTUAL_CLASS_MAP[0] == "Person"


def test_colors_no_duplicate_head_glasses():
    assert CLASS_COLORS["Head"] != CLASS_COLORS["Glasses"], "Head y Glasses no deben compartir color"
    assert CLASS_COLORS["Safety-vest"] != CLASS_COLORS["Head"]


def test_display_names_covers_map():
    for name in ACTUAL_CLASS_MAP.values():
        assert name in DISPLAY_NAMES, f"{name} falta en DISPLAY_NAMES"


def test_box_area():
    assert box_area((0, 0, 10, 10)) == 100
    assert box_area((0, 0, 0, 10)) == 0
    assert box_area((5, 5, 5, 5)) == 0


def test_intersection():
    assert intersection_area((0, 0, 10, 10), (5, 5, 15, 15)) == 25
    assert intersection_area((0, 0, 10, 10), (20, 20, 30, 30)) == 0


def test_head_has_helmet_overlap():
    assert head_has_helmet((0, 0, 10, 10), [(0, 0, 10, 10)]) is True
    assert head_has_helmet((0, 0, 10, 10), []) is False
    assert head_has_helmet((0, 100, 10, 110), [(0, 0, 10, 10)]) is False


def test_head_has_helmet_proximity():
    # Casco justo encima de cabeza (solapamiento vertical <0.5*head_h)
    assert head_has_helmet((0, 20, 10, 30), [(0, 5, 10, 16)]) is True


def test_vest_overlaps_head():
    assert vest_overlaps_head((0, 0, 50, 50), [(0, 0, 20, 20)]) is True
    assert vest_overlaps_head((100, 100, 150, 150), [(0, 0, 20, 20)]) is False


def test_vest_valid_inside_person():
    assert vest_is_valid_inside_person((30, 100, 80, 200), [(0, 0, 100, 300)], []) is True
    # Sin persona -> ahora True (fallback restaurado para no perder chaleco si persona filtrada)
    assert vest_is_valid_inside_person((30, 100, 80, 200), [], []) is True
    # Solapado con cabeza -> False
    assert vest_is_valid_inside_person((0, 0, 50, 50), [(0, 0, 100, 300)], [(0, 0, 50, 50)]) is False


def test_passes_filters_person():
    frame_area = 640 * 480
    assert passes_class_filters("Person", 0.9, 0, 0, 10, 10, frame_area, []) is False  # muy pequeño
    assert passes_class_filters("Person", 0.9, 0, 0, 300, 300, frame_area, []) is True


def test_passes_filters_conf():
    frame_area = 640 * 480
    # Conf bajo rechaza
    assert passes_class_filters("Helmet", 0.05, 10, 10, 100, 100, frame_area, [], [], 0.25, 0.25, 0.15) is False
    assert passes_class_filters("Helmet", 0.2, 10, 10, 100, 100, frame_area, [], [], 0.25, 0.25, 0.15) is True


def test_analyze_compliance_vest():
    # Sin vest, con check_vest=True -> alerta
    alerts = analyze_compliance(["Person"], 0, check_vest=True)
    assert any("chaleco" in a for a in alerts)
    # Sin check_vest -> no alerta
    alerts2 = analyze_compliance(["Person"], 0, check_vest=False)
    assert not any("chaleco" in a for a in alerts2)


def test_analyze_compliance_hands():
    assert "Posible falta de guantes" in analyze_compliance(["Hands"], 0)
    assert "Posible falta de guantes" not in analyze_compliance(["Hands", "Gloves"], 0)


def test_model_supports():
    class FakeModel:
        names = {0: "Person", 9: "Safety-vest"}

    assert model_supports_class_id(FakeModel(), 9) is True
    assert model_supports_class_id(FakeModel(), 5) is False
    assert filter_supported_class_ids(FakeModel(), [0, 5, 9]) == [0, 9]


def test_image_extensions():
    assert ".jpg" in IMAGE_EXTENSIONS
    assert ".mp4" not in IMAGE_EXTENSIONS
