"""
Detection post-processing sanity checks.
"""
from __future__ import annotations

from src.core.logic.postprocess import filter_detections
from src.core.logic.types import Detection


def _box(x1: float, y1: float, x2: float, y2: float):
    return ((x1, y1), (x2, y1), (x2, y2), (x1, y2))


def test_nms_reduces_overlaps() -> None:
    dets = [
        Detection(label="a", confidence=0.9, obb=_box(0, 0, 10, 10)),
        Detection(label="a", confidence=0.8, obb=_box(1, 1, 9, 9)),
    ]
    kept = filter_detections(dets, min_confidence=0.0, nms_iou=0.1, max_detections=10)
    assert len(kept) == 1
    assert kept[0].label == "a"


def test_normalizes_obb_order() -> None:
    dets = [
        Detection(label="a", confidence=0.9, obb=((10, 10), (0, 10), (0, 0), (10, 0))),
    ]
    kept = filter_detections(dets, min_confidence=0.0, nms_iou=0.0, max_detections=10)
    assert len(kept) == 1
    first = kept[0].obb[0]
    assert first == (0.0, 0.0)


def test_nms_keeps_different_labels_when_class_aware() -> None:
    dets = [
        Detection(label="car", confidence=0.9, obb=_box(0, 0, 10, 10)),
        Detection(label="truck", confidence=0.85, obb=_box(1, 1, 9, 9)),
    ]
    kept = filter_detections(
        dets,
        min_confidence=0.0,
        nms_iou=0.1,
        max_detections=10,
        class_agnostic_nms=False,
    )
    assert len(kept) == 2


def test_nms_suppresses_different_labels_when_class_agnostic() -> None:
    dets = [
        Detection(label="car", confidence=0.9, obb=_box(0, 0, 10, 10)),
        Detection(label="truck", confidence=0.85, obb=_box(1, 1, 9, 9)),
    ]
    kept = filter_detections(
        dets,
        min_confidence=0.0,
        nms_iou=0.1,
        max_detections=10,
        class_agnostic_nms=True,
    )
    assert len(kept) == 1
    assert kept[0].label == "car"


def test_rejects_tiny_detections_by_area() -> None:
    dets = [
        Detection(label="a", confidence=0.9, obb=_box(0, 0, 2, 2)),
        Detection(label="b", confidence=0.9, obb=_box(0, 0, 10, 10)),
    ]
    kept = filter_detections(
        dets,
        min_confidence=0.0,
        nms_iou=0.0,
        max_detections=10,
        min_area_px=10.0,
    )
    assert len(kept) == 1
    assert kept[0].label == "b"


