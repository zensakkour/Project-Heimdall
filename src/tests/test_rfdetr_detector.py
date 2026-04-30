from __future__ import annotations

import sys
import types
import builtins

import numpy as np

from src.core.detection.factory import create_detector
from src.core.detection.rfdetr_detector import RFDetrDetector
from src.core.logic.config import DetectorConfig


class _FakeDetections:
    xyxy = np.asarray([[1.0, 2.0, 11.0, 22.0], [3.0, 4.0, 5.0, 6.0]], dtype=np.float32)
    confidence = np.asarray([0.91, 0.10], dtype=np.float32)
    class_id = np.asarray([2, 0], dtype=np.int64)


class _FakeRFDETRMedium:
    def predict(self, image_path: str, threshold: float):
        assert image_path == "street.jpg"
        assert threshold == 0.5
        return _FakeDetections()


def _install_fake_rfdetr(monkeypatch):
    module = types.ModuleType("rfdetr")
    module.RFDETRMedium = _FakeRFDETRMedium
    assets = types.ModuleType("rfdetr.assets")
    coco = types.ModuleType("rfdetr.assets.coco_classes")
    coco.COCO_CLASSES = {2: "car", 0: "person"}
    monkeypatch.setitem(sys.modules, "rfdetr", module)
    monkeypatch.setitem(sys.modules, "rfdetr.assets", assets)
    monkeypatch.setitem(sys.modules, "rfdetr.assets.coco_classes", coco)


def test_rfdetr_detector_converts_boxes_to_detections(monkeypatch) -> None:
    _install_fake_rfdetr(monkeypatch)

    detector = RFDetrDetector(model_size="medium", min_confidence=0.5, min_area_px=1.0)
    detections = detector.predict("street.jpg")

    assert len(detections) == 1
    det = detections[0]
    assert det.label == "car"
    assert round(det.confidence, 2) == 0.91
    xs = sorted(point[0] for point in det.obb)
    ys = sorted(point[1] for point in det.obb)
    assert xs == [1.0, 1.0, 11.0, 11.0]
    assert ys == [2.0, 2.0, 22.0, 22.0]


def test_factory_uses_rfdetr_backend(monkeypatch) -> None:
    _install_fake_rfdetr(monkeypatch)

    created = create_detector(
        DetectorConfig(
            backend="rfdetr",
            weights_path="ignored-by-rfdetr",
            min_confidence=0.5,
            rfdetr_model_size="medium",
        )
    )

    assert created is not None
    detector, backend = created
    assert isinstance(detector, RFDetrDetector)
    assert backend == "rfdetr"


def test_factory_falls_back_when_rfdetr_missing(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "rfdetr", raising=False)
    real_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "rfdetr":
            raise ImportError("mocked missing rfdetr")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    created = create_detector(
        DetectorConfig(
            backend="rfdetr",
            min_confidence=0.5,
            use_sidecar=True,
            use_classic=True,
        )
    )

    assert created is not None
    detector, backend = created
    assert detector.__class__.__name__ == "SidecarDetector"
    assert backend == "sidecar"
