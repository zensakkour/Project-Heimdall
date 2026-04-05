from __future__ import annotations

import numpy as np
from PIL import Image

from src.core.geo import retrieval_provider as rp


def _require_torch():
    if rp.torch is None:
        raise RuntimeError("torch_not_available_for_test")
    return rp.torch


def test_clip_embedder_falls_back_to_auto_model_when_clip_loader_fails(monkeypatch) -> None:
    torch = _require_torch()

    class _FakeClipProcessor:
        @classmethod
        def from_pretrained(cls, _model_id):
            raise RuntimeError("clip_processor_fail")

    class _FakeClipModel:
        @classmethod
        def from_pretrained(cls, _model_id):
            raise RuntimeError("clip_model_fail")

    class _FakeAutoProcessor:
        @classmethod
        def from_pretrained(cls, _model_id):
            return cls()

        def __call__(self, *, images, return_tensors):
            assert images is not None
            assert return_tensors == "pt"
            return {"pixel_values": torch.ones((1, 3, 2, 2), dtype=torch.float32)}

    class _FakeAutoModel:
        @classmethod
        def from_pretrained(cls, _model_id):
            return cls()

        def to(self, _device):
            return self

        def eval(self):
            return self

        def get_image_features(self, **_inputs):
            return torch.tensor([[3.0, 4.0]], dtype=torch.float32)

    monkeypatch.setattr(rp, "CLIPProcessor", _FakeClipProcessor)
    monkeypatch.setattr(rp, "CLIPModel", _FakeClipModel)
    monkeypatch.setattr(rp, "AutoImageProcessor", _FakeAutoProcessor)
    monkeypatch.setattr(rp, "AutoProcessor", _FakeAutoProcessor)
    monkeypatch.setattr(rp, "AutoModel", _FakeAutoModel)

    embedder = rp.ClipEmbedder("fake/model", "cpu")
    out = embedder.embed(Image.new("RGB", (4, 4), color=(0, 0, 0)))
    assert out.shape == (1, 2)
    assert np.isclose(float(out[0, 0]), 0.6, atol=1e-4)
    assert np.isclose(float(out[0, 1]), 0.8, atol=1e-4)


def test_clip_embedder_auto_model_without_image_feature_method_uses_output_tensor(monkeypatch) -> None:
    torch = _require_torch()

    class _FakeClipProcessor:
        @classmethod
        def from_pretrained(cls, _model_id):
            raise RuntimeError("clip_processor_fail")

    class _FakeClipModel:
        @classmethod
        def from_pretrained(cls, _model_id):
            raise RuntimeError("clip_model_fail")

    class _FakeAutoProcessor:
        @classmethod
        def from_pretrained(cls, _model_id):
            return cls()

        def __call__(self, *, images, return_tensors):
            assert images is not None
            assert return_tensors == "pt"
            return {"pixel_values": torch.ones((1, 3, 2, 2), dtype=torch.float32)}

    class _FakeOutput:
        def __init__(self):
            self.last_hidden_state = torch.tensor([[[2.0, 0.0], [2.0, 0.0]]], dtype=torch.float32)

    class _FakeAutoModel:
        @classmethod
        def from_pretrained(cls, _model_id):
            return cls()

        def to(self, _device):
            return self

        def eval(self):
            return self

        def __call__(self, **_inputs):
            return _FakeOutput()

    monkeypatch.setattr(rp, "CLIPProcessor", _FakeClipProcessor)
    monkeypatch.setattr(rp, "CLIPModel", _FakeClipModel)
    monkeypatch.setattr(rp, "AutoImageProcessor", _FakeAutoProcessor)
    monkeypatch.setattr(rp, "AutoProcessor", _FakeAutoProcessor)
    monkeypatch.setattr(rp, "AutoModel", _FakeAutoModel)

    embedder = rp.ClipEmbedder("fake/model", "cpu")
    out = embedder.embed(Image.new("RGB", (4, 4), color=(0, 0, 0)))
    assert out.shape == (1, 2)
    assert np.isclose(float(out[0, 0]), 1.0, atol=1e-6)
    assert np.isclose(float(out[0, 1]), 0.0, atol=1e-6)
