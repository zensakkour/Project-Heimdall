from __future__ import annotations

import json
import tempfile
from pathlib import Path

from src.tools.run_geo_eval import build_retrieval_provider, predict_latlon_retrieval
from src.core.logic.config import load_config
from src.core.logic.types import GeoCandidate


def test_build_retrieval_provider_none_without_index() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "cfg.json"
        path.write_text(json.dumps({"geolocator": {"retrieval_index_path": None}}), encoding="utf-8")
        cfg = load_config(str(path))
        provider = build_retrieval_provider(cfg)
        assert provider is None


def test_build_retrieval_provider_with_multi_index_paths() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "cfg.json"
        path.write_text(
            json.dumps(
                {
                    "geolocator": {
                        "retrieval_index_path": None,
                        "retrieval_index_paths": ["data/geo_index/a.npz", "data/geo_index/b.npz"],
                        "retrieval_index_weights": [1.0, 0.7],
                    }
                }
            ),
            encoding="utf-8",
        )
        cfg = load_config(str(path))
        provider = build_retrieval_provider(cfg)
        assert provider is not None


def test_predict_latlon_retrieval_handles_missing_provider() -> None:
    pred, score, err = predict_latlon_retrieval("missing.jpg", None)
    assert pred is None
    assert score is None
    assert err == "index_not_configured"


def test_predict_latlon_retrieval_uses_provider_output() -> None:
    class StubProvider:
        def __init__(self) -> None:
            self.last_error = None

        def candidates(self, _image_path: str):
            return [GeoCandidate(latitude=48.8566, longitude=2.3522, retrieval_score=0.91, match_id="x")]

    provider = StubProvider()
    pred, score, err = predict_latlon_retrieval("x.jpg", provider)  # type: ignore[arg-type]
    assert pred == (48.8566, 2.3522)
    assert score == 0.91
    assert err is None
