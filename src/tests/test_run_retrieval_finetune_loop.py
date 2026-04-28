from __future__ import annotations

import json
import tempfile
from pathlib import Path

from src.tools import run_retrieval_finetune_loop as loop


def test_is_better_prefers_higher_within_2km_and_lower_mean_on_tie() -> None:
    incumbent = {"within_2km_pct": 30.0, "mean_km": 15.0}
    better = {"within_2km_pct": 31.0, "mean_km": 20.0}
    tied_better_mean = {"within_2km_pct": 30.0, "mean_km": 14.0}
    worse = {"within_2km_pct": 29.0, "mean_km": 10.0}

    assert loop._is_better(better, incumbent, "within_2km_pct") is True
    assert loop._is_better(tied_better_mean, incumbent, "within_2km_pct") is True
    assert loop._is_better(worse, incumbent, "within_2km_pct") is False


def test_is_better_prefers_lower_mean_km_and_higher_within_2km_on_tie() -> None:
    incumbent = {"mean_km": 15.0, "within_2km_pct": 30.0}
    better = {"mean_km": 14.0, "within_2km_pct": 20.0}
    tied_better_w2 = {"mean_km": 15.0, "within_2km_pct": 31.0}
    worse = {"mean_km": 16.0, "within_2km_pct": 50.0}

    assert loop._is_better(better, incumbent, "mean_km") is True
    assert loop._is_better(tied_better_w2, incumbent, "mean_km") is True
    assert loop._is_better(worse, incumbent, "mean_km") is False


def test_build_aux_fused_config_appends_tuned_sources_without_inheriting_projection() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        bootstrap = root / "bootstrap.json"
        tuned_index = root / "tuned.npz"
        tuned_dba = root / "tuned_dba.npz"
        output = root / "aux.json"
        bootstrap.write_text(
            json.dumps(
                {
                    "geolocator": {
                        "retrieval_index_path": "data/base_primary.npz",
                        "retrieval_index_paths": ["data/base_dba.npz"],
                        "retrieval_index_weights": [1.0, 1.0],
                        "retrieval_model_id": "openai/clip-vit-large-patch14",
                        "retrieval_projection_path": "runs/base_proj.npz",
                    }
                }
            ),
            encoding="utf-8",
        )
        tuned_index.write_bytes(b"index")
        tuned_dba.write_bytes(b"dba")

        payload = loop._build_aux_fused_config(
            bootstrap_config_path=bootstrap,
            output_config_path=output,
            tuned_model_ref="runs/model_dir",
            tuned_index_path=tuned_index,
            tuned_dba_index_path=tuned_dba,
            aux_index_weight=0.35,
            aux_dba_weight=0.2,
        )

        geo = payload["geolocator"]
        assert geo["retrieval_index_paths"] == [
            "data/base_dba.npz",
            str(tuned_index),
            str(tuned_dba),
        ]
        assert geo["retrieval_index_weights"] == [1.0, 1.0, 0.35, 0.2]
        assert geo["retrieval_index_model_ids"] == [
            "openai/clip-vit-large-patch14",
            "runs/model_dir",
            "runs/model_dir",
        ]
        assert geo["retrieval_index_projection_paths"] == [
            "runs/base_proj.npz",
            None,
            None,
        ]
        assert geo["retrieval_source_fusion_mode"] == "rrf"
        assert output.exists()
