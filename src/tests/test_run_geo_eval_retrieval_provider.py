from __future__ import annotations

import json
import tempfile
from pathlib import Path

from src.tools.run_geo_eval import build_pipeline, build_retrieval_provider, load_metadata_records, predict_latlon_retrieval
from src.tools.run_geo_eval import capture_time_from_record, normalize_metadata_records
from src.core.logic.config import DetectorConfig, FusionConfig, GeoConfig, HeimdallConfig, ScoreConfig, VerificationConfig
from src.core.logic.config import load_config
from src.core.logic.types import GeoCandidate


class _TinyFrame:
    def __init__(self, records: list[dict]) -> None:
        self._records = records
        self.columns = list(records[0].keys()) if records else []

    def to_dict(self, orient: str):
        assert orient == "records"
        return list(self._records)


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


def test_normalize_metadata_records_accepts_realistic_pair_columns() -> None:
    df = _TinyFrame(
        [
            {
                "street_path": "images/query.jpg",
                "lat": 48.8566,
                "lon": 2.3522,
                "captured_at": "2026-06-01T12:00:00Z",
            }
        ]
    )

    records = normalize_metadata_records(df)

    assert records[0]["path"] == "images/query.jpg"
    assert records[0]["latitude"] == 48.8566
    assert records[0]["longitude"] == 2.3522
    assert capture_time_from_record(records[0]) is not None


def test_capture_time_from_record_accepts_mapillary_epoch_millis() -> None:
    captured = capture_time_from_record({"captured_at": "1518353752470"})

    assert captured is not None
    assert captured.year == 2018


def test_load_metadata_records_uses_csv_without_pandas_dependency() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "metadata.csv"
        path.write_text(
            "street_path,lat,lon,captured_at\n"
            "images/query.jpg,48.8566,2.3522,2026-06-01T12:00:00Z\n",
            encoding="utf-8",
        )

        records = load_metadata_records(path)

        assert records[0]["path"] == "images/query.jpg"
        assert float(records[0]["latitude"]) == 48.8566
        assert float(records[0]["longitude"]) == 2.3522


def test_load_metadata_records_enriches_pair_rows_from_query_metadata() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        pairs = root / "test_pairs.csv"
        query = root / "query_metadata.csv"
        pairs.write_text(
            "street_id,street_path,lat,lon\n"
            "812627149621312,images/mapillary__812627149621312.jpg,48.8566,2.3522\n",
            encoding="utf-8",
        )
        query.write_text(
            "image_id,path,lat,lon,captured_at,source\n"
            "812627149621312,images/mapillary__812627149621312.jpg,48.8566,2.3522,1518353752470,mapillary\n",
            encoding="utf-8",
        )

        records = load_metadata_records(pairs, query_metadata_path=query)

        assert str(records[0]["captured_at"]) == "1518353752470"
        assert records[0]["source"] == "mapillary"
        assert records[0]["_query_metadata_enriched"] is True
        assert capture_time_from_record(records[0]) is not None


def test_load_metadata_records_auto_discovers_images_dir_metadata() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        images_dir = root / "street_combined"
        images_dir.mkdir()
        pairs = root / "splits" / "test_pairs.csv"
        pairs.parent.mkdir()
        metadata = images_dir / "metadata.csv"
        pairs.write_text(
            "street_id,street_path,lat,lon\n"
            "812627149621312,images/mapillary__812627149621312.jpg,48.8566,2.3522\n",
            encoding="utf-8",
        )
        metadata.write_text(
            "image_id,path,lat,lon,captured_at\n"
            "812627149621312,images/mapillary__812627149621312.jpg,48.8566,2.3522,2026-06-01T12:00:00Z\n",
            encoding="utf-8",
        )

        records = load_metadata_records(pairs, images_dir=images_dir)

        assert records[0]["captured_at"] == "2026-06-01T12:00:00Z"
        assert capture_time_from_record(records[0]) is not None


def test_build_pipeline_unpacks_detector_factory_tuple(monkeypatch) -> None:
    class StubDetector:
        def predict(self, _image_path: str):
            return []

    import src.core.detection.factory as factory

    monkeypatch.setattr(factory, "create_detector", lambda _cfg: (StubDetector(), "stub"))
    cfg = HeimdallConfig(
        detector=DetectorConfig(),
        geolocator=GeoConfig(retrieval_index_path=None),
        fusion=FusionConfig(),
        score=ScoreConfig(),
        verification=VerificationConfig(),
    )

    pipeline = build_pipeline(cfg)

    assert isinstance(pipeline.detector, StubDetector)
    assert pipeline.detector_backend == "stub"
