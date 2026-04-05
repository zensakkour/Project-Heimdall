from __future__ import annotations

from src.tools.download_open_geo import _parse_commons_geosearch_payload, _safe_ext


def test_parse_commons_geosearch_payload_extracts_items() -> None:
    payload = {
        "query": {
            "pages": [
                {
                    "title": "File:Example.jpg",
                    "coordinates": [{"lat": 48.85, "lon": 2.35}],
                    "imageinfo": [
                        {
                            "url": "https://upload.wikimedia.org/wikipedia/commons/a/a0/Example.jpg",
                            "descriptionurl": "https://commons.wikimedia.org/wiki/File:Example.jpg",
                            "extmetadata": {
                                "LicenseShortName": {"value": "CC BY-SA 4.0"},
                                "LicenseUrl": {"value": "https://creativecommons.org/licenses/by-sa/4.0/"},
                            },
                        }
                    ],
                }
            ]
        }
    }
    rows = _parse_commons_geosearch_payload(payload)
    assert len(rows) == 1
    assert rows[0]["title"] == "Example.jpg"
    assert rows[0]["latitude"] == 48.85
    assert rows[0]["longitude"] == 2.35
    assert rows[0]["license"] == "CC BY-SA 4.0"


def test_parse_commons_geosearch_payload_skips_invalid_rows() -> None:
    payload = {
        "query": {
            "pages": [
                {"title": "Category:NotAFile"},
                {"title": "File:NoCoords.jpg", "imageinfo": [{"url": "https://x"}]},
                {"title": "File:NoInfo.jpg", "coordinates": [{"lat": 1.0, "lon": 2.0}]},
            ]
        }
    }
    rows = _parse_commons_geosearch_payload(payload)
    assert rows == []


def test_safe_ext_defaults_to_jpg() -> None:
    assert _safe_ext("https://example.com/a.png") == ".png"
    assert _safe_ext("https://example.com/no_ext") == ".jpg"
