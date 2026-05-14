import pytest
from src.core.geo.street_view import LocalStreetViewProvider
from pathlib import Path
import math

def test_local_street_view_provider_no_data(tmp_path):
    provider = LocalStreetViewProvider(data_dir=str(tmp_path))
    res = provider.find_nearest(48.8, 2.3)
    assert res is None

def test_local_street_view_provider_with_data(tmp_path):
    # Setup dummy data
    metadata_file = tmp_path / "metadata.csv"
    with open(metadata_file, "w") as f:
        f.write("image_id,path,lat,lon,heading_deg,captured_at,camera_type,width,height,quality_score,sequence,source,license_info\n")
        f.write("1,test_img.jpg,48.8566,2.3522,90,2024-01-01,test,800,600,1.0,1,test_provider,test\n")

    provider = LocalStreetViewProvider(data_dir=str(tmp_path))

    # Very close coordinate
    res = provider.find_nearest(48.8566, 2.3522)
    assert res is not None
    assert res["url"] == "/api/operator/street_view/image/1"
    assert res["image_id"] == "1"
    assert res["heading"] == 90.0
    assert res["provider"] == "test_provider"
    assert math.isclose(res["distance_km"], 0.0, abs_tol=0.01)

    # Far coordinate
    res2 = provider.find_nearest(40.0, -74.0, max_dist_km=0.1)
    assert res2 is None
