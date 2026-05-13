import csv
from pathlib import Path
import math

class LocalStreetViewProvider:
    def __init__(self, data_dir: str = "data/paris_realistic_v1/street_combined"):
        self.data_dir = Path(data_dir)
        self.metadata_file = self.data_dir / "metadata.csv"
        self.points = []
        self._load_metadata()

    def _load_metadata(self):
        if not self.metadata_file.exists():
            return

        try:
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        self.points.append({
                            "lat": float(row["lat"]),
                            "lon": float(row["lon"]),
                            "heading": float(row["heading_deg"]) if "heading_deg" in row and row["heading_deg"] else None,
                            "captured_at": row.get("captured_at"),
                            "path": row.get("path"),
                            "source": row.get("source", "local"),
                        })
                    except ValueError:
                        continue
        except Exception:
            pass

    def _haversine(self, lat1, lon1, lat2, lon2):
        R = 6371.0 # Earth radius in kilometers

        dLat = math.radians(lat2 - lat1)
        dLon = math.radians(lon2 - lon1)

        a = (math.sin(dLat / 2) * math.sin(dLat / 2) +
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
             math.sin(dLon / 2) * math.sin(dLon / 2))
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def find_nearest(self, lat: float, lon: float, max_dist_km: float = 0.1):
        if not self.points:
            return None

        nearest = None
        min_dist = float('inf')

        for p in self.points:
            dist = self._haversine(lat, lon, p["lat"], p["lon"])
            if dist < min_dist and dist <= max_dist_km:
                min_dist = dist
                nearest = p

        if nearest:
            # Reconstruct full path
            # Assuming images are in 'images/' relative to data_dir if 'path' is a filename
            image_url = nearest['path']
            if image_url and not image_url.startswith("http"):
                 image_url = f"/data/paris_realistic_v1/street_combined/images/{image_url}"

            return {
                "url": image_url,
                "heading": nearest["heading"],
                "captured_at": nearest["captured_at"],
                "provider": nearest["source"],
                "distance_km": min_dist
            }

        return None
