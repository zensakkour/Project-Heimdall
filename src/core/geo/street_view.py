import csv
from pathlib import Path
import math
from typing import Optional


class LocalStreetViewProvider:
    def __init__(self, data_dir: str = "data/paris_realistic_v1/street_combined"):
        self.data_dir = Path(data_dir)
        self.metadata_file = self.data_dir / "metadata.csv"
        self.points: list[dict] = []
        self._by_id: dict[str, dict] = {}
        self._by_sequence: dict[str, list[dict]] = {}
        self._load_metadata()

    def _load_metadata(self) -> None:
        if not self.metadata_file.exists():
            return

        try:
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        point = {
                            "image_id": row.get("image_id", ""),
                            "lat": float(row["lat"]),
                            "lon": float(row["lon"]),
                            "heading": float(row["heading_deg"]) if row.get("heading_deg") else None,
                            "captured_at": row.get("captured_at"),
                            "path": row.get("path", ""),
                            "source": row.get("source", "local"),
                            "sequence": row.get("sequence", ""),
                        }
                        self.points.append(point)
                    except (ValueError, KeyError):
                        continue
        except Exception:
            pass

        # Build fast-lookup indexes
        self._by_id = {p["image_id"]: p for p in self.points if p.get("image_id")}
        for p in self.points:
            seq = p.get("sequence") or ""
            self._by_sequence.setdefault(seq, []).append(p)
        for seq in self._by_sequence:
            self._by_sequence[seq].sort(key=lambda p: p.get("captured_at") or "")

    def _haversine(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        R = 6371.0
        dLat = math.radians(lat2 - lat1)
        dLon = math.radians(lon2 - lon1)
        a = (math.sin(dLat / 2) ** 2 +
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
             math.sin(dLon / 2) ** 2)
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def _heading_diff(self, h1: Optional[float], h2: Optional[float]) -> float:
        """Angular difference between two headings (0-180)."""
        if h1 is None or h2 is None:
            return 180.0
        diff = abs(h1 - h2) % 360
        return diff if diff <= 180 else 360 - diff

    def _point_to_response(self, point: dict, dist_km: float = 0.0) -> dict:
        return {
            "image_id": point["image_id"],
            "url": f"/api/operator/street_view/image/{point['image_id']}",
            "heading": point["heading"],
            "captured_at": point.get("captured_at"),
            "provider": point.get("source", "local"),
            "distance_km": dist_km,
            "lat": point["lat"],
            "lon": point["lon"],
            "sequence": point.get("sequence", ""),
        }

    def find_nearest(self, lat: float, lon: float, max_dist_km: float = 0.1) -> Optional[dict]:
        if not self.points:
            return None

        nearest = None
        min_dist = float("inf")

        for p in self.points:
            dist = self._haversine(lat, lon, p["lat"], p["lon"])
            if dist < min_dist and dist <= max_dist_km:
                min_dist = dist
                nearest = p

        if nearest:
            return self._point_to_response(nearest, min_dist)
        return None

    def find_nearest_by_heading(
        self,
        lat: float,
        lon: float,
        prefer_heading: float,
        max_dist_km: float = 0.15,
    ) -> Optional[dict]:
        """Find the nearest image that best matches the preferred heading."""
        if not self.points:
            return None

        candidates = []
        for p in self.points:
            dist = self._haversine(lat, lon, p["lat"], p["lon"])
            if dist <= max_dist_km:
                candidates.append((dist, p))

        if not candidates:
            return None

        # Sort by heading similarity first, then distance as tiebreaker
        candidates.sort(key=lambda t: (self._heading_diff(t[1]["heading"], prefer_heading), t[0]))
        dist, point = candidates[0]
        return self._point_to_response(point, dist)

    def get_sequence_neighbors(self, image_id: str) -> dict:
        """Return prev/next images in the same sequence, plus position metadata."""
        point = self._by_id.get(image_id)
        if not point:
            return {"prev": None, "next": None, "position": 1, "total": 1}

        seq = point.get("sequence") or ""
        seq_list = self._by_sequence.get(seq, [point])
        total = len(seq_list)

        try:
            idx = next(i for i, p in enumerate(seq_list) if p["image_id"] == image_id)
        except StopIteration:
            return {"prev": None, "next": None, "position": 1, "total": total}

        prev_point = seq_list[idx - 1] if idx > 0 else None
        next_point = seq_list[idx + 1] if idx < total - 1 else None

        return {
            "prev": self._point_to_response(prev_point) if prev_point else None,
            "next": self._point_to_response(next_point) if next_point else None,
            "position": idx + 1,
            "total": total,
        }
