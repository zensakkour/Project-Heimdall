"""
GeoCLIP/GeoFT candidate provider with optional real model integration.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from src.core.logic.image_meta import extract_gps
from src.core.logic.types import GeoCandidate

try:
    from geoclip.model.GeoCLIP import GeoCLIP  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    GeoCLIP = None

try:
    from huggingface_hub import hf_hub_download  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    hf_hub_download = None

try:
    from safetensors.torch import load_file as safe_load  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    safe_load = None


def _extract_tensor(output):
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "image_embeds") and output.image_embeds is not None:
        return output.image_embeds
    if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        return output.last_hidden_state.mean(dim=1)
    if isinstance(output, dict):
        if "pooler_output" in output and output["pooler_output"] is not None:
            return output["pooler_output"]
        if "image_embeds" in output and output["image_embeds"] is not None:
            return output["image_embeds"]
        if "last_hidden_state" in output and output["last_hidden_state"] is not None:
            return output["last_hidden_state"].mean(dim=1)
    return output


def _is_geospot_model(model_id: Optional[str], model_path: Optional[str], model_cache_dir: Optional[str]) -> bool:
    for item in (model_id, model_path, model_cache_dir):
        if item and "geospot" in item.lower():
            return True
    return False


def _build_geospot_model(model_path: str):
    import torch  # type: ignore
    from transformers import Siglip2ImageProcessor, Siglip2VisionConfig, Siglip2VisionModel  # type: ignore
    from geoclip.model.GeoCLIP import GeoCLIP  # type: ignore

    class Siglip2Inputs:
        def __init__(self, data):
            self.data = data

        def to(self, device):
            for key, value in self.data.items():
                self.data[key] = value.to(device)
            return self

        def __getitem__(self, key):
            return self.data[key]

    class Siglip2ImageEncoder(torch.nn.Module):
        def __init__(self, model_id: str = "google/siglip2-so400m-patch16-512") -> None:
            super().__init__()
            config = Siglip2VisionConfig.from_pretrained(model_id)
            self.backbone = Siglip2VisionModel(config)
            self.image_processor = Siglip2ImageProcessor.from_pretrained(model_id)
            hidden = config.hidden_size
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(hidden, hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden, 512),
            )

        def preprocess_image(self, image):
            inputs = self.image_processor(images=image, return_tensors="pt")
            return Siglip2Inputs(inputs)

        def forward(self, x):
            if isinstance(x, Siglip2Inputs):
                pixel_values = x["pixel_values"]
                pixel_attention_mask = x["pixel_attention_mask"]
                spatial_shapes = x["spatial_shapes"]
            elif isinstance(x, dict):
                pixel_values = x["pixel_values"]
                pixel_attention_mask = x["pixel_attention_mask"]
                spatial_shapes = x["spatial_shapes"]
            else:
                pixel_values = x
                pixel_attention_mask = None
                spatial_shapes = None
            if pixel_attention_mask is None or spatial_shapes is None:
                outputs = self.backbone(pixel_values=pixel_values)
            else:
                outputs = self.backbone(
                    pixel_values=pixel_values,
                    pixel_attention_mask=pixel_attention_mask,
                    spatial_shapes=spatial_shapes,
                )
            feats = _extract_tensor(outputs)
            return self.mlp(feats)

    model = GeoCLIP(from_pretrained=False)
    model.image_encoder = Siglip2ImageEncoder()
    model.eval()
    return model


def _patch_geoclip_output(model) -> None:
    clip = getattr(getattr(model, "image_encoder", None), "CLIP", None)
    if clip is None:
        return
    original = clip.get_image_features

    def _wrapped(*args, **kwargs):
        out = original(*args, **kwargs)
        try:
            return _extract_tensor(out)
        except Exception:
            return out

    clip.get_image_features = _wrapped


class GeoCLIPProvider:
    def __init__(
        self,
        model_path: str | None = None,
        model_id: str | None = None,
        model_cache_dir: str | None = None,
        encoder_name: str | None = None,
        top_n: int = 5,
        use_sidecar: bool = True,
        use_exif: bool = True,
        score_scale: float = 1.0,
    ) -> None:
        self.model_path = model_path
        self.model_id = model_id
        self.model_cache_dir = model_cache_dir
        self.encoder_name = encoder_name
        self.top_n = top_n
        self.use_sidecar = use_sidecar
        self.use_exif = use_exif
        self.score_scale = max(0.0, float(score_scale))
        self._model = None
        self._model_failed = False
        self.last_error: Optional[str] = None

    def candidates(self, image_path: str) -> List[GeoCandidate]:
        results: List[GeoCandidate] = []
        if self.use_sidecar:
            results.extend(_load_sidecar_candidates(image_path))

        if self.use_exif:
            gps = extract_gps(image_path)
            if gps is not None:
                lat, lon = gps
                if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
                    results.append(
                        GeoCandidate(
                            latitude=lat,
                            longitude=lon,
                            retrieval_score=0.25,
                            match_id="exif:gps",
                        )
                    )
        if not results:
            model = self._ensure_model()
            if model is not None:
                try:
                    top_gps, top_probs = model.predict(image_path, top_k=self.top_n)
                    for (lat, lon), prob in zip(top_gps, top_probs):
                        score = float(prob) * self.score_scale
                        results.append(
                            GeoCandidate(
                                latitude=float(lat),
                                longitude=float(lon),
                                retrieval_score=score,
                                match_id="geoclip",
                            )
                        )
                except Exception as exc:
                    self.last_error = str(exc)
                    self._model_failed = True
            else:
                self.last_error = "model_unavailable"
        else:
            self.last_error = None

        return results[: self.top_n]

    def _ensure_model(self):
        if self._model_failed:
            return None
        if self._model is not None:
            return self._model
        if GeoCLIP is None:
            self.last_error = "geoclip_import_failed"
            self._model_failed = True
            return None

        model_path = self._resolve_model_path()
        use_geospot = _is_geospot_model(self.model_id, self.model_path, self.model_cache_dir)

        if model_path is None and self.model_id is None and self.model_path is None:
            model = GeoCLIP(from_pretrained=True)
            _patch_geoclip_output(model)
            model.eval()
            self._model = model
            self.last_error = None
            return model

        if model_path is None:
            self.last_error = "model_path_not_found"
            self._model_failed = True
            return None

        try:
            import torch  # type: ignore
        except Exception:
            self.last_error = "torch_import_failed"
            self._model_failed = True
            return None

        if use_geospot:
            try:
                model = _build_geospot_model(model_path)
            except Exception as exc:
                self.last_error = str(exc)
                self._model_failed = True
                return None
        else:
            if self.encoder_name:
                try:
                    model = GeoCLIP(from_pretrained=False, encoder_name=self.encoder_name)
                except TypeError:
                    model = GeoCLIP(from_pretrained=False)
            else:
                model = GeoCLIP(from_pretrained=False)
            _patch_geoclip_output(model)
        try:
            if safe_load is not None:
                state_dict = safe_load(model_path)
            else:
                state_dict = torch.load(model_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            self._model = model
            self.last_error = None
            return model
        except Exception as exc:
            self.last_error = str(exc)
            self._model_failed = True
            return None

    def _resolve_model_path(self) -> Optional[str]:
        if self.model_path:
            path = Path(self.model_path)
            if path.is_dir():
                path = path / "model.safetensors"
            if path.exists():
                return str(path)
        if self.model_id and hf_hub_download is not None:
            cache_dir = self.model_cache_dir or "data/models/geoclip"
            try:
                return hf_hub_download(
                    repo_id=self.model_id,
                    filename="model.safetensors",
                    cache_dir=cache_dir,
                )
            except Exception:
                return None
        return None


def _load_sidecar_candidates(image_path: str) -> List[GeoCandidate]:
    path = Path(image_path)
    candidates = [
        Path(str(path) + ".geo.json"),
        path.with_suffix(".geo.json"),
        Path(str(path) + ".geoloc.json"),
        path.with_suffix(".geoloc.json"),
    ]
    sidecar = next((p for p in candidates if p.exists()), None)
    if sidecar is None:
        return []

    try:
        raw = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception:
        return []

    if isinstance(raw, dict) and "candidates" in raw:
        items = raw.get("candidates")
    else:
        items = [raw]

    results: List[GeoCandidate] = []
    if not isinstance(items, list):
        return results

    for item in items:
        if not isinstance(item, dict):
            continue
        lat = item.get("latitude")
        lon = item.get("longitude")
        score = item.get("retrieval_score", item.get("confidence", 0.3))
        match_id = item.get("match_id") or item.get("landmark_id") or item.get("tile_id")
        if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
            continue
        if not (-90.0 <= float(lat) <= 90.0 and -180.0 <= float(lon) <= 180.0):
            continue
        if not isinstance(score, (int, float)):
            score = 0.3
        results.append(
            GeoCandidate(
                latitude=float(lat),
                longitude=float(lon),
                retrieval_score=float(score),
                match_id=None if match_id is None else str(match_id),
            )
        )

    return results


