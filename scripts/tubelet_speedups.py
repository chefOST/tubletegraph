"""Lossless SAM2 speedups for TubeletGraph.

``patch_tubeletgraph.py`` copies this module into
``TubeletGraph/tracker/moscato_speedups.py`` during the Modal image build and
wires it into TubeletGraph's SAM2 wrapper and SAM2's video predictor.

TubeletGraph's tubelet stage builds three SAM2.1 Hiera-L predictors with the
same weights, loads every frame three times, and re-runs the frozen image
encoder on every frame of every tracking pass (SAM2 only keeps the most recent
frame's features). The image encoder never sees prompts or memory, so its
output for a frame is identical on every pass. These helpers:

* ``TUBELETGRAPH_SHARE_SAM2=1``: build one predictor per (config, weights,
  device) and load each video's frames once per process.
* ``TUBELETGRAPH_CACHE_SAM2_FEATURES=1``: keep each frame's image-encoder
  output and reuse it on later passes, on GPU up to
  ``TUBELETGRAPH_FEATURE_CACHE_GPU_GB`` and then on CPU up to
  ``TUBELETGRAPH_FEATURE_CACHE_CPU_GB``. Frames beyond both budgets are
  recomputed as before.

With both flags unset, behaviour is exactly upstream TubeletGraph.
"""

from __future__ import annotations

import atexit
import os
from typing import Any

SHARE_ENV = "TUBELETGRAPH_SHARE_SAM2"
CACHE_ENV = "TUBELETGRAPH_CACHE_SAM2_FEATURES"
GPU_GB_ENV = "TUBELETGRAPH_FEATURE_CACHE_GPU_GB"
CPU_GB_ENV = "TUBELETGRAPH_FEATURE_CACHE_CPU_GB"
DEFAULT_GPU_GB = 24.0
DEFAULT_CPU_GB = 24.0
LOG_PREFIX = "[moscato-speedups]"

_PREDICTORS: dict[tuple[str, str, str], Any] = {}
_FRAMES: dict[tuple[str, int, bool, str], Any] = {}
_FEATURE_CACHES: dict[tuple[int, str, int], "FeatureCache"] = {}


def enabled(name: str) -> bool:
    return os.environ.get(name) == "1"


def _budget_bytes(name: str, default_gb: float) -> int:
    value = os.environ.get(name)
    gigabytes = float(value) if value else default_gb
    if gigabytes < 0:
        raise ValueError(f"{name} must be non-negative; got {value!r}")
    return int(gigabytes * 1024**3)


class FeatureCache:
    """Per-frame SAM2 image-encoder outputs for one video and one predictor."""

    def __init__(self, owner: Any, gpu_bytes: int, cpu_bytes: int) -> None:
        # Holding the predictor keeps id(owner) unique for this cache's key.
        self.owner = owner
        self.gpu_bytes = gpu_bytes
        self.cpu_bytes = cpu_bytes
        self.used_gpu = 0
        self.used_cpu = 0
        self.entries: dict[int, list[Any]] = {}
        self.pos_enc: list[Any] | None = None
        self.per_frame_pos_enc: dict[int, list[Any]] | None = None
        self.hits = 0
        self.misses = 0
        self.uncached = 0

    def get(self, frame_idx: int, device: Any) -> dict[str, Any] | None:
        fpn = self.entries.get(frame_idx)
        if fpn is None:
            self.misses += 1
            return None
        self.hits += 1
        fpn = [tensor.to(device, non_blocking=True) for tensor in fpn]
        if self.per_frame_pos_enc is not None:
            pos_enc = list(self.per_frame_pos_enc[frame_idx])
        else:
            assert self.pos_enc is not None
            pos_enc = list(self.pos_enc)
        return {
            "vision_features": fpn[-1],
            "vision_pos_enc": pos_enc,
            "backbone_fpn": fpn,
        }

    def put(self, frame_idx: int, backbone_out: dict[str, Any]) -> None:
        import torch

        fpn = [tensor.detach() for tensor in backbone_out["backbone_fpn"]]
        pos_enc = [tensor.detach() for tensor in backbone_out["vision_pos_enc"]]
        # Positional encodings depend only on feature shape, so one copy is
        # shared. Verify that on the second frame and fall back if not.
        if self.pos_enc is None:
            self.pos_enc = pos_enc
        elif self.per_frame_pos_enc is None and len(self.entries) == 1:
            same = len(pos_enc) == len(self.pos_enc) and all(
                torch.equal(a, b) for a, b in zip(pos_enc, self.pos_enc)
            )
            if not same:
                print(f"{LOG_PREFIX} positional encodings differ per frame; "
                      "storing them per frame", flush=True)
                first_idx = next(iter(self.entries))
                self.per_frame_pos_enc = {first_idx: self.pos_enc}
        nbytes = sum(t.numel() * t.element_size() for t in fpn)
        if self.per_frame_pos_enc is not None:
            nbytes += sum(t.numel() * t.element_size() for t in pos_enc)
        if self.used_gpu + nbytes <= self.gpu_bytes:
            stored = fpn
            self.used_gpu += nbytes
        elif self.used_cpu + nbytes <= self.cpu_bytes:
            stored = [t.to("cpu") for t in fpn]
            if self.per_frame_pos_enc is not None:
                pos_enc = [t.to("cpu") for t in pos_enc]
            self.used_cpu += nbytes
        else:
            self.uncached += 1
            return
        self.entries[frame_idx] = stored
        if self.per_frame_pos_enc is not None:
            self.per_frame_pos_enc[frame_idx] = pos_enc

    def summary(self) -> str:
        return (
            f"frames_cached={len(self.entries)} hits={self.hits} "
            f"misses={self.misses} over_budget={self.uncached} "
            f"gpu_gb={self.used_gpu / 1024**3:.2f} "
            f"cpu_gb={self.used_cpu / 1024**3:.2f}"
        )


def _install_frame_cache() -> None:
    import sam2.sam2_video_predictor as video_predictor
    import torch

    original = video_predictor.load_video_frames
    if getattr(original, "_moscato_cached", False):
        return

    def cached_load_video_frames(
        video_path,
        image_size,
        offload_video_to_cpu,
        *args,
        async_loading_frames=False,
        compute_device=torch.device("cuda"),
        **kwargs,
    ):
        if async_loading_frames or args or kwargs:
            return original(
                video_path,
                image_size,
                offload_video_to_cpu,
                *args,
                async_loading_frames=async_loading_frames,
                compute_device=compute_device,
                **kwargs,
            )
        key = (
            os.path.abspath(str(video_path)),
            int(image_size),
            bool(offload_video_to_cpu),
            str(compute_device),
        )
        if key in _FRAMES:
            print(f"{LOG_PREFIX} reusing loaded frames for {key[0]}", flush=True)
        else:
            _FRAMES[key] = original(
                video_path,
                image_size,
                offload_video_to_cpu,
                async_loading_frames=False,
                compute_device=compute_device,
            )
        return _FRAMES[key]

    cached_load_video_frames._moscato_cached = True  # type: ignore[attr-defined]
    video_predictor.load_video_frames = cached_load_video_frames


def build_predictor(builder, model_cfg, model_weights, device):
    """Build a SAM2 video predictor, reusing one per process when enabled."""
    if not enabled(SHARE_ENV):
        return builder(model_cfg, model_weights, device=device)
    _install_frame_cache()
    key = (str(model_cfg), os.path.abspath(str(model_weights)), str(device))
    if key in _PREDICTORS:
        print(f"{LOG_PREFIX} reusing SAM2 predictor {key[0]}", flush=True)
    else:
        _PREDICTORS[key] = builder(model_cfg, model_weights, device=device)
        print(f"{LOG_PREFIX} built shared SAM2 predictor {key[0]}", flush=True)
    return _PREDICTORS[key]


def attach_feature_cache(predictor, inference_state: dict, video_dir: str) -> None:
    """Give an inference state the feature cache for its video and predictor."""
    if not enabled(CACHE_ENV):
        return
    key = (
        id(predictor),
        os.path.abspath(str(video_dir)),
        int(inference_state["num_frames"]),
    )
    cache = _FEATURE_CACHES.get(key)
    if cache is None:
        cache = FeatureCache(
            predictor,
            _budget_bytes(GPU_GB_ENV, DEFAULT_GPU_GB),
            _budget_bytes(CPU_GB_ENV, DEFAULT_CPU_GB),
        )
        _FEATURE_CACHES[key] = cache
    inference_state["_moscato_feature_cache"] = cache


@atexit.register
def _report() -> None:
    for (_, video_dir, _), cache in _FEATURE_CACHES.items():
        print(f"{LOG_PREFIX} feature cache {video_dir}: {cache.summary()}", flush=True)
