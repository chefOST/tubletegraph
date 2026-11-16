"""Video timing and bounded sampling for automatically initialized inference."""

from __future__ import annotations

from bisect import bisect_left
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import subprocess


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_video(video: dict) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", video["video_id"]):
        raise ValueError("Unsafe video ID")
    if not re.fullmatch(r"[0-9a-f]{64}", video["video_sha256"]):
        raise ValueError("A pinned video SHA256 is required")
    relative = PurePosixPath(video["r2_path"])
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("Video path must stay inside the R2 release")


def probe_video(path: Path) -> dict:
    response = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams",
         "-show_format", "-show_frames", "-show_entries",
         "stream=width,height,avg_frame_rate,duration:format=duration:frame=best_effort_timestamp_time",
         "-of", "json", str(path)], check=True, capture_output=True, text=True,
    )
    data = json.loads(response.stdout)
    stream = data["streams"][0]
    timestamps = [float(frame["best_effort_timestamp_time"]) for frame in data["frames"]]
    if not timestamps or any(not math.isfinite(t) for t in timestamps):
        raise ValueError("Video has no usable presentation timestamps")
    if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError("Video presentation timestamps must increase strictly")
    offset = timestamps[0]
    return {
        "source_fps": float(Fraction(stream["avg_frame_rate"])),
        "duration_seconds": float(stream.get("duration") or data["format"]["duration"]),
        "source_frame_count": len(timestamps),
        "width": int(stream["width"]), "height": int(stream["height"]),
        "first_pts_seconds": offset, "timestamps_seconds": [t - offset for t in timestamps],
    }


def resolve_seed(video: dict, ingredient: dict, media: dict) -> int:
    if ingredient["status"] != "ready":
        raise ValueError("Ingredient is not ready for initialization")
    first, seed = ingredient["first_visible_annotation_index"], ingredient["seed_annotation_index"]
    if type(first) is not int or first < 0 or type(seed) is not int or seed != first + 1:
        raise ValueError("Initialization must use the annotation immediately after first visibility")
    if seed >= video["annotation_frame_count"]:
        raise ValueError("Initialization is beyond the annotation timeline")
    index = seed
    if ingredient.get("seed_source_frame") != index:
        raise ValueError("Planned source frame does not match the visibility seed")
    if not 0 <= index < media["source_frame_count"]:
        raise ValueError("Initialization is outside the video")
    return index


def sampled_frames(timestamps: list[float], seed: int, sample_fps: float = 5.0) -> list[dict]:
    """Keep the exact seed, then sample through video end using actual timestamps."""
    if not math.isfinite(sample_fps) or sample_fps <= 0:
        raise ValueError("Sampling rate must be finite and positive")
    if not 0 <= seed < len(timestamps):
        raise ValueError("Invalid seed frame")
    if any(not math.isfinite(t) for t in timestamps) or any(b <= a for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError("Expected increasing finite timestamps")
    indices = [seed]
    tick = 1
    while True:
        target = timestamps[seed] + tick / sample_fps
        index = bisect_left(timestamps, target - 1e-9, lo=indices[-1] + 1)
        if index >= len(timestamps):
            break
        indices.append(index)
        tick += 1
    return [{"local_frame": local, "source_frame": source,
             "timestamp_seconds": timestamps[source]} for local, source in enumerate(indices)]
