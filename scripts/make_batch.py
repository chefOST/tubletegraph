#!/usr/bin/env python3
"""Write a modal_tubelet.py batch file: one entry per annotated (video, object).

Each object's clip is its annotated span in the MOSCATO ground truth
[first annotated frame, last annotated frame + 1), in source-video frames,
with the start moved back onto the frame-stride grid (at most stride - 1
frames) so all objects of a video sample the same frames.
Videos are resolved to R2 paths through the bucket's manifest.json, fetched with

    modal run modal_tubelet.py --r2-manifest-out r2_manifest.json

Example:

    python3 scripts/make_batch.py \
      --data-repo ../chefOST-data \
      --manifest r2_manifest.json \
      --out configs/batch_moscato_cmu_egtea.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# Sampling per dataset: CMU-MMAC is 30 fps (stride 6 -> 5 fps, as in the S12
# run); EGTEA Gaze+ is 24 fps (stride 24 -> 1 fps keeps long clips affordable).
DATASETS = {
    "CMU": {"gt": "MOSCATO/annotations/ground_truth/CMU/gt_annotations_cmu.json",
            "fps": 30, "frame_stride": 6},
    "EGTEA": {"gt": "MOSCATO/annotations/ground_truth/EGTEA/gt_annotations_egtea.json",
              "fps": 24, "frame_stride": 24},
}


def object_spans(annotations: dict[str, Any]) -> dict[str, tuple[int, int]]:
    spans = {}
    for object_name, per_frame in annotations["state"].items():
        annotated = [index for index, states in enumerate(per_frame) if states]
        if annotated:
            spans[object_name] = (annotated[0], annotated[-1] + 1)
    return spans


def manifest_paths(manifest: Any) -> list[str]:
    """Every string in the manifest that looks like a video path."""
    found: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
        elif isinstance(node, str) and node.lower().endswith((".mp4", ".avi", ".mov")):
            found.append(node)

    walk(manifest)
    return found


def resolve(video_id: str, paths: list[str]) -> str | None:
    matches = sorted({p for p in paths if Path(p).stem == video_id})
    if len(matches) > 1:
        raise ValueError(f"{video_id} matches several manifest paths: {matches}")
    return matches[0] if matches else None


def entry_paths(manifest: Any) -> dict[str, str] | None:
    """Map video ID to its path under the R2 video folder.

    Returns None for manifests without the expected ``entries`` list. Entries
    whose download is not verified (for example blocked EgoPER videos) are left
    out.
    """
    entries = manifest.get("entries") if isinstance(manifest, dict) else None
    if not isinstance(entries, list):
        return None
    mapping = {}
    for entry in entries:
        if not str(entry.get("download_status", "")).startswith("downloaded"):
            continue
        # local_path is the object key under the R2 video folder (CMU videos sit
        # under _copies/MP4/, EGTEA videos at the top level).
        mapping[entry["ground_truth_video_id"]] = entry["local_path"]
    return mapping


def build(
    data_repo: Path,
    manifest: Any | None,
    datasets: list[str],
    sample_fps: dict[str, int] | None = None,
) -> dict[str, Any]:
    mapping = entry_paths(manifest) if manifest is not None else None
    paths = manifest_paths(manifest) if manifest is not None and mapping is None else []
    videos, unresolved = [], []
    for dataset in datasets:
        config = dict(DATASETS[dataset])
        if sample_fps and dataset in sample_fps:
            if config["fps"] % sample_fps[dataset]:
                raise ValueError(f"{dataset} fps {config['fps']} is not divisible by {sample_fps[dataset]}")
            config["frame_stride"] = config["fps"] // sample_fps[dataset]
        ground_truth = json.loads((data_repo / config["gt"]).read_text(encoding="utf-8"))
        for video_id in sorted(ground_truth):
            if mapping is not None:
                r2_path = mapping.get(video_id)
            else:
                r2_path = resolve(video_id, paths) if manifest is not None else None
            if manifest is not None and r2_path is None:
                unresolved.append(video_id)
                continue
            for object_name, (start, end) in sorted(object_spans(ground_truth[video_id]).items()):
                # Snap the start onto the stride grid so every object of a video
                # samples the same source frames and shares the CropFormer cache.
                start -= start % config["frame_stride"]
                entry = {
                    "video_id": video_id,
                    "object_name": object_name,
                    "dataset": dataset,
                    "fps": config["fps"],
                    "frame_stride": config["frame_stride"],
                    "clip_start_global": start,
                    "clip_end_global": end,
                }
                if r2_path:
                    entry["r2_path"] = r2_path
                videos.append(entry)
    return {"defaults": {"object_id": 1}, "videos": videos, "unresolved_videos": unresolved}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, help="R2 manifest.json; omit to leave r2_path unset")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cmu-fps", type=int, help="CMU sampling rate (default 5)")
    parser.add_argument("--egtea-fps", type=int, help="EGTEA sampling rate (default 1)")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8")) if args.manifest else None
    sample_fps = {k: v for k, v in {"CMU": args.cmu_fps, "EGTEA": args.egtea_fps}.items() if v}
    batch = build(args.data_repo, manifest, args.datasets, sample_fps)
    unresolved = batch.pop("unresolved_videos")
    args.out.write_text(json.dumps(batch, indent=2) + "\n", encoding="utf-8")
    per_video = {}
    for entry in batch["videos"]:
        per_video.setdefault(entry["video_id"], 0)
        per_video[entry["video_id"]] += 1
    print(f"wrote {len(batch['videos'])} object runs over {len(per_video)} videos to {args.out}")
    if unresolved:
        print(f"WARNING: not found in manifest, skipped: {unresolved}")


if __name__ == "__main__":
    main()
