"""Plan ingredient seeds from object presence; never export ground-truth states."""

from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
GROUPS = {
    "CMU": ("CMU/gt_annotations_cmu.json", "CMU/object_dict.json"),
    "EGTEA": ("EGTEA/gt_annotations_egtea.json", "EGTEA/obj_dict.json"),
    **{f"EgoPER/{task}": (f"EgoPER/{task}/gt_annotations_{task}.json",
                           f"EgoPER/{task}/object_dict.json")
       for task in ("coffee", "oatmeal", "pinwheels", "quesadilla", "tea")},
}


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _data_revision(root: Path, paths: list[str]) -> str:
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(root), *args], check=True,
                              capture_output=True, text=True).stdout.strip()
    if Path(git("rev-parse", "--show-toplevel")).resolve() != root.resolve():
        raise ValueError("--data-root must be the chefOST-data repository root")
    refs = [git("rev-parse", ref) for ref in ("HEAD", "main", "origin/main")]
    if len(set(refs)) != 1 or git("status", "--porcelain", "--", *paths):
        raise ValueError("Selected data must be unchanged on matching HEAD/main/origin/main")
    return refs[0]


def _normalize(label: str) -> str:
    if not isinstance(label, str) or not label.strip():
        raise ValueError("Object names must be nonempty strings")
    return " ".join(label.casefold().split())


def _lookup(vocab: dict) -> dict[str, str]:
    canonical = vocab["o2i"]
    lookup = {}
    for label, name in [(name, name) for name in canonical] + list(vocab.get("all2one", {}).items()):
        if name not in canonical:
            raise ValueError(f"Unknown canonical object: {name}")
        key = _normalize(label)
        if key in lookup and lookup[key] != name:
            raise ValueError(f"Ambiguous object alias: {label}")
        lookup[key] = name
    return lookup


def _media(entry: dict | None) -> tuple[Fraction | None, int | None]:
    streams = (entry or {}).get("media", {}).get("streams", [])
    stream = next((s for s in streams if "width" in s and "height" in s), {})
    try:
        fps = Fraction(stream["avg_frame_rate"])
        count = int(stream["nb_frames"])
        if fps <= 0 or count <= 0:
            raise ValueError
        return fps, count
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None, None


def _duration(entry: dict | None) -> Fraction | None:
    media = (entry or {}).get("media", {})
    stream = next((s for s in media.get("streams", []) if "width" in s and "height" in s), {})
    for value in (stream.get("duration"), media.get("format", {}).get("duration")):
        try:
            duration = Fraction(str(value))
            if duration > 0:
                return duration
        except (ValueError, ZeroDivisionError):
            pass
    return None


def alignment_for(entry: dict | None, count: int) -> dict:
    """Infer the uniform annotation timeline from count and video duration.

    Equal source/annotation counts preserve exact frame indices. Other timelines
    produce timestamps for lookup against decoded source presentation timestamps.
    This is duration-based inference, not independent semantic GT verification.
    """
    fps, source_count = _media(entry)
    duration = _duration(entry)
    result = {"policy": "infer_from_duration", "status": "blocked", "index_basis": None,
              "source_offset_frames": 0, "reason": None,
              "independently_verified": False}
    if fps is None or duration is None or count <= 0:
        return {**result, "reason": "missing_media_timing"}
    inferred_fps = count / duration
    if count == source_count:
        basis, annotation_fps, method = "source_frame_indices", fps, "matching_source_and_annotation_counts"
    else:
        basis, annotation_fps, method = "uniform_annotation_timeline", inferred_fps, "annotation_count_divided_by_duration"
    return {**result, "status": "inferred_from_duration", "index_basis": basis,
            "method": method, "annotation_fps": float(annotation_fps),
            "annotation_fps_rational": str(annotation_fps),
            "inferred_annotation_fps": float(inferred_fps),
            "source_duration_seconds": float(duration)}


def build_plan(data_root: Path, manifest: dict, ingredient_policy: dict,
               alignment_policy: str = "infer_from_duration", *,
               manifest_sha256: str | None = None, release: dict | None = None) -> dict:
    if alignment_policy != "infer_from_duration":
        raise ValueError(f"Unknown alignment policy: {alignment_policy}")
    rules = ingredient_policy.get("datasets", {})
    if ingredient_policy.get("scope") != "ingredients" or not rules or set(rules) - GROUPS.keys():
        raise ValueError("Expected explicit ingredient rules for supported dataset groups")
    data_root = Path(data_root).resolve()
    paths = [prefix + path for group in rules for prefix, path in zip(
        ("MOSCATO/annotations/ground_truth/", "MOSCATO/vocabulary/"), GROUPS[group])]
    revision = _data_revision(data_root, paths)
    files = {}

    def read(relative: str) -> dict:
        content = (data_root / relative).read_bytes()
        files[relative] = hashlib.sha256(content).hexdigest()
        return json.loads(content)

    entries = {}
    for entry in manifest.get("entries", []):
        key = (entry["dataset"], entry["ground_truth_video_id"])
        if key in entries:
            raise ValueError(f"Duplicate manifest identity: {key}")
        entries[key] = entry
    videos = []
    for group, rule in rules.items():
        annotation_path, vocabulary_path = GROUPS[group]
        annotations = read("MOSCATO/annotations/ground_truth/" + annotation_path)
        vocab = read("MOSCATO/vocabulary/" + vocabulary_path)
        lookup = _lookup(vocab)
        included, excluded = rule["include"], rule["exclude"]
        if (len(included) != len(set(included)) or len(excluded) != len(set(excluded))
                or set(included) & set(excluded)
                or set(included) | set(excluded) != set(vocab["o2i"])):
            raise ValueError(f"Ingredient policy must classify every {group} object exactly once")
        dataset = group.split("/")[0]
        for video_id, annotation in sorted(annotations.items()):
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", video_id):
                raise ValueError(f"Invalid video ID: {video_id}")
            presence = annotation["object"]
            if not isinstance(presence, list) or not presence:
                raise ValueError(f"Missing object-presence timeline for {video_id}")

            def canonical(label: str) -> str:
                try:
                    return lookup[_normalize(label)]
                except KeyError:
                    raise ValueError(f"Unknown object in {video_id}: {label}") from None

            first = {}
            frame_objects = []
            for index, labels in enumerate(presence):
                if not isinstance(labels, list):
                    raise ValueError(f"Invalid presence row in {video_id}")
                names = {canonical(label) for label in labels}
                frame_objects.append(names)
                for name in names:
                    first.setdefault(name, index)
            # Keys establish inventory coverage only; state-label values are never used.
            targets = (set(first) | {canonical(name) for name in annotation["state"]}) & set(included)
            entry = entries.get((dataset, video_id))
            available = bool(entry and str(entry.get("download_status", "")).startswith("downloaded"))
            r2_path = entry.get("local_path") if entry else None
            if available and (not r2_path or PurePosixPath(r2_path).is_absolute() or ".." in PurePosixPath(r2_path).parts):
                raise ValueError(f"Invalid manifest video path: {r2_path}")
            fps, source_count = _media(entry)
            alignment = alignment_for(entry, len(presence))
            ingredients = []
            for name in sorted(targets):
                identity = _digest([dataset, video_id, name])[:20]
                visible = first.get(name)
                seed = visible + 1 if visible is not None else None
                issues = []
                if not available:
                    issues.append("missing_r2_video")
                if visible is None:
                    issues.append("no_visibility_annotation")
                elif seed >= len(presence):
                    issues.append("seed_after_annotation_end")
                elif name not in frame_objects[seed]:
                    issues.append("not_visible_at_seed")
                source_seed = None
                seed_timestamp = None
                if alignment["status"] == "blocked":
                    issues.append(alignment["reason"])
                elif seed is not None:
                    seed_timestamp = float(Fraction(seed) / Fraction(alignment["annotation_fps_rational"]))
                    if alignment["index_basis"] == "source_frame_indices":
                        source_seed = seed
                    if (source_seed is not None and source_seed >= source_count) or seed_timestamp >= float(_duration(entry)):
                        issues.append("seed_after_video_end")
                non_alignment_issue = any(reason in issues for reason in (
                    "missing_r2_video", "no_visibility_annotation", "seed_after_annotation_end",
                    "not_visible_at_seed", "seed_after_video_end"))
                ingredients.append({
                    "id": identity, "object_name": name, "mask_name": f"masks/auto-{identity}.png",
                    "first_visible_annotation_index": visible, "seed_annotation_index": seed,
                    "seed_source_frame": source_seed,
                    "seed_timestamp_seconds": seed_timestamp,
                    "status": "skipped" if non_alignment_issue else "blocked" if issues else "ready",
                    "skip_reason": issues[0] if issues else None, "issues": issues,
                })
            videos.append({
                "id": _digest([dataset, video_id])[:20], "dataset": dataset, "dataset_group": group,
                "video_id": video_id, "r2_path": r2_path,
                "video_sha256": entry.get("sha256") if entry else None,
                "source_fps": float(fps) if fps else None,
                "source_fps_rational": str(fps) if fps else None,
                "source_frame_count": source_count, "annotation_frame_count": len(presence),
                "duration_seconds": float(_duration(entry)) if _duration(entry) else None,
                "availability": "available" if available else "missing_r2_video",
                "availability_reason": None if available else (entry or {}).get("blocking_reason", "Not available in the pinned R2 release"),
                "alignment": alignment, "ingredients": ingredients,
            })
    ingredients = [item for video in videos for item in video["ingredients"]]
    return {
        "schema_version": 1,
        "provenance": {"data_repository": "https://github.com/chefOST/chefOST-data",
                       "data_commit": revision, "data_file_sha256": files,
                       "manifest_sha256": manifest_sha256,
                       "manifest_content_sha256": _digest(manifest),
                       "r2_release": release, "ingredient_policy_sha256": _digest(ingredient_policy)},
        "policy": {"target_scope": "ingredients", "target_definition": ingredient_policy["definition"],
                   "visibility_source": "object_presence_rows", "state_values_used": False,
                   "seed_rule": "first_visible_annotation_index + 1; no searching or fallback",
                   "alignment_policy": alignment_policy,
                   "source_frame_rounding": "first source frame at or after the seed timestamp"},
        "summary": {"videos": len(videos), "ingredient_pairs": len(ingredients),
                    "available_videos": sum(v["availability"] == "available" for v in videos),
                    "status_counts": dict(Counter(i["status"] for i in ingredients)),
                    "skip_reason_counts": dict(Counter(i["skip_reason"] for i in ingredients if i["skip_reason"]))},
        "videos": videos,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ingredient-policy", type=Path, default=ROOT / "configs/ingredient_objects.json")
    parser.add_argument("--r2-release", type=Path, default=ROOT / "configs/r2_release.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    content = args.manifest.read_bytes()
    manifest_sha = hashlib.sha256(content).hexdigest()
    release = json.loads(args.r2_release.read_text())
    if manifest_sha != release["release_sha256"]:
        raise ValueError("Manifest bytes do not match the pinned R2 release hash")
    plan = build_plan(args.data_root, json.loads(content), json.loads(args.ingredient_policy.read_text()),
                      manifest_sha256=manifest_sha, release=release)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(plan["summary"]))


if __name__ == "__main__":
    main()
