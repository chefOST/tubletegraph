"""Patch the pinned TubeletGraph checkout during the Modal image build.

1. Later-object tracking can be disabled at runtime. An early return in
   ``compute_save_all_tubes`` is taken when
   ``TUBELETGRAPH_SKIP_LATER_TRACKING=1``. The loop it skips re-tracks newly
   found entities every ``collect_spacing`` frames to the end of the video,
   which scales roughly quadratically with frame count.
2. Lossless SAM2 speedups from ``scripts/tubelet_speedups.py``: one shared
   predictor and one frame load per process, and a per-frame image-encoder
   feature cache. They are off unless ``TUBELETGRAPH_SHARE_SAM2=1`` and
   ``TUBELETGRAPH_CACHE_SAM2_FEATURES=1`` are set.
3. CropFormer per-frame result cache. When
   ``TUBELETGRAPH_CROPFORMER_CACHE_DIR`` is set, each frame's entity masks are
   stored under a key made from the frame's bytes and the CropFormer config,
   and reused by later runs on the same video. CropFormer processes every
   frame independently, so a hit returns what it would have computed.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ENV_FLAG = "TUBELETGRAPH_SKIP_LATER_TRACKING"
MARKER = "    print('Running later object tracking...')\n"
GUARD = (
    f"    if os.environ.get('{ENV_FLAG}') == '1':\n"
    f"        print('Skipping later object tracking ({ENV_FLAG}=1)')\n"
    "        save_all_to_json({'all_tracks': all_tracks, 'tracked_objs': tracked_objs}, final_all_tracks_path, dim=dim)\n"
    "        return\n"
)


SPEEDUPS_SOURCE = Path(__file__).resolve().with_name("tubelet_speedups.py")

# SAM2 video predictor: consult the shared feature cache before running the
# image encoder, and fill it after.
SAM2_FEATURE_MARKER = (
    "            backbone_out = self.forward_image(image)\n"
    "            # Cache the most recent frame's feature (for repeated interactions with\n"
)
SAM2_FEATURE_PATCH = (
    "            _moscato_cache = inference_state.get(\"_moscato_feature_cache\")\n"
    "            backbone_out = (\n"
    "                _moscato_cache.get(frame_idx, device)\n"
    "                if _moscato_cache is not None\n"
    "                else None\n"
    "            )\n"
    "            if backbone_out is None:\n"
    "                backbone_out = self.forward_image(image)\n"
    "                if _moscato_cache is not None:\n"
    "                    _moscato_cache.put(frame_idx, backbone_out)\n"
    "            # Cache the most recent frame's feature (for repeated interactions with\n"
)

# TubeletGraph's SAM2 wrapper: shared predictor and feature-cache attachment.
WRAPPER_IMPORT_MARKER = "from sam2.build_sam import build_sam2_video_predictor\n"
WRAPPER_IMPORT_PATCH = (
    WRAPPER_IMPORT_MARKER
    + "from . import moscato_speedups as _moscato_speedups\n"
)
WRAPPER_BUILD_MARKER = (
    "        self.predictor = build_sam2_video_predictor(model_cfg, model_weights, device=device)\n"
)
WRAPPER_BUILD_PATCH = (
    "        self.predictor = _moscato_speedups.build_predictor(\n"
    "            build_sam2_video_predictor, model_cfg, model_weights, device\n"
    "        )\n"
)
WRAPPER_INIT_MARKER = (
    "        self.inference_state = self.predictor.init_state(video_path=info['video_dir'])\n"
)
WRAPPER_INIT_PATCH = WRAPPER_INIT_MARKER + (
    "        _moscato_speedups.attach_feature_cache(\n"
    "            self.predictor, self.inference_state, info['video_dir']\n"
    "        )\n"
)


CROPFORMER_HELPERS_MARKER = "def setup_cfg(config_file, opts):\n"
CROPFORMER_HELPERS = '''def _moscato_cf_key(path, cf_cfg, threshold):
    """Cache path for one frame's CropFormer result, or None if disabled."""
    cache_dir = os.environ.get("TUBELETGRAPH_CROPFORMER_CACHE_DIR")
    if not cache_dir:
        return None
    import hashlib
    fingerprint = hashlib.sha256(json.dumps([
        str(cf_cfg.config_path), [str(o) for o in cf_cfg.opts], float(threshold),
    ]).encode("utf-8")).hexdigest()[:16]
    with open(path, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    return osp.join(cache_dir, fingerprint, digest[:2], digest + ".json")


def _moscato_cf_get(key):
    if key is None or not osp.isfile(key):
        return None
    with open(key, "r") as handle:
        return json.load(handle)


def _moscato_cf_put(key, value):
    if key is None:
        return
    os.makedirs(osp.dirname(key), exist_ok=True)
    tmp = "{}.{}.tmp".format(key, os.getpid())
    with open(tmp, "w") as handle:
        json.dump(value, handle)
    os.replace(tmp, key)


'''
CROPFORMER_LOOP_MARKER = "        for path in tqdm.tqdm(frame_paths, desc=vid_dir):\n"
CROPFORMER_LOOP_PATCH = CROPFORMER_LOOP_MARKER + (
    "            _moscato_key = _moscato_cf_key(path, cf_cfg, confidence_threshold)\n"
    "            _moscato_hit = _moscato_cf_get(_moscato_key)\n"
    "            if _moscato_hit is not None:\n"
    "                out[len(out)] = _moscato_hit\n"
    "                continue\n"
)
CROPFORMER_STORE_MARKER = (
    "            out[len(out)] = {ii: bmask_to_rle(mask_id==mid) for ii, mid in enumerate(unique_mask_id)}\n"
)
CROPFORMER_STORE_PATCH = CROPFORMER_STORE_MARKER + (
    "            _moscato_cf_put(_moscato_key, out[len(out) - 1])\n"
)


def replace_once(path: Path, marker: str, replacement: str, label: str) -> bool:
    """Replace exactly one marker; return False if already applied."""
    source = path.read_text(encoding="utf-8")
    if replacement in source:
        return False
    if source.count(marker) != 1:
        raise SystemExit(f"expected exactly one {label} marker in {path}")
    path.write_text(source.replace(marker, replacement), encoding="utf-8")
    return True


def patch(path: Path) -> bool:
    source = path.read_text(encoding="utf-8")
    if GUARD in source:
        return False
    if source.count(MARKER) != 1:
        raise SystemExit(f"expected exactly one later-tracking marker in {path}")
    path.write_text(source.replace(MARKER, GUARD + MARKER), encoding="utf-8")
    return True


def patch_speedups(repository: Path) -> list[tuple[Path, bool]]:
    tracker_dir = repository / "TubeletGraph" / "tracker"
    module = tracker_dir / "moscato_speedups.py"
    shutil.copyfile(SPEEDUPS_SOURCE, module)
    wrapper = tracker_dir / "sam2.py"
    predictor = (
        repository / "thirdparty" / "sam2" / "sam2" / "sam2_video_predictor.py"
    )
    return [
        (module, True),
        (wrapper, replace_once(wrapper, WRAPPER_IMPORT_MARKER, WRAPPER_IMPORT_PATCH, "wrapper import")),
        (wrapper, replace_once(wrapper, WRAPPER_BUILD_MARKER, WRAPPER_BUILD_PATCH, "wrapper build")),
        (wrapper, replace_once(wrapper, WRAPPER_INIT_MARKER, WRAPPER_INIT_PATCH, "wrapper init_state")),
        (predictor, replace_once(predictor, SAM2_FEATURE_MARKER, SAM2_FEATURE_PATCH, "SAM2 feature")),
    ]


def patch_cropformer_cache(repository: Path) -> list[tuple[Path, bool]]:
    target = repository / "TubeletGraph" / "entity_segmentation" / "cropformer.py"
    return [
        (target, replace_once(target, CROPFORMER_HELPERS_MARKER, CROPFORMER_HELPERS + CROPFORMER_HELPERS_MARKER, "cropformer helpers")),
        (target, replace_once(target, CROPFORMER_LOOP_MARKER, CROPFORMER_LOOP_PATCH, "cropformer loop")),
        (target, replace_once(target, CROPFORMER_STORE_MARKER, CROPFORMER_STORE_PATCH, "cropformer store")),
    ]


def main(argv: list[str]) -> int:
    repository = Path(argv[1] if len(argv) > 1 else "/opt/TubeletGraph")
    target = repository / "TubeletGraph" / "tubelet" / "compute_tubelets_sam.py"
    changed = patch(target)
    print(f"{'patched' if changed else 'already patched'}: {target}")
    for path, changed in patch_speedups(repository) + patch_cropformer_cache(repository):
        print(f"{'patched' if changed else 'already patched'}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
