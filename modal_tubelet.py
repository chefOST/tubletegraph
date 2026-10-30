"""Run pinned TubeletGraph inference for the CMU S12 MOSCATO experiment on Modal.

One-time local setup:
    python3 -m pip install modal
    modal setup
    modal volume create tubelet-data
    modal volume create tubelet-ground-truth
    modal secret create tubelet-openai OPENAI_API_KEY="$OPENAI_API_KEY"
    modal secret create wandb WANDB_API_KEY="$WANDB_API_KEY"

Smoke test the official bundled example first:
    modal run modal_tubelet.py --smoke-test

Run the prepared S12 clip:
    modal run modal_tubelet.py \
        --video-id S12_Sandwich_7150991-2470 \
        --fps 30 \
        --clip-start-global 2824 \
        --object-name "bread slices"

After visual validation, evaluate and publish to W&B:
    modal run modal_tubelet.py \
        --evaluate-run-name <completed-run-name>
"""

from __future__ import annotations

from datetime import datetime, timezone
import os
import re

import modal


APP_NAME = "tubeletgraph-moscato-cmu"
VOLUME_NAME = "tubelet-data"
GROUND_TRUTH_VOLUME_NAME = "tubelet-ground-truth"
SECRET_NAME = "tubelet-openai"
WANDB_SECRET_NAME = "wandb"
R2_SECRET_NAME = "chefost-r2"
R2_BUCKET = "chefost-moscato-gt-videos"
R2_ENDPOINT = "https://2aee25b6b08cc209d7cab66f1d0942fd.r2.cloudflarestorage.com"
R2_VIDEO_PREFIX = "moscato/releases/564f350e536734db/videos/"
# Environment flags read by the patched TubeletGraph (scripts/tubelet_speedups.py).
SAM2_SPEEDUP_ENV = {
    "TUBELETGRAPH_SHARE_SAM2": "1",
    "TUBELETGRAPH_CACHE_SAM2_FEATURES": "1",
}
WANDB_EXPERIMENTS_PROJECT = "tubeletgraph-experiments"
WANDB_FULL_EVAL_PROJECT = "tubeletgraph - full evals"
DEFAULT_WANDB_PROJECT = WANDB_FULL_EVAL_PROJECT
DEFAULT_WANDB_ENTITY = "chefOST"
TUBELETGRAPH_REPOSITORY = "https://github.com/YihongSun/TubeletGraph.git"
TUBELETGRAPH_COMMIT = "fdb05b6fbd7f4644aea990bf967cc18d82bf291b"
DETECTRON2_COMMIT = "a2f4a8771ab77e8411c26b27f24f9489a28a2453"
DEFAULT_VIDEO_ID = "S12_Sandwich_7150991-2470"
DEFAULT_OBJECT = "bread slices"
DEFAULT_CLIP_START = 2824
DEFAULT_FPS = 30
DEFAULT_OBJECT_ID = 1
DEFAULT_EXPECTED_FRAMES = 3890


app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
ground_truth_volume = modal.Volume.from_name(
    GROUND_TRUTH_VOLUME_NAME, create_if_missing=True
)
openai_secret = modal.Secret.from_name(
    SECRET_NAME, required_keys=["OPENAI_API_KEY"]
)
wandb_secret = modal.Secret.from_name(
    WANDB_SECRET_NAME, required_keys=["WANDB_API_KEY"]
)
# Shared MOSCATO videos on R2 (see chefOST-data README). Read-only; mounted
# only by the CPU frame-preparation function. Set MOSCATO_DISABLE_R2=1 to run
# the app in a workspace that does not have the chefost-r2 Secret yet.
R2_ENABLED = os.environ.get("MOSCATO_DISABLE_R2") != "1"
r2_videos = modal.CloudBucketMount(
    bucket_name=R2_BUCKET,
    bucket_endpoint_url=R2_ENDPOINT,
    key_prefix=R2_VIDEO_PREFIX,
    secret=modal.Secret.from_name(
        R2_SECRET_NAME,
        required_keys=["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"],
    ),
    read_only=True,
)


tubelet_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.3-devel-ubuntu22.04",
        add_python="3.10",
    )
    .entrypoint([])
    .apt_install(
        "build-essential",
        "ca-certificates",
        "ffmpeg",
        "git",
        "libgl1",
        "libglib2.0-0",
        "libjpeg-dev",
        "libpng-dev",
        "ninja-build",
        "wget",
    )
    .env(
        {
            "FORCE_CUDA": "1",
            "TORCH_CUDA_ARCH_LIST": "8.0",
            "PYTHONUNBUFFERED": "1",
        }
    )
    .run_commands(
        f"git clone --recurse-submodules {TUBELETGRAPH_REPOSITORY} /opt/TubeletGraph",
        f"cd /opt/TubeletGraph && git checkout {TUBELETGRAPH_COMMIT} && "
        "git submodule update --init --recursive",
        "cd /opt/TubeletGraph && python -m pip install "
        "torch==2.7.0 torchvision==0.22.0 "
        "--index-url https://download.pytorch.org/whl/cu126",
        "python -m pip install --upgrade "
        "pip==25.3 setuptools==80.9.0 wheel",
        "cd /opt/TubeletGraph && "
        "grep -v '^mmcv==' requirements.txt > /tmp/tubelet-requirements.txt && "
        "python -m pip install -r /tmp/tubelet-requirements.txt",
        # TubeletGraph does not import mmcv.ops; use OpenMMLab's official
        # non-op distribution and avoid compiling optional CUDA extensions.
        "python -m pip install mmcv-lite==2.2.0",
        "cd /opt/TubeletGraph && bash thirdparty/setup_ckpts.sh",
        "cd /opt/TubeletGraph/thirdparty/sam2 && python -m pip install -e .",
        "cd /opt/TubeletGraph/thirdparty/sam2 && "
        'python -m pip install -e ".[notebooks]"',
        "cd /opt/TubeletGraph/thirdparty/sam2 && "
        "CC=gcc CXX=g++ MAX_JOBS=8 python setup.py build_ext --inplace",
        "cd /opt/TubeletGraph/thirdparty && "
        "git clone https://github.com/facebookresearch/detectron2.git",
        f"cd /opt/TubeletGraph/thirdparty/detectron2 && "
        f"git checkout {DETECTRON2_COMMIT}",
        "cd /opt/TubeletGraph/thirdparty && "
        "CC=gcc CXX=g++ MAX_JOBS=8 "
        "python -m pip install -e detectron2 --no-build-isolation",
        "ln -s /opt/TubeletGraph/thirdparty/Entity/Entityv2/CropFormer "
        "/opt/TubeletGraph/thirdparty/detectron2/projects/CropFormer",
        "cd /opt/TubeletGraph/thirdparty/detectron2/projects/"
        "CropFormer/mask2former/modeling/pixel_decoder/ops && "
        "CC=gcc CXX=g++ MAX_JOBS=8 bash make.sh",
        "cd /opt/TubeletGraph/thirdparty/fc-clip && "
        "python -m pip install -r requirements.txt",
    )
    .add_local_dir(
        "scripts",
        "/opt/moscato/scripts",
        copy=True,
        ignore=["**/__pycache__/**", "**/*.pyc"],
    )
    .add_local_file(
        "configs/cmu_object_state_candidates.json",
        "/opt/moscato/configs/cmu_object_state_candidates.json",
        copy=True,
    )
    .run_commands(
        "python /opt/moscato/scripts/patch_tubeletgraph.py /opt/TubeletGraph",
    )
)

evaluation_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "scikit-learn>=1.7,<2",
        "wandb==0.28.2",
    )
    .add_local_dir(
        "scripts",
        "/opt/moscato/scripts",
        copy=True,
        ignore=["**/__pycache__/**", "**/*.pyc"],
    )
)


prepare_image = modal.Image.debian_slim(python_version="3.12").apt_install("ffmpeg")


def _safe_component(value: str, label: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ValueError(f"Unsafe {label}: {value!r}")
    return value


def _wandb_project(value: str) -> str:
    # The team-mandated project names contain spaces, so allow them verbatim.
    if value in (WANDB_EXPERIMENTS_PROJECT, WANDB_FULL_EVAL_PROJECT):
        return value
    return _safe_component(value, "wandb_project")


def _mask_names(input_root) -> list[str]:
    names = []
    if (input_root / "mask.png").is_file():
        names.append("mask.png")
    masks_dir = input_root / "masks"
    if masks_dir.is_dir():
        names.extend(f"masks/{path.name}" for path in sorted(masks_dir.glob("*.png")))
    return names


def _run_name(value: str) -> str:
    if value:
        return _safe_component(value, "run_name")
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


@app.function(
    image=tubelet_image,
    gpu="A100-80GB",
    cpu=8.0,
    # 64 GiB leaves room for the SAM2 feature cache to spill to CPU memory on
    # long clips. Clips under ~1,400 sampled frames fit the GPU cache budget,
    # so MOSCATO_GPU_MEMORY_MB=32768 is enough there and costs less.
    memory=int(os.environ.get("MOSCATO_GPU_MEMORY_MB", "65536")),
    timeout=12 * 60 * 60,
    volumes={"/data": data_volume},
    secrets=[openai_secret],
    single_use_containers=True,
)
def run_tubelet(
    video_id: str = DEFAULT_VIDEO_ID,
    fps: int = DEFAULT_FPS,
    clip_start_global: int = DEFAULT_CLIP_START,
    object_name: str = DEFAULT_OBJECT,
    object_id: int = DEFAULT_OBJECT_ID,
    expected_frames: int = DEFAULT_EXPECTED_FRAMES,
    max_frames: int = 0,
    frame_stride: int = 1,
    run_name: str = "",
    smoke_test: bool = False,
    persist_intermediates: bool = False,
    later_tracking: bool = True,
    sam2_speedups: bool = True,
    map_states: bool = True,
    clip_end_global: int = 0,
    mask_name: str = "mask.png",
    collect_spacing: int = 30,
) -> dict:
    """Validate inputs, run TubeletGraph, map states, and persist artifacts.

    With ``map_states=False`` the MOSCATO adapter is skipped and only
    TubeletGraph's raw output (masks and its own VLM state text under
    ``predictions/``) is saved, for downstream label matching elsewhere.
    """
    import json
    import os
    from pathlib import Path
    import shutil
    import subprocess
    import sys
    import traceback

    import numpy as np
    from PIL import Image
    import torch

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY was not injected by the Modal Secret")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to run TubeletGraph")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if max_frames < 0:
        raise ValueError("max_frames must be non-negative")
    if frame_stride < 1:
        raise ValueError("frame_stride must be at least 1")
    if collect_spacing < 1:
        raise ValueError("collect_spacing must be at least 1")
    if fps % frame_stride != 0:
        raise ValueError(
            f"fps={fps} must be divisible by frame_stride={frame_stride} "
            "so the subsampled clip has an integer frame rate"
        )
    tubelet_fps = fps // frame_stride

    video_id = _safe_component(video_id, "video_id")
    resolved_run_name = _run_name(run_name)
    repository = Path("/opt/TubeletGraph")
    started_at = datetime.now(timezone.utc).isoformat()

    if smoke_test:
        logical_video_id = "0334_cut_fruit_1"
        input_frames = repository / "assets/example/0334_cut_fruit_1"
        input_mask = repository / "assets/example/0334_cut_fruit_1_0000000.png"
        run_dir = Path("/data/smoke_tests") / resolved_run_name
        expected_input_frames = None
        state_dict_path = None
    else:
        logical_video_id = video_id
        input_root = Path("/data") / video_id
        input_frames = input_root / "frames"
        input_mask = (input_root / mask_name).resolve()
        if input_root.resolve() not in input_mask.parents:
            raise ValueError(f"mask_name escapes the video folder: {mask_name!r}")
        state_dict_path = input_root / "state_dict.json"
        legacy_ground_truth_path = input_root / "ground_truth.json"
        run_dir = input_root / "runs" / resolved_run_name
        expected_input_frames = int(expected_frames) or None

    data_volume.reload()
    if run_dir.exists():
        raise FileExistsError(
            f"Run output already exists: {run_dir}. Choose a different --run-name."
        )
    run_dir.mkdir(parents=True)
    log_path = run_dir / "tubeletgraph.log"
    manifest_path = run_dir / "run_manifest.json"

    manifest = {
        "status": "starting",
        "started_at": started_at,
        "run_name": resolved_run_name,
        "smoke_test": smoke_test,
        "video_id": logical_video_id,
        "fps": fps,
        "frame_stride": int(frame_stride),
        "tubelet_fps": tubelet_fps,
        "clip_start_global": None if smoke_test else int(clip_start_global),
        "target_object": None if smoke_test else object_name,
        "target_object_id": None if smoke_test else int(object_id),
        "expected_local_frames": expected_input_frames,
        "max_frames": int(max_frames) or None,
        "later_object_tracking": bool(later_tracking),
        "sam2_speedups": dict(SAM2_SPEEDUP_ENV) if sam2_speedups else {},
        "moscato_adapter": bool(map_states),
        "collect_spacing": int(collect_spacing),
        "tubeletgraph_repository": TUBELETGRAPH_REPOSITORY,
        "tubeletgraph_commit": TUBELETGRAPH_COMMIT,
        "detectron2_commit": DETECTRON2_COMMIT,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_vram_gb": round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 3
        ),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "openai_secret_present": True,
        "ground_truth_available_to_gpu_job": False,
        "ground_truth_volume_mounted": False,
        "adapter_input_mode": "tubelet_semantic_text_plus_single_event_frame",
        "adapter_prompt_contains_video_id": False,
        "adapter_prompt_contains_frame_number": False,
        "artifacts": {},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    caught_error: Exception | None = None
    try:
        if not input_frames.is_dir():
            raise FileNotFoundError(input_frames)
        if not input_mask.is_file():
            raise FileNotFoundError(input_mask)
        if map_states and not smoke_test and (
            state_dict_path is None or not state_dict_path.is_file()
        ):
            raise FileNotFoundError(
                f"{state_dict_path} is required for closed-vocabulary state mapping"
            )
        if not smoke_test and legacy_ground_truth_path.is_file():
            raise RuntimeError(
                "Ground truth is present on the inference Volume. Move it to "
                f"{GROUND_TRUTH_VOLUME_NAME!r} before running inference."
            )

        entries = sorted(path for path in input_frames.iterdir() if path.is_file())
        non_jpegs = [path.name for path in entries if path.suffix.lower() != ".jpg"]
        if non_jpegs:
            raise ValueError(f"Frame directory contains non-JPEG files: {non_jpegs[:5]}")
        frame_paths = [path for path in entries if path.suffix.lower() == ".jpg"]
        if not frame_paths:
            raise ValueError(f"No JPEG frames found in {input_frames}")
        if expected_input_frames is not None and len(frame_paths) != expected_input_frames:
            raise ValueError(
                f"Expected {expected_input_frames} frames; found {len(frame_paths)}"
            )
        for index, frame_path in enumerate(frame_paths):
            if frame_path.name != f"{index:07d}.jpg":
                raise ValueError(
                    f"Frame numbering breaks at {index}: found {frame_path.name}"
                )
        # Videos prepared from R2 keep the full video in frames/ with a
        # source.json; cut this object's clip from it by global frame index.
        source_json = None if smoke_test else input_root / "source.json"
        if source_json is not None and source_json.is_file():
            source_info = json.loads(source_json.read_text(encoding="utf-8"))
            source_start = int(source_info.get("clip_start_global", 0))
            offset = int(clip_start_global) - source_start
            end = int(clip_end_global) - source_start if clip_end_global else len(frame_paths)
            if offset < 0 or end > len(frame_paths) or end <= offset:
                raise ValueError(
                    f"Clip [{clip_start_global}, {clip_end_global or 'end'}) is outside "
                    f"prepared frames [{source_start}, {source_start + len(frame_paths)})"
                )
            frame_paths = frame_paths[offset:end]
            manifest["source_clip_offset"] = offset
        elif clip_end_global:
            raise ValueError(
                "clip_end_global needs frames prepared from R2 (source.json); "
                f"{input_frames} holds a pre-cut clip"
            )
        source_frame_count = len(frame_paths)
        if max_frames:
            if max_frames > source_frame_count:
                raise ValueError(
                    f"max_frames={max_frames} exceeds the {source_frame_count} input frames"
                )
            frame_paths = frame_paths[:max_frames]
        # Keep every ``frame_stride``-th frame (e.g. 30 fps -> 5 fps with 6).
        # Local frames are renumbered consecutively below; ``frame_stride`` is
        # recorded so evaluation can expand predictions back to source frames.
        selected_source_count = len(frame_paths)
        frame_paths = frame_paths[::frame_stride]

        first_frame = Image.open(frame_paths[0])
        with Image.open(input_mask) as mask_image:
            mask_source_mode = mask_image.mode
            mask = np.asarray(mask_image)
        if mask.ndim != 2:
            raise ValueError(f"Mask must be single channel; shape is {mask.shape}")
        if mask.shape != (first_frame.height, first_frame.width):
            raise ValueError(
                f"Mask shape {mask.shape} != first frame "
                f"{(first_frame.height, first_frame.width)}"
            )
        mask_values = sorted(int(value) for value in np.unique(mask))
        if smoke_test:
            if not set(mask_values).issubset(set(range(256))):
                raise ValueError(f"Unexpected smoke-test mask values: {mask_values}")
        elif mask_values != [0, int(object_id)]:
            raise ValueError(
                f"Expected mask values [0, {object_id}]; found {mask_values}"
            )

        # Copy Volume inputs to local SSD. Naming the directory after video_id is
        # essential because TubeletGraph derives its output instance name from it.
        local_input_root = Path("/tmp/tubelet_input")
        local_frames = local_input_root / logical_video_id
        local_mask = local_input_root / f"{logical_video_id}_0000000.png"
        local_input_root.mkdir(parents=True, exist_ok=True)
        local_frames.mkdir(parents=True)
        for local_index, frame_path in enumerate(frame_paths):
            shutil.copy2(frame_path, local_frames / f"{local_index:07d}.jpg")
        # TubeletGraph's loader requires an indexed PNG. Preserve the numeric
        # object IDs while normalizing legacy grayscale masks to palette mode.
        mask_uint8 = np.ascontiguousarray(mask, dtype=np.uint8)
        indexed_mask = Image.frombytes(
            "P",
            (mask_uint8.shape[1], mask_uint8.shape[0]),
            mask_uint8.tobytes(),
        )
        indexed_mask.putpalette(
            [channel for index in range(256) for channel in (index, index, index)]
        )
        indexed_mask.save(local_mask, "PNG")
        with Image.open(local_mask) as normalized_mask:
            if normalized_mask.mode != "P":
                raise RuntimeError(
                    f"Normalized mask must be indexed PNG; found {normalized_mask.mode}"
                )

        command = [
            "python",
            "quick_run.py",
            "--input_dir",
            str(local_frames),
            "--input_mask",
            str(local_mask),
            "--fps",
            str(tubelet_fps),
        ]
        manifest["source_input_frame_count"] = source_frame_count
        manifest["selected_source_frame_count"] = selected_source_count
        manifest["input_frame_count"] = len(frame_paths)
        manifest["input_dimensions"] = [first_frame.width, first_frame.height]
        manifest["mask_values"] = mask_values
        manifest["mask_source_mode"] = mask_source_mode
        manifest["mask_runtime_mode"] = "P"
        manifest["command"] = command
        manifest["status"] = "running"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

        if collect_spacing != 30:
            # Frames between later-object re-tracking rounds. The container is
            # single-use, so editing the base config only affects this run.
            base_config = repository / "configs" / "default.yaml"
            text = base_config.read_text(encoding="utf-8")
            updated, count = re.subn(
                r"(?m)^(\s*collect_spacing:\s*)\d+", rf"\g<1>{int(collect_spacing)}", text
            )
            if count != 1:
                raise RuntimeError("Expected one collect_spacing entry in configs/default.yaml")
            base_config.write_text(updated, encoding="utf-8")

        with log_path.open("w", encoding="utf-8") as log_handle:
            tubelet_env = dict(os.environ)
            if not later_tracking:
                tubelet_env["TUBELETGRAPH_SKIP_LATER_TRACKING"] = "1"
            if sam2_speedups:
                tubelet_env.update(SAM2_SPEEDUP_ENV)
                if not smoke_test:
                    # Per-frame CropFormer results keyed by frame bytes, shared
                    # by every object run on this video.
                    tubelet_env["TUBELETGRAPH_CROPFORMER_CACHE_DIR"] = str(
                        input_root / "cropformer_cache"
                    )
            process = subprocess.Popen(
                command,
                cwd=repository,
                env=tubelet_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                log_handle.write(line)
                log_handle.flush()
            return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)

        if not smoke_test:
            prediction_json = (
                repository
                / "_pred_out"
                / f"custom-{video_id}-Ours_gpt-4.1"
                / f"{video_id}_{object_id}.json"
            )
            if not prediction_json.is_file():
                raise FileNotFoundError(
                    f"Expected TubeletGraph VLM prediction was not produced: {prediction_json}"
                )

        if not smoke_test and map_states:
            sys.path.insert(0, "/opt/moscato")
            from openai import OpenAI
            from scripts.build_event_draft import build_draft
            from scripts.map_event_states import map_events

            prediction = json.loads(prediction_json.read_text(encoding="utf-8"))
            state_dict = json.loads(state_dict_path.read_text(encoding="utf-8"))
            candidate_config = json.loads(
                Path(
                    "/opt/moscato/configs/cmu_object_state_candidates.json"
                ).read_text(encoding="utf-8")
            )
            state_candidates = (
                candidate_config.get("videos", {})
                .get(video_id, {})
                .get(object_name)
            )
            event_config = {
                "video_id": video_id,
                "target_object": object_name,
                "clip_start_global": int(clip_start_global),
                "clip_end_global": int(clip_start_global) + len(frame_paths),
            }
            draft = build_draft(prediction, event_config)
            draft["frame_stride"] = int(frame_stride)
            draft["source_fps"] = int(fps)
            draft["tubelet_fps"] = int(tubelet_fps)
            draft["source_frame_count"] = int(selected_source_count)
            draft_path = run_dir / "tubelet_events.draft.json"
            draft_path.write_text(json.dumps(draft, indent=2) + "\n")
            events = map_events(
                draft,
                state_dict,
                local_frames,
                OpenAI(),
                "gpt-4.1",
                allowed_states=state_candidates,
            )
            events_path = run_dir / "tubelet_events.json"
            events_path.write_text(json.dumps(events, indent=2) + "\n")
            manifest["artifacts"]["event_draft"] = draft_path.name
            manifest["artifacts"]["mapped_events"] = events_path.name
            manifest["state_candidate_policy"] = candidate_config["policy"]
            manifest["state_candidates"] = state_candidates

        manifest["status"] = "succeeded"
    except Exception as error:  # Persist partial outputs before re-raising.
        caught_error = error
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        manifest["traceback"] = traceback.format_exc()
    finally:
        artifact_errors = []
        artifact_sources = {
            "predictions": repository / "_pred_out",
            "visualizations": repository / "_vis_out",
        }
        if persist_intermediates:
            artifact_sources["intermediates"] = repository / "_interm_out"
        for artifact_name, source in artifact_sources.items():
            if not source.exists():
                continue
            destination = run_dir / artifact_name
            try:
                shutil.copytree(source, destination)
                manifest["artifacts"][artifact_name] = artifact_name
            except Exception as artifact_error:
                artifact_errors.append(
                    f"{artifact_name}: {type(artifact_error).__name__}: {artifact_error}"
                )
        custom_configs = sorted((repository / "configs").glob("custom_*.yaml"))
        if custom_configs:
            config_dir = run_dir / "generated_configs"
            config_dir.mkdir(exist_ok=True)
            for config_path in custom_configs:
                shutil.copy2(config_path, config_dir / config_path.name)
            manifest["artifacts"]["generated_configs"] = "generated_configs"
        if artifact_errors:
            manifest["artifact_errors"] = artifact_errors
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest["volume_path"] = str(run_dir)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        data_volume.commit()

    if caught_error is not None:
        raise caught_error
    return {
        "status": manifest["status"],
        "volume": VOLUME_NAME,
        "volume_path": str(run_dir),
        "manifest": str(manifest_path),
    }


@app.function(
    image=prepare_image,
    cpu=4.0,
    memory=8192,
    timeout=2 * 60 * 60,
    volumes={"/data": data_volume, **({"/videos": r2_videos} if R2_ENABLED else {})},
)
def prepare_from_r2(
    video_id: str,
    r2_path: str,
    clip_start_global: int = 0,
    clip_end_global: int = 0,
    dataset: str = "CMU",
    overwrite: bool = False,
    skip_existing: bool = False,
) -> dict:
    """Decode an R2 video into the zero-based clip frames the GPU run expects.

    Mirrors scripts/prepare_frames.py: decode every frame with ``-vsync 0`` so
    indices match source frames, then keep [clip_start_global, clip_end_global).
    ``clip_end_global=0`` keeps every frame from the start. The hand-drawn
    first-frame masks must still be uploaded, one per object, to
    /<video_id>/masks/<object-slug>.png. Batch runs prepare the full video once
    and each object run cuts its own clip.
    """
    import json
    from pathlib import Path
    import shutil
    import subprocess
    import tempfile

    video_id = _safe_component(video_id, "video_id")
    dataset = _safe_component(dataset, "dataset")
    if not R2_ENABLED:
        raise RuntimeError("R2 is disabled (MOSCATO_DISABLE_R2=1); create the chefost-r2 Secret")
    videos_root = Path("/videos").resolve()
    source = (videos_root / r2_path).resolve()
    if videos_root not in source.parents:
        raise ValueError(f"r2_path escapes the video folder: {r2_path!r}")
    if not source.is_file():
        raise FileNotFoundError(f"Not found on R2 mount: {r2_path}")
    if clip_start_global < 0 or (clip_end_global and clip_end_global <= clip_start_global):
        raise ValueError("Need 0 <= clip_start_global < clip_end_global (or end=0)")

    data_volume.reload()
    input_root = Path("/data") / video_id
    frames_dir = input_root / "frames"
    if frames_dir.exists() and skip_existing and not overwrite:
        source_json = input_root / "source.json"
        if not source_json.is_file():
            raise FileExistsError(
                f"{frames_dir} was not prepared from R2 (no source.json), so its frame "
                "numbering is unknown. Pass overwrite in the batch entry to rebuild it."
            )
        metadata = json.loads(source_json.read_text())
        metadata.update(
            video_id=video_id,
            clip_frames=sum(1 for _ in frames_dir.glob("*.jpg")),
            masks=_mask_names(input_root),
            reused_existing_frames=True,
        )
        return metadata
    if frames_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{frames_dir} exists; pass overwrite=True to rebuild")
        shutil.rmtree(frames_dir)

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=avg_frame_rate,width,height",
         "-of", "json", str(source)],
        check=True, capture_output=True, text=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    numerator, denominator = (int(x) for x in stream["avg_frame_rate"].split("/"))

    with tempfile.TemporaryDirectory() as raw:
        raw_dir = Path(raw)
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(source), "-vsync", "0",
             "-q:v", "2", "-start_number", "0", str(raw_dir / "%07d.jpg")],
            check=True,
        )
        raw_frames = sorted(raw_dir.glob("*.jpg"))
        end = clip_end_global or len(raw_frames)
        if end > len(raw_frames):
            raise ValueError(
                f"clip_end_global={end} exceeds the {len(raw_frames)} decoded frames"
            )
        frames_dir.mkdir(parents=True)
        for local_index, global_index in enumerate(range(clip_start_global, end)):
            shutil.copy2(raw_frames[global_index], frames_dir / f"{local_index:07d}.jpg")

    metadata = {
        "video_id": video_id,
        "r2_bucket": R2_BUCKET,
        "r2_key": R2_VIDEO_PREFIX + r2_path,
        "source_fps": numerator / denominator if denominator else None,
        "source_dimensions": [int(stream["width"]), int(stream["height"])],
        "decoded_frames": len(raw_frames),
        "clip_start_global": int(clip_start_global),
        "clip_end_global": int(end),
        "clip_frames": int(end - clip_start_global),
        "dataset": dataset,
    }
    (input_root / "source.json").write_text(json.dumps(metadata, indent=2) + "\n")
    data_volume.commit()
    metadata["masks"] = _mask_names(input_root)
    return metadata


@app.function(
    image=evaluation_image,
    cpu=2.0,
    memory=4096,
    timeout=20 * 60,
    volumes={"/data": data_volume, "/ground-truth": ground_truth_volume},
    secrets=[wandb_secret],
    single_use_containers=True,
)
def evaluate_and_log(
    video_id: str,
    object_name: str,
    run_name: str,
    wandb_project: str,
    wandb_entity: str = "",
    wandb_tags: str = "",
) -> dict:
    """Evaluate an isolated completed run, then publish metrics to W&B."""
    import hashlib
    import json
    from pathlib import Path
    import sys

    video_id = _safe_component(video_id, "video_id")
    run_name = _safe_component(run_name, "evaluate_run_name")
    wandb_project = _wandb_project(wandb_project)
    if wandb_entity:
        wandb_entity = _safe_component(wandb_entity, "wandb_entity")

    input_root = Path("/data") / video_id
    run_dir = input_root / "runs" / run_name
    annotations_path = Path("/ground-truth") / video_id / "ground_truth.json"
    state_dict_path = input_root / "state_dict.json"
    events_path = run_dir / "tubelet_events.json"
    manifest_path = run_dir / "run_manifest.json"
    required = [annotations_path, state_dict_path, events_path, manifest_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required evaluation files are missing: {missing}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "succeeded":
        raise RuntimeError(
            f"Refusing to evaluate run with status {manifest.get('status')!r}"
        )

    sys.path.insert(0, "/opt/moscato")
    from scripts.evaluate import evaluate_events, load_json, write_csv
    from scripts.log_wandb import log_evaluation, run_tags

    rows, summary = evaluate_events(
        load_json(annotations_path),
        load_json(state_dict_path),
        load_json(events_path),
    )
    if summary["object"] != object_name:
        raise ValueError(
            f"Event object {summary['object']!r} does not match {object_name!r}"
        )

    def sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    evaluation_dir = run_dir / "evaluation"
    evaluation_dir.mkdir(exist_ok=True)
    csv_path = evaluation_dir / "frame_predictions.csv"
    summary_path = evaluation_dir / "summary.json"
    wandb_result_path = evaluation_dir / "wandb.json"
    write_csv(csv_path, rows)
    summary.update(
        {
            "run_name": run_name,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "provenance": {
                "annotations_sha256": sha256(annotations_path),
                "state_dict_sha256": sha256(state_dict_path),
                "events_sha256": sha256(events_path),
                "ground_truth_used_for_inference": False,
                "ground_truth_volume": GROUND_TRUTH_VOLUME_NAME,
                "ground_truth_mounted_only_by_evaluation": True,
            },
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    eval_kind = (
        "full-eval" if wandb_project == WANDB_FULL_EVAL_PROJECT else "experiment"
    )
    try:
        wandb_result = log_evaluation(
            summary=summary,
            rows=rows,
            project=wandb_project,
            entity=wandb_entity,
            run_name=f"{video_id}-{object_name}-{run_name}",
            artifact_paths={
                "evaluation": csv_path,
                "summary": summary_path,
                "events": events_path,
                "manifest": manifest_path,
            },
            config={
                "video_id": video_id,
                "object": object_name,
                "run_name": run_name,
                "eval_kind": eval_kind,
                "fps": manifest.get("fps"),
                "frame_stride": manifest.get("frame_stride"),
                "tubelet_fps": manifest.get("tubelet_fps"),
                "later_object_tracking": manifest.get("later_object_tracking"),
                "adapter_input_mode": manifest.get("adapter_input_mode"),
                "tubeletgraph_commit": TUBELETGRAPH_COMMIT,
                "detectron2_commit": DETECTRON2_COMMIT,
                "ground_truth_used_for_inference": False,
                "ground_truth_volume": GROUND_TRUTH_VOLUME_NAME,
                "ground_truth_mounted_only_by_evaluation": True,
                "metric_policy": summary["policy"],
            },
            tags=run_tags(
                eval_kind=eval_kind,
                video_id=video_id,
                object_name=object_name,
                manifest=manifest,
                extra=[t.strip() for t in wandb_tags.split(",") if t.strip()],
            ),
        )
        wandb_result_path.write_text(
            json.dumps(wandb_result, indent=2) + "\n", encoding="utf-8"
        )
    finally:
        data_volume.commit()

    return {
        "status": "succeeded",
        "metrics": {
            key: summary[key]
            for key in ("precision", "accuracy", "f1", "f1_max")
        },
        "summary": str(summary_path),
        "wandb": wandb_result,
    }


@app.function(
    image=prepare_image,
    cpu=1.0,
    memory=1024,
    timeout=10 * 60,
    volumes={"/videos": r2_videos} if R2_ENABLED else {},
)
def r2_manifest() -> str:
    """Return /videos/manifest.json from the R2 mount."""
    from pathlib import Path

    if not R2_ENABLED:
        raise RuntimeError("R2 is disabled (MOSCATO_DISABLE_R2=1); create the chefost-r2 Secret")
    return Path("/videos/manifest.json").read_text(encoding="utf-8")


BATCH_RUN_KEYS = {
    "video_id", "fps", "clip_start_global", "object_name", "object_id",
    "expected_frames", "max_frames", "frame_stride", "run_name",
    "later_tracking", "persist_intermediates", "map_states",
    "clip_end_global", "mask_name", "collect_spacing",
}
BATCH_PREPARE_KEYS = {"r2_path", "dataset", "overwrite"}


def object_slug(object_name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", object_name).strip("-").lower()
    if not slug:
        raise ValueError(f"Cannot form a file name from object {object_name!r}")
    return slug


def load_batch(path: str) -> list[dict]:
    """Read a batch file: {"defaults": {...}, "videos": [{...}, ...]}.

    Each video needs ``video_id`` and ``object_name``. Add ``r2_path`` (relative
    to the R2 video folder) to have frames prepared from R2 first.
    """
    import json
    from pathlib import Path

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    defaults = payload.get("defaults", {})
    videos = payload.get("videos")
    if not isinstance(videos, list) or not videos:
        raise ValueError("Batch file needs a non-empty 'videos' list")
    allowed = BATCH_RUN_KEYS | BATCH_PREPARE_KEYS
    specs, seen = [], set()
    for index, video in enumerate(videos):
        spec = {**defaults, **video}
        unknown = sorted(set(spec) - allowed)
        if unknown:
            raise ValueError(f"videos[{index}] has unknown keys: {unknown}")
        for key in ("video_id", "object_name"):
            if not spec.get(key):
                raise ValueError(f"videos[{index}] is missing {key!r}")
        _safe_component(spec["video_id"], "video_id")
        identity = (spec["video_id"], spec["object_name"])
        if identity in seen:
            raise ValueError(f"Duplicate batch entry for {identity}")
        seen.add(identity)
        spec.setdefault("expected_frames", 0)
        spec.setdefault("mask_name", f"masks/{object_slug(spec['object_name'])}.png")
        specs.append(spec)
    return specs


def run_batch(
    specs: list[dict],
    *,
    run_name: str,
    max_parallel: int,
    prepare_only: bool,
    sam2_speedups: bool,
) -> list[dict]:
    """Prepare inputs from R2 in parallel, then run every ready video in parallel."""
    results: list[dict] = []
    ready: list[dict] = []

    by_video: dict[str, list[dict]] = {}
    for spec in specs:
        by_video.setdefault(spec["video_id"], []).append(spec)
    prepare_calls = []
    for video_id, video_specs in by_video.items():
        r2_paths = {spec.get("r2_path") for spec in video_specs}
        if len(r2_paths) > 1:
            raise ValueError(f"{video_id} lists different r2_path values: {sorted(map(str, r2_paths))}")
        r2_path = r2_paths.pop()
        if not r2_path:
            ready.extend(video_specs)
            continue
        first = video_specs[0]
        prepare_calls.append((video_specs, prepare_from_r2.spawn(
            video_id=video_id,
            r2_path=r2_path,
            dataset=first.get("dataset", "CMU"),
            overwrite=bool(first.get("overwrite", False)),
            skip_existing=True,
        )))
    for video_specs, call in prepare_calls:
        try:
            info = call.get()
        except Exception as error:
            results.extend({"video_id": spec["video_id"], "object_name": spec["object_name"],
                            "status": "prepare_failed", "error": f"{type(error).__name__}: {error}"}
                           for spec in video_specs)
            continue
        for spec in video_specs:
            if spec["mask_name"] not in info.get("masks", []):
                results.append({"video_id": spec["video_id"], "object_name": spec["object_name"],
                                "status": "needs_mask", "mask_name": spec["mask_name"]})
            else:
                ready.append(spec)
    # Run one object per video first so later objects reuse its CropFormer cache.
    ready.sort(key=lambda spec: [s["object_name"] for s in by_video[spec["video_id"]]].index(spec["object_name"]))

    if prepare_only:
        results.extend({"video_id": s["video_id"], "object_name": s["object_name"],
                        "status": "ready"} for s in ready)
        return results

    def launch(spec: dict):
        kwargs = {key: spec[key] for key in BATCH_RUN_KEYS if key in spec}
        kwargs.setdefault("run_name", f"{run_name}-{object_slug(spec['object_name'])}")
        # Batch runs keep TubeletGraph's raw output; labels are matched downstream.
        kwargs.setdefault("map_states", False)
        return run_tubelet.spawn(sam2_speedups=sam2_speedups, **kwargs)

    window = max_parallel if max_parallel > 0 else len(ready)
    pending: list[tuple[dict, object]] = []
    queue = list(ready)
    while queue or pending:
        while queue and len(pending) < window:
            spec = queue.pop(0)
            pending.append((spec, launch(spec)))
            print(f"launched {spec['video_id']} / {spec['object_name']}", flush=True)
        spec, call = pending.pop(0)
        try:
            outcome = call.get()
            results.append({"video_id": spec["video_id"], "object_name": spec["object_name"], **outcome})
        except Exception as error:
            results.append({"video_id": spec["video_id"], "object_name": spec["object_name"],
                            "status": "failed", "error": f"{type(error).__name__}: {error}"})
        print(f"finished {spec['video_id']}: {results[-1]['status']}", flush=True)
    return results


@app.local_entrypoint()
def main(
    video_id: str = DEFAULT_VIDEO_ID,
    fps: int = DEFAULT_FPS,
    clip_start_global: int = DEFAULT_CLIP_START,
    object_name: str = DEFAULT_OBJECT,
    object_id: int = DEFAULT_OBJECT_ID,
    expected_frames: int = DEFAULT_EXPECTED_FRAMES,
    max_frames: int = 0,
    frame_stride: int = 1,
    run_name: str = "",
    smoke_test: bool = False,
    persist_intermediates: bool = False,
    later_tracking: bool = True,
    evaluate_run_name: str = "",
    wandb_project: str = "",
    wandb_entity: str = DEFAULT_WANDB_ENTITY,
    wandb_tags: str = "",
    experiment: bool = False,
    sam2_speedups: bool = True,
    batch_file: str = "",
    max_parallel: int = 0,
    prepare_only: bool = False,
    r2_manifest_out: str = "",
) -> None:
    import json
    from pathlib import Path

    if r2_manifest_out:
        Path(r2_manifest_out).write_text(r2_manifest.remote(), encoding="utf-8")
        print(f"wrote R2 manifest to {r2_manifest_out}")
        return

    if batch_file:
        if smoke_test or evaluate_run_name:
            raise ValueError("--batch-file cannot be combined with --smoke-test or --evaluate-run-name")
        results = run_batch(
            load_batch(batch_file),
            run_name=_run_name(run_name),
            max_parallel=max_parallel,
            prepare_only=prepare_only,
            sam2_speedups=sam2_speedups,
        )
        print(json.dumps(results, indent=2))
        counts: dict[str, int] = {}
        for result in results:
            counts[result["status"]] = counts.get(result["status"], 0) + 1
        print("batch summary:", json.dumps(counts, sort_keys=True))
        failed = [r for r in results if r["status"] in ("failed", "prepare_failed")]
        if failed:
            raise SystemExit(f"{len(failed)} of {len(results)} batch entries failed")
        return

    # Evaluations log to "tubeletgraph - full evals" by default; pass
    # --experiment for development runs (tubeletgraph-experiments).
    if experiment and wandb_project and wandb_project != WANDB_EXPERIMENTS_PROJECT:
        raise ValueError("--experiment cannot be combined with a different --wandb-project")
    if not wandb_project:
        wandb_project = WANDB_EXPERIMENTS_PROJECT if experiment else DEFAULT_WANDB_PROJECT

    if evaluate_run_name:
        if smoke_test or run_name:
            raise ValueError(
                "--evaluate-run-name cannot be combined with --smoke-test or --run-name"
            )
        result = evaluate_and_log.remote(
            video_id=video_id,
            object_name=object_name,
            run_name=evaluate_run_name,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            wandb_tags=wandb_tags,
        )
        print(json.dumps(result, indent=2))
        return

    result = run_tubelet.remote(
        video_id=video_id,
        fps=fps,
        clip_start_global=clip_start_global,
        object_name=object_name,
        object_id=object_id,
        expected_frames=expected_frames,
        max_frames=max_frames,
        frame_stride=frame_stride,
        run_name=run_name,
        smoke_test=smoke_test,
        persist_intermediates=persist_intermediates,
        later_tracking=later_tracking,
        sam2_speedups=sam2_speedups,
    )
    print(json.dumps(result, indent=2))
