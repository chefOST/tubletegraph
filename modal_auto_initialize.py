"""Cloud video preparation, automatic seed masks, native inference and publication."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import modal

from modal_tubelet import (DETECTRON2_COMMIT, R2_BUCKET, R2_VIDEO_PREFIX,
                          TUBELETGRAPH_COMMIT, _safe_component, data_volume,
                          openai_secret, prepare_image, r2_videos, tubelet_image,
                          wandb_secret)
from scripts.auto_initialize import DETECTOR_MODEL_ID, DETECTOR_REVISION, TRANSFORMERS_VERSION
from scripts.auto_video import (probe_video, resolve_seed, sampled_frames,
                               sha256_file, validate_video, verify_timing)


ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path("/data")
REPOSITORY = Path("/opt/TubeletGraph")
SAMPLE_FPS = 5
INFERENCE_ENV = {
    "TUBELETGRAPH_SHARE_SAM2": "1",
    "TUBELETGRAPH_CACHE_SAM2_FEATURES": "1",
    "TUBELETGRAPH_LAZY_CPU_FRAMES": "1",
    "TUBELETGRAPH_CPU_FRAME_CACHE_SIZE": "16",
    "TUBELETGRAPH_FEATURE_CACHE_GPU_GB": "8",
    "TUBELETGRAPH_FEATURE_CACHE_CPU_GB": "8",
}

app = modal.App("tubeletgraph-auto-initialization")
auto_image = (
    tubelet_image.pip_install(f"transformers=={TRANSFORMERS_VERSION}")
    .env({"HF_HUB_CACHE": "/opt/huggingface-cache"})
    .run_commands(
        "python -c \"from huggingface_hub import snapshot_download; "
        f"snapshot_download('{DETECTOR_MODEL_ID}', revision='{DETECTOR_REVISION}', "
        "allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model'])\""
    )
)
publication_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("wandb==0.28.2")
    .add_local_file(str(ROOT / "weights_and_biases/settings.json"), "/opt/wandb-settings.json", copy=True)
)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _prepared_root(video: dict) -> Path:
    validate_video(video)
    return DATA_ROOT / "auto_inputs" / video["video_id"] / video["video_sha256"]


@app.function(image=prepare_image, cpu=4.0, memory=8192, timeout=7200,
              volumes={"/data": data_volume, "/videos": r2_videos}, retries=0)
def prepare_auto_video(video: dict) -> dict:
    """Decode and cache a verified video once; no annotation files are mounted."""
    root = _prepared_root(video)
    data_volume.reload()
    metadata_path = root / "source.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        verify_timing(video, metadata)
        if metadata["video_sha256"] != video["video_sha256"]:
            raise ValueError("Cached video identity differs from the plan")
        if len(list((root / "frames").glob("*.jpg"))) != metadata["source_frame_count"]:
            raise ValueError("Cached video frames are incomplete")
        return {"root": str(root), "metadata": str(metadata_path)}
    source_root = Path("/videos").resolve()
    source = (source_root / video["r2_path"]).resolve()
    if source_root not in source.parents or not source.is_file():
        raise FileNotFoundError("Pinned video is not available in the R2 release")
    if sha256_file(source) != video["video_sha256"]:
        raise ValueError("Video SHA256 differs from the pinned release")
    metadata = probe_video(source)
    verify_timing(video, metadata)
    root.mkdir(parents=True, exist_ok=True)
    frames = root / "frames"
    if frames.exists():
        # This is an incomplete private cache, never another run's artifacts.
        shutil.rmtree(frames)
    frames.mkdir()
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(source), "-vsync", "0", "-q:v", "2",
         "-start_number", "0", str(frames / "%07d.jpg")], check=True,
    )
    if len(list(frames.glob("*.jpg"))) != metadata["source_frame_count"]:
        raise ValueError("Decoded JPEG count differs from source timestamps")
    metadata.update(video_id=video["video_id"], video_sha256=video["video_sha256"],
                    r2_bucket=R2_BUCKET, r2_key=R2_VIDEO_PREFIX + video["r2_path"],
                    annotation_alignment=video["alignment"])
    _write_json(metadata_path, metadata)
    data_volume.commit()
    return {"root": str(root), "metadata": str(metadata_path)}


def _command(command: list[str], log_path: Path, *, env: dict | None = None) -> None:
    with log_path.open("w") as handle:
        subprocess.run(command, cwd=REPOSITORY, stdout=handle,
                       stderr=subprocess.STDOUT, env=env, check=True)


@app.function(image=auto_image, gpu="L40S", cpu=8.0, memory=65536, timeout=7200,
              volumes={"/data": data_volume}, secrets=[openai_secret],
              retries=0, single_use_containers=True)
def run_auto_initialized(video: dict, ingredient: dict, prepared: dict, run_name: str) -> dict:
    """Generate one ingredient mask, then track it through the remaining video."""
    import os

    root = _prepared_root(video)
    _safe_component(run_name, "run_name")
    if prepared != {"root": str(root), "metadata": str(root / "source.json")}:
        raise ValueError("Prepared video path does not match its identity")
    data_volume.reload()
    run_dir = DATA_ROOT / video["video_id"] / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = run_dir / "run_manifest.json"
    manifest = {
        "schema_version": 1, "status": "starting", "run_name": run_name,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "video_id": video["video_id"], "target_object": ingredient["object_name"],
        "video_sha256": video["video_sha256"], "sample_fps": SAMPLE_FPS,
        "tubeletgraph_commit": TUBELETGRAPH_COMMIT, "detectron2_commit": DETECTRON2_COMMIT,
        "requested_gpu": "L40S", "cpu_cores": 8, "memory_mib": 65536,
        "output_kind": "raw_tubeletgraph_inference", "evaluation_status": "not_scored",
        "gt_visibility_assisted": True, "ground_truth_state_values_used": False,
        "annotation_alignment": video["alignment"], "memory_settings": INFERENCE_ENV,
    }
    _write_json(run_dir / "input.json", {"video": video, "ingredient": ingredient})
    _write_json(manifest_path, manifest)
    stage = "preflight"
    try:
        media = json.loads((root / "source.json").read_text())
        if media["video_sha256"] != video["video_sha256"]:
            raise ValueError("Prepared source identity differs from plan")
        seed = resolve_seed(video, ingredient, media)
        rows = sampled_frames(media["timestamps_seconds"], seed, SAMPLE_FPS)
        manifest.update(seed_source_frame=seed, seed_timestamp_seconds=media["timestamps_seconds"][seed],
                        first_visible_annotation_index=ingredient["first_visible_annotation_index"],
                        seed_annotation_index=ingredient["seed_annotation_index"], sampled_frames=len(rows),
                        source_frame_count=media["source_frame_count"], clip_policy="seed_through_video_end")
        _write_json(run_dir / "frames.json", {"sample_fps": SAMPLE_FPS, "frames": rows})
        source_frame = root / "frames" / f"{seed:07d}.jpg"
        shutil.copy2(source_frame, run_dir / "initial_frame.jpg")
        with tempfile.TemporaryDirectory(prefix="tubelet-auto-") as temp:
            temp_root = Path(temp)
            frames = temp_root / video["video_id"]
            frames.mkdir()
            for row in rows:
                shutil.copy2(root / "frames" / f"{row['source_frame']:07d}.jpg",
                             frames / f"{row['local_frame']:07d}.jpg")
            stage = "initialization"
            manifest["status"] = "initializing"
            _write_json(manifest_path, manifest)
            data_volume.commit()
            _command(
                [sys.executable, "/opt/moscato/scripts/auto_initialize.py", "--frame", str(source_frame),
                 "--object-name", ingredient["object_name"], "--mask-out", str(run_dir / "initial_mask.png"),
                 "--provenance-out", str(run_dir / "initialization.json")],
                run_dir / "initialization.log",
            )
            stage = "inference"
            manifest["status"] = "inferring"
            _write_json(manifest_path, manifest)
            data_volume.commit()
            env = {**os.environ, **INFERENCE_ENV,
                   "TUBELETGRAPH_CROPFORMER_CACHE_DIR": str(root / "cropformer_cache")}
            _command(
                [sys.executable, "/opt/moscato/scripts/run_native_inference.py", "--input-dir", str(frames),
                 "--input-mask", str(run_dir / "initial_mask.png"), "--fps", str(SAMPLE_FPS)],
                run_dir / "tubeletgraph.log", env=env,
            )
            prediction = REPOSITORY / "_pred_out" / f"custom-{video['video_id']}-Ours_gpt-4.1" / f"{video['video_id']}_1.json"
            if not prediction.is_file() or not prediction.with_suffix(".html").is_file():
                raise RuntimeError("Native TubeletGraph prediction or prompt transcript is missing")
            output = json.loads(prediction.read_text())
            if len(output.get("prediction", {})) != len(rows):
                raise RuntimeError("Native prediction frame count does not match the sampled inputs")
            manifest["semantic_entries"] = len(output.get("obj_info", {}))
            manifest["status"] = "succeeded"
    except Exception as error:
        manifest.update(status=f"{stage}_failed", error=f"{type(error).__name__}: {error}")
    finally:
        try:
            predictions = REPOSITORY / "_pred_out"
            if predictions.exists():
                shutil.copytree(predictions, run_dir / "predictions")
            configs = sorted((REPOSITORY / "configs").glob("custom_*.yaml"))
            if configs:
                (run_dir / "generated_configs").mkdir()
                for config in configs:
                    shutil.copy2(config, run_dir / "generated_configs" / config.name)
        except Exception as error:
            manifest.update(status="artifact_failed", artifact_error=f"{type(error).__name__}: {error}")
        manifest.update(finished_at=datetime.now(timezone.utc).isoformat(), volume_path=str(run_dir))
        _write_json(manifest_path, manifest)
        data_volume.commit()
    return {"status": manifest["status"], "volume_path": str(run_dir), "manifest": str(manifest_path),
            **({"error": manifest["error"]} if "error" in manifest else {})}


@app.function(image=publication_image, cpu=2.0, memory=4096, timeout=1800,
              volumes={"/data": data_volume}, secrets=[wandb_secret], retries=0)
def publish_auto_result(video: dict, ingredient: dict, result: dict, run_name: str,
                        plan_provenance: dict) -> dict:
    """Upload artifacts directly from cloud storage, without downloading them locally."""
    import wandb

    validate_video(video)
    _safe_component(run_name, "run_name")
    data_volume.reload()
    run_dir = DATA_ROOT / video["video_id"] / "runs" / run_name
    destinations = json.loads(Path("/opt/wandb-settings.json").read_text())
    project = destinations["projects"]["experiments"]
    config = {"video": video, "ingredient": ingredient, "provenance": plan_provenance,
              "gt_visibility_assisted": True, "ground_truth_state_values_used": False,
              "output_kind": "raw_tubeletgraph_inference", "sample_fps": SAMPLE_FPS,
              "tubeletgraph_commit": TUBELETGRAPH_COMMIT}
    with wandb.init(entity=destinations["entity"], project=project, name=run_name,
                    job_type="inference", group=video["video_id"],
                    tags=["tubeletgraph", "auto-initialized", "per-ingredient", "raw-inference"], config=config,
                    settings=wandb.Settings(x_disable_stats=True)) as run:
        if run.entity.casefold() != destinations["entity"].casefold() or run.project != project:
            raise RuntimeError("Unexpected W&B destination")
        run.summary["inference/status"] = result["status"]
        if result.get("error"):
            run.summary["inference/error"] = result["error"]
        artifact = wandb.Artifact(f"tubeletgraph-{run_name}", type="inference-output")
        with tempfile.TemporaryDirectory() as temp:
            receipt = Path(temp) / "result.json"
            _write_json(receipt, {"result": result, "config": config})
            artifact.add_file(str(receipt), name="result.json")
            if run_dir.is_dir():
                for path in sorted(run_dir.rglob("*")):
                    if path.is_file():
                        artifact.add_file(str(path), name=path.relative_to(run_dir).as_posix())
                manifest_path = run_dir / "run_manifest.json"
                if manifest_path.is_file():
                    manifest = json.loads(manifest_path.read_text())
                    for key in ("sampled_frames", "semantic_entries"):
                        if key in manifest:
                            run.summary[f"inference/{key}"] = manifest[key]
            run.log_artifact(artifact).wait()
        return {"wandb_url": run.url}
