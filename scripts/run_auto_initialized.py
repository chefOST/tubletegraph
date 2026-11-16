"""Preview or execute a cloud-only plan for automatically initialized TubeletGraph runs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib import import_module
import json
import math
from pathlib import Path
import re
import sys
import time
import uuid


GPU_HOURLY_USD = 2.839968  # L40S + 8 CPU cores + 64 GiB; estimate, not a billing cap.
BOOT_ALLOWANCE_SECONDS = 120
VIDEO_FIELDS = {
    "id", "video_id", "dataset", "r2_path", "video_sha256", "source_fps",
    "source_frame_count", "annotation_frame_count", "duration_seconds", "alignment",
}
INGREDIENT_FIELDS = {
    "id", "job_id", "object_name", "mask_name", "first_visible_annotation_index",
    "seed_annotation_index", "seed_source_frame", "seed_timestamp_seconds",
    "status", "skip_reason",
}


def _job_id(ingredient: dict) -> str:
    value = str(ingredient.get("job_id") or ingredient.get("id") or "")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", value):
        raise ValueError("Each ingredient needs a safe, stable id or job_id")
    return value


def load_plan(path: Path) -> dict:
    plan = json.loads(path.read_text())
    if not isinstance(plan.get("videos"), list):
        raise ValueError("Plan must contain a videos list")
    seen = set()
    for video in plan["videos"]:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", video["video_id"]):
            raise ValueError("Unsafe video_id")
        for ingredient in video["ingredients"]:
            key = _job_id(ingredient)
            if key in seen:
                raise ValueError(f"Duplicate ingredient id: {key}")
            seen.add(key)
            if ingredient["status"] not in {"ready", "skipped", "blocked"}:
                raise ValueError(f"Unknown plan status for {key}")
            if ingredient["status"] == "ready":
                frame = ingredient["seed_source_frame"]
                if frame is None:
                    timestamp = ingredient.get("seed_timestamp_seconds")
                    if not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp) or not 0 <= timestamp < video["duration_seconds"]:
                        raise ValueError(f"Invalid seed timestamp for {key}")
                elif not isinstance(frame, int) or isinstance(frame, bool) or not 0 <= frame < video["source_frame_count"]:
                    raise ValueError(f"Invalid seed frame for {key}")
                if not video.get("r2_path") or not video.get("video_sha256"):
                    raise ValueError(f"Missing video provenance for {key}")
    return plan


def preview(plan: dict) -> dict:
    counts = {status: 0 for status in ("ready", "skipped", "blocked")}
    for video in plan["videos"]:
        for ingredient in video["ingredients"]:
            counts[ingredient["status"]] += 1
    return {"mode": "preview", "videos": len(plan["videos"]), **counts,
            "gt_visibility_assisted": True, "ground_truth_states_used": False,
            "execution": "Requires --execute and --budget-usd; no cloud calls made."}


def _budget_seconds(budget_usd: float, reserve_usd: float, max_job_seconds: int) -> float:
    if not math.isfinite(budget_usd) or budget_usd <= 0:
        raise ValueError("Budget must be finite and positive")
    if not math.isfinite(reserve_usd) or reserve_usd < 0:
        raise ValueError("Reserve must be finite and nonnegative")
    if max_job_seconds < 60:
        raise ValueError("--max-job-seconds must be at least 60")
    seconds = (budget_usd - reserve_usd) / GPU_HOURLY_USD * 3600
    if seconds < BOOT_ALLOWANCE_SECONDS + 60:
        raise ValueError("Compute allowance needs at least 60 seconds plus startup allowance")
    return seconds


def _save(path: Path, state: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(path)


def _run_job(runtime, video, ingredient, prepared, run_name, timeout, remaining):
    call = None
    fatal = None
    started = time.monotonic()
    result = {"status": "failed", "volume_path": f"/data/{video['video_id']}/runs/{run_name}",
              "manifest": f"/data/{video['video_id']}/runs/{run_name}/run_manifest.json"}

    def cancel():
        if call is not None:
            try:
                call.cancel(terminate_containers=True)
            except Exception as error:
                result["cancellation_error"] = f"{type(error).__name__}: {error}"

    try:
        gpu = runtime.run_auto_initialized.with_options(
            cpu=(8.0, 8.0), memory=(65536, 65536), timeout=timeout,
            retries=0, max_containers=1, scaledown_window=2,
        )
        call = gpu.spawn(video=video, ingredient=ingredient, prepared=prepared, run_name=run_name)
        while True:
            elapsed = time.monotonic() - started
            if elapsed >= min(timeout + BOOT_ALLOWANCE_SECONDS, remaining):
                result.update(status="timed_out", error="Cancelled at the time-based GPU budget limit")
                cancel()
                break
            try:
                response = call.get(timeout=min(30, max(1, remaining - elapsed)))
                if not isinstance(response, dict) or "status" not in response:
                    raise ValueError("GPU function returned no status")
                result.update(response)
                break
            except TimeoutError:
                continue
    except BaseException as error:
        cancel()
        result["error"] = f"{type(error).__name__}: {error}"
        if not isinstance(error, Exception):
            result["status"] = "interrupted"
            fatal = error
    return result, time.monotonic() - started, fatal


def execute_plan(plan: dict, *, budget_usd: float, reserve_usd: float,
                 max_job_seconds: int, output_dir: Path | None = None) -> dict:
    allowance = _budget_seconds(budget_usd, reserve_usd, max_job_seconds)
    name = datetime.now(timezone.utc).strftime("auto-%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:6]
    out = output_dir or Path("results") / "auto-initialized" / name
    out.mkdir(parents=True, exist_ok=False)
    records = {
        _job_id(ingredient): {"video_id": video["video_id"], "object_name": ingredient["object_name"],
                              "status": ingredient["status"], "skip_reason": ingredient.get("skip_reason")}
        for video in plan["videos"] for ingredient in video["ingredients"]
    }
    provenance = {**plan.get("provenance", {}), "gt_visibility_assisted": True,
                  "ground_truth_states_used": False, "initialization": "GroundingDINO + SAM2 on exact planned seed frame"}
    state = {"run_name": name, "status": "starting", "budget_usd": budget_usd,
             "reserve_usd": reserve_usd, "gpu_hourly_usd": GPU_HOURLY_USD,
             "estimated_gpu_compute_usd": 0.0,
             "budget_note": "Time-based GPU estimate; reserve covers separately billed CPU/API/build/storage and is not an account spending cap.",
             "provenance": provenance, "jobs": records}
    progress = out / "progress.json"
    _save(progress, state)
    if not any(record["status"] == "ready" for record in records.values()):
        state["status"] = "no_ready_jobs"
        _save(progress, state)
        return state

    # No cloud SDK or model code is imported on the preview path.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    runtime = None
    spent = 0.0

    def publish(video, ingredient, result, run_name):
        record = records[_job_id(ingredient)]
        record.update(result, run_name=run_name)
        _save(progress, state)
        try:
            publication = runtime.publish_auto_result.remote(
                video=video, ingredient=ingredient, result=result,
                run_name=run_name, plan_provenance=provenance,
            )
            record.update(publication)
        except Exception as error:
            record["publication_error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            _save(progress, state)

    try:
        runtime = import_module("modal_auto_initialize")
        with runtime.app.run(detach=True):
            state["status"] = "running"
            for source_video in plan["videos"]:
                ready = [item for item in source_video["ingredients"] if item["status"] == "ready"]
                if not ready:
                    continue
                video = {key: value for key, value in source_video.items() if key in VIDEO_FIELDS}
                if allowance - spent < BOOT_ALLOWANCE_SECONDS + 60:
                    break
                try:
                    prepared = runtime.prepare_auto_video.remote(video=video)
                except Exception as error:
                    for item in ready:
                        ingredient = {key: value for key, value in item.items() if key in INGREDIENT_FIELDS}
                        publish(video, ingredient, {"status": "prepare_failed", "error": f"{type(error).__name__}: {error}"},
                                f"{name}-{_job_id(ingredient)}")
                    continue
                for item in ready:
                    ingredient = {key: value for key, value in item.items() if key in INGREDIENT_FIELDS}
                    timeout = min(max_job_seconds, int(allowance - spent - BOOT_ALLOWANCE_SECONDS))
                    if timeout < 60:
                        break
                    run_name = f"{name}-{_job_id(ingredient)}"
                    records[_job_id(ingredient)].update(status="running", run_name=run_name)
                    _save(progress, state)
                    result, elapsed, fatal = _run_job(
                        runtime, video, ingredient, prepared, run_name, timeout, allowance - spent,
                    )
                    spent += elapsed
                    state["estimated_gpu_compute_usd"] = spent / 3600 * GPU_HOURLY_USD
                    publish(video, ingredient, result, run_name)
                    if fatal is not None:
                        raise fatal
                    if result.get("cancellation_error"):
                        raise RuntimeError("GPU cancellation could not be confirmed; stopping further launches")
            pending = [record for record in records.values() if record["status"] == "ready"]
            for record in pending:
                record["status"] = "budget_exhausted"
            failures = [record for record in records.values()
                        if record["status"] not in {"succeeded", "skipped", "blocked", "budget_exhausted"}]
            state["status"] = ("budget_exhausted" if pending else
                               "completed_with_failures" if failures else "completed")
    except BaseException as error:
        state.update(status="stopped", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        _save(progress, state)
    return state


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--budget-usd", type=float)
    parser.add_argument("--reserve-usd", type=float, default=2.0)
    parser.add_argument("--max-job-seconds", type=int, default=7200)
    parser.add_argument("--output-dir", type=Path, help="Local progress metadata only; model artifacts stay in cloud storage")
    args = parser.parse_args(argv)
    plan = load_plan(args.plan)
    if not args.execute:
        print(json.dumps(preview(plan), indent=2))
        return
    if args.budget_usd is None:
        parser.error("--execute requires --budget-usd")
    result = execute_plan(plan, budget_usd=args.budget_usd, reserve_usd=args.reserve_usd,
                          max_job_seconds=args.max_job_seconds, output_dir=args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
