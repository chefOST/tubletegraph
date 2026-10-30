#!/usr/bin/env python3
"""Publish a completed MOSCATO evaluation to Weights & Biases."""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any


PRIMARY_METRICS = ("precision", "accuracy", "f1", "f1_max")


def metric_payload(summary: dict[str, Any]) -> dict[str, float]:
    """Map evaluation output to one clearly grouped W&B metric namespace."""
    missing = [key for key in PRIMARY_METRICS if key not in summary]
    if missing:
        raise KeyError(f"Evaluation summary is missing metrics: {missing}")
    metrics = {
        f"word_accuracy/{key}": float(summary[key]) for key in PRIMARY_METRICS
    }
    metrics.update(
        {
            "word_accuracy/f1_max_threshold": float(
                summary["f1_max_threshold"]
            ),
            "word_accuracy/evaluated_frames": float(
                summary["evaluated_frames"]
            ),
            "word_accuracy/coverage": float(summary.get("coverage", 1.0)),
            "word_accuracy/abstained_frames": float(
                summary.get("abstained_frames", 0)
            ),
        }
    )
    return metrics


def _artifact_component(value: str) -> str:
    component = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    if not component:
        raise ValueError(f"Cannot form a W&B artifact component from {value!r}")
    return component[:128]


MAX_TAG_LENGTH = 64


def _tag(value: str) -> str:
    tag = re.sub(r"[^A-Za-z0-9_.:=-]+", "-", str(value)).strip("-")
    return tag[:MAX_TAG_LENGTH]


def run_tags(
    *,
    eval_kind: str,
    video_id: str,
    object_name: str,
    manifest: dict[str, Any],
    extra: tuple[str, ...] | list[str] = (),
) -> list[str]:
    """Build searchable W&B tags from the run identity and its manifest."""
    tags = ["tubeletgraph", "moscato", "cmu", "evaluation", eval_kind]
    tags.append(f"video:{video_id}")
    tags.append(f"session:{video_id.split('_', 1)[0]}")
    tags.append(f"object:{object_name}")
    if manifest.get("fps") is not None:
        tags.append(f"fps:{manifest['fps']}")
    if manifest.get("frame_stride") is not None:
        tags.append(f"stride:{manifest['frame_stride']}")
    if "later_object_tracking" in manifest:
        state = "on" if manifest["later_object_tracking"] else "off"
        tags.append(f"later-tracking:{state}")
    if manifest.get("adapter_input_mode"):
        tags.append(f"adapter:{manifest['adapter_input_mode']}")
    if manifest.get("tubeletgraph_commit"):
        tags.append(f"tubeletgraph-commit:{manifest['tubeletgraph_commit'][:7]}")
    if manifest.get("smoke_test"):
        tags.append("smoke-test")
    tags.extend(extra)
    seen: list[str] = []
    for tag in map(_tag, tags):
        if tag and tag not in seen:
            seen.append(tag)
    return seen


def log_evaluation(
    *,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    project: str,
    entity: str,
    run_name: str,
    artifact_paths: dict[str, Path],
    config: dict[str, Any],
    tags: list[str],
) -> dict[str, str]:
    """Log scalar cards, threshold curve, frame table, and source artifacts."""
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError(
            "WANDB_API_KEY was not injected by the Modal Secret named 'wandb'"
        )

    import wandb

    metrics = metric_payload(summary)
    resolved_name = _artifact_component(run_name)
    with wandb.init(
        project=project,
        entity=entity or None,
        name=resolved_name,
        job_type="evaluation",
        config=config,
        tags=tags,
    ) as run:
        for key, value in metrics.items():
            run.summary[key] = value
        run.log(metrics)

        curve = summary["f1_threshold_curve"]
        curve_table = wandb.Table(
            columns=["threshold", "precision", "recall", "f1"],
            data=[
                [
                    point["threshold"],
                    point["precision"],
                    point["recall"],
                    point["f1"],
                ]
                for point in curve
            ],
        )
        log_items: dict[str, Any] = {
            "word_accuracy/f1_threshold_curve": curve_table,
            "word_accuracy/f1_by_threshold": wandb.plot.line(
                curve_table,
                "threshold",
                "f1",
                title="MOSCATO micro-F1 by global threshold",
            ),
        }
        if rows:
            columns = list(rows[0])
            log_items["evaluation/frame_predictions"] = wandb.Table(
                columns=columns,
                data=[[row[column] for column in columns] for row in rows],
            )
        run.log(log_items)

        artifact_name = _artifact_component("-".join(
            [
                "moscato-evaluation",
                _artifact_component(str(summary["video_id"])),
                _artifact_component(str(summary["object"])),
                resolved_name,
            ]
        ))
        artifact = wandb.Artifact(
            artifact_name,
            type="evaluation",
            metadata={
                "video_id": summary["video_id"],
                "object": summary["object"],
                "metrics": metrics,
                "ground_truth_used_for_inference": False,
            },
        )
        for artifact_key, path in artifact_paths.items():
            if path.is_file():
                artifact.add_file(str(path), name=f"{artifact_key}/{path.name}")
        logged_artifact = run.log_artifact(artifact, aliases=["latest"])
        result = {
            "run_url": run.url,
            "run_path": run.path,
            "artifact_name": logged_artifact.name,
        }
    return result
