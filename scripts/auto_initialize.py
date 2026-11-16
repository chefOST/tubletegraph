"""Initialize one object category on one exact frame with Grounding DINO and SAM2."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
from typing import Any

DETECTOR_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
DETECTOR_REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71"
TRANSFORMERS_VERSION = "4.56.1"
SAM2_SOURCE_REVISION = "fb5e452074cd8bf2da3e2d9b4108e480b7f07276"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CHECKPOINT = Path("/opt/TubeletGraph/_ckpts/sam2.1_hiera_large.pt")


class InitializationError(RuntimeError):
    """The supplied frame did not produce a usable mask; provenance is saved."""

    def __init__(self, message: str, provenance: dict[str, Any]):
        super().__init__(message)
        self.provenance = provenance


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _threshold(value: float, name: str) -> float:
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return value


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def detect_objects(image: Image.Image, prompt: str, *, device: str,
                   box_threshold: float, text_threshold: float) -> list[dict[str, Any]]:
    """Run the pinned detector locally; no hosted inference endpoint is used."""
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing a CPU fallback")
    processor = AutoProcessor.from_pretrained(
        DETECTOR_MODEL_ID, revision=DETECTOR_REVISION, use_fast=False,
    )
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        DETECTOR_MODEL_ID, revision=DETECTOR_REVISION,
        use_safetensors=True, disable_custom_kernels=True,
    ).to(device).eval()
    try:
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = model(**inputs)
        result = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=box_threshold,
            text_threshold=text_threshold, target_sizes=[(image.height, image.width)],
        )[0]
        return [
            {"box_xyxy": box.detach().cpu().tolist(), "score": float(score), "text": text}
            for box, score, text in zip(result["boxes"], result["scores"], result["text_labels"])
        ]
    finally:
        # SAM2 runs afterward, so these models do not need to coexist on the GPU.
        del model
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


def segment_boxes(image: Image.Image, boxes: list[list[float]], *, device: str,
                  sam2_checkpoint: Path, sam2_config: str) -> list[tuple[np.ndarray, float]]:
    """Embed the frame once and predict one binary SAM2 mask per box."""
    import numpy as np
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model = build_sam2(sam2_config, str(sam2_checkpoint), device=device, mode="eval")
    predictor = SAM2ImagePredictor(model)
    try:
        with torch.inference_mode():
            predictor.set_image(np.asarray(image))
            predictions = []
            for box in boxes:
                masks, scores, _ = predictor.predict(
                    box=np.asarray(box, dtype=np.float32), multimask_output=False,
                    return_logits=False,
                )
                predictions.append((masks[0], float(scores[0])))
        return predictions
    finally:
        del predictor, model
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


def validate_detections(detections: list[dict[str, Any]], image_size: tuple[int, int],
                        box_threshold: float) -> list[dict[str, Any]]:
    """Keep all category detections passing confidence and geometry checks."""
    width, height = image_size
    records = []
    for index, detection in enumerate(detections):
        score = _number(detection.get("score"))
        box = [_number(x) for x in detection.get("box_xyxy", [])]
        text = str(detection.get("text", "")).strip()
        record = {"detection_index": index, "detector_box_xyxy": box,
                  "score": score, "text": text, "accepted": False}
        reason = None
        if score is None or not 0 <= score <= 1 or score < box_threshold:
            reason = "invalid_or_low_detection_score"
        elif not text:
            reason = "empty_detection_label"
        elif len(box) != 4 or any(x is None for x in box):
            reason = "invalid_box"
        else:
            x0, y0, x1, y1 = box
            clipped = [max(0.0, min(width, x0)), max(0.0, min(height, y0)),
                       max(0.0, min(width, x1)), max(0.0, min(height, y1))]
            record["box_xyxy"] = clipped
            if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
                reason = "empty_box"
        record["rejection_reason"] = reason
        records.append(record)
    return records


def union_masks(records: list[dict[str, Any]], predictions: list[tuple[np.ndarray, float]],
                image_size: tuple[int, int], min_mask_score: float) -> np.ndarray:
    """Union accepted instances of the requested category into object ID 1."""
    import numpy as np

    candidates = [record for record in records if record["rejection_reason"] is None]
    if len(candidates) != len(predictions):
        raise ValueError("SAM2 returned a different number of masks than input boxes")
    width, height = image_size
    combined = np.zeros((height, width), dtype=bool)
    for record, (mask, score) in zip(candidates, predictions):
        score = _number(score)
        record["mask_score"] = score
        mask = np.asarray(mask)
        reason = None
        if score is None or not 0 <= score <= 1 or score < min_mask_score:
            reason = "invalid_or_low_mask_score"
        elif mask.shape != combined.shape:
            reason = "mask_shape_mismatch"
        elif not np.all(np.isfinite(mask)) or not np.all((mask == 0) | (mask == 1)):
            reason = "invalid_binary_mask"
        else:
            mask = mask.astype(bool)
            pixels = int(np.count_nonzero(mask))
            record["mask_pixels"] = pixels
            if pixels == 0:
                reason = "empty_mask"
            elif pixels == mask.size:
                reason = "full_frame_mask"
            else:
                record["accepted"] = True
                combined |= mask
        record["rejection_reason"] = reason
    return combined


def initialize_mask(frame_path: Path, object_name: str, mask_path: Path,
                    provenance_path: Path, *, sam2_checkpoint: Path = SAM2_CHECKPOINT,
                    sam2_config: str = SAM2_CONFIG, device: str = "cuda",
                    box_threshold: float = 0.35, text_threshold: float = 0.25,
                    min_mask_score: float = 0.5) -> dict[str, Any]:
    """Save an indexed mask and provenance; never search other frames or use GT states."""
    import numpy as np
    from PIL import Image

    frame_path, mask_path, provenance_path = map(Path, (frame_path, mask_path, provenance_path))
    sam2_checkpoint = Path(sam2_checkpoint)
    if mask_path.resolve() == provenance_path.resolve():
        raise ValueError("Mask and provenance paths must differ")
    if mask_path.exists() or provenance_path.exists():
        raise FileExistsError("Initialization outputs already exist; choose new output paths")
    if mask_path.suffix.lower() != ".png":
        raise ValueError("Mask output must be a PNG")
    object_name = " ".join(object_name.split()).strip(" .")
    if not object_name or "." in object_name:
        raise ValueError("Provide one nonempty object category, not a list of prompts")
    thresholds = {"box": _threshold(box_threshold, "box_threshold"),
                  "text": _threshold(text_threshold, "text_threshold"),
                  "mask_score": _threshold(min_mask_score, "min_mask_score")}
    if not sam2_checkpoint.is_file():
        raise FileNotFoundError(sam2_checkpoint)
    with Image.open(frame_path) as source:
        image = source.convert("RGB")
    provenance: dict[str, Any] = {
        "schema_version": 1, "status": "running", "object_name": object_name,
        "text_prompt": object_name.lower() + ".", "image_size": list(image.size),
        "input_frame_filename": frame_path.name, "frame_sha256": file_sha256(frame_path),
        "policy": "union_all_accepted_detections_of_one_object_category",
        "frame_policy": "exact_supplied_frame_no_temporal_search",
        "thresholds": thresholds, "detections": [],
        "models": {
            "detector": {"id": DETECTOR_MODEL_ID, "revision": DETECTOR_REVISION,
                         "transformers_version": TRANSFORMERS_VERSION,
                         "disable_custom_kernels": True, "dtype": "float32"},
            "sam2": {"config": sam2_config, "checkpoint_sha256": file_sha256(sam2_checkpoint),
                     "source_revision": SAM2_SOURCE_REVISION, "multimask_output": False,
                     "dtype": "float32"},
        },
    }
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        detections = detect_objects(image, provenance["text_prompt"], device=device,
                                    box_threshold=box_threshold, text_threshold=text_threshold)
        records = validate_detections(detections, image.size, box_threshold)
        provenance["detections"] = records
        boxes = [record["box_xyxy"] for record in records if record["rejection_reason"] is None]
        if not boxes:
            raise ValueError("No valid detections on the supplied frame")
        predictions = segment_boxes(image, boxes, device=device,
                                    sam2_checkpoint=sam2_checkpoint, sam2_config=sam2_config)
        combined = union_masks(records, predictions, image.size, min_mask_score)
        pixels = int(np.count_nonzero(combined))
        if pixels == 0 or pixels == combined.size:
            raise ValueError("No usable foreground/background mask on the supplied frame")
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        output = Image.frombytes("P", image.size, combined.astype(np.uint8).tobytes())
        output.putpalette([channel for index in range(256) for channel in (index, index, index)])
        output.save(mask_path, format="PNG")
        provenance.update(status="succeeded", mask_pixels=pixels, object_id=1,
                          mask_sha256=file_sha256(mask_path), mask_values=[0, 1])
    except Exception as error:
        provenance.update(status="failed", error=f"{type(error).__name__}: {error}")
        provenance_path.write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
        raise InitializationError(str(error), provenance) from error
    provenance_path.write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", type=Path, required=True)
    parser.add_argument("--object-name", required=True)
    parser.add_argument("--mask-out", type=Path, required=True)
    parser.add_argument("--provenance-out", type=Path, required=True)
    parser.add_argument("--sam2-checkpoint", type=Path, default=SAM2_CHECKPOINT)
    parser.add_argument("--sam2-config", default=SAM2_CONFIG)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--box-threshold", type=float, default=0.35)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--min-mask-score", type=float, default=0.5)
    args = parser.parse_args()
    result = initialize_mask(args.frame, args.object_name, args.mask_out, args.provenance_out,
                             sam2_checkpoint=args.sam2_checkpoint, sam2_config=args.sam2_config,
                             device=args.device, box_threshold=args.box_threshold,
                             text_threshold=args.text_threshold, min_mask_score=args.min_mask_score)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
