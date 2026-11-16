import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from scripts import auto_initialize as auto


class AutoInitializeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.frame = self.root / "0000123.jpg"
        Image.new("RGB", (8, 6), "white").save(self.frame)
        self.checkpoint = self.root / "sam2.pt"
        self.checkpoint.write_bytes(b"mock model weights")
        self.mask_path = self.root / "mask.png"
        self.provenance_path = self.root / "initialization.json"

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def detection(box=(1, 1, 4, 4), score=0.8, text="bread"):
        return {"box_xyxy": list(box), "score": score, "text": text}

    @staticmethod
    def mask(x0, x1):
        mask = np.zeros((6, 8), dtype=bool)
        mask[1:4, x0:x1] = True
        return mask

    def initialize(self, detections, masks):
        with patch.object(auto, "detect_objects", return_value=detections) as detector, \
             patch.object(auto, "segment_boxes", return_value=masks) as segmenter:
            result = auto.initialize_mask(
                self.frame, "Bread slices", self.mask_path, self.provenance_path,
                sam2_checkpoint=self.checkpoint, device="cpu",
            )
        return result, detector, segmenter

    def test_union_retains_multiple_instances_and_saves_indexed_mask(self):
        left, right = self.mask(1, 3), self.mask(5, 7)
        result, detector, segmenter = self.initialize(
            [self.detection(), self.detection(box=(5, 1, 7, 4))],
            [(left, 0.9), (right, 0.85)],
        )
        with Image.open(self.mask_path) as saved:
            self.assertEqual(saved.mode, "P")
            self.assertEqual(saved.size, (8, 6))
            np.testing.assert_array_equal(np.asarray(saved), (left | right).astype(np.uint8))
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["mask_values"], [0, 1])
        self.assertEqual(result["mask_pixels"], 12)
        self.assertEqual(result["frame_sha256"], hashlib.sha256(self.frame.read_bytes()).hexdigest())
        self.assertEqual(result["mask_sha256"], hashlib.sha256(self.mask_path.read_bytes()).hexdigest())
        self.assertEqual(result["models"]["detector"]["revision"], auto.DETECTOR_REVISION)
        self.assertEqual(result["models"]["sam2"]["source_revision"], auto.SAM2_SOURCE_REVISION)
        self.assertTrue(all(row["accepted"] for row in result["detections"]))
        self.assertEqual(json.loads(self.provenance_path.read_text()), result)
        self.assertEqual(detector.call_args.args[1], "bread slices.")
        self.assertEqual(segmenter.call_args.args[1], [[1, 1, 4, 4], [5, 1, 7, 4]])

    def test_detection_geometry_and_confidence_are_validated_before_sam(self):
        result, _, segmenter = self.initialize(
            [self.detection(score=0.1), self.detection(box=(3, 2, 3, 5)),
             self.detection(box=(float("nan"), 1, 3, 5)), self.detection(text=""),
             self.detection(box=(-2, 1, 4, 9))],
            [(self.mask(1, 3), 0.9)],
        )
        self.assertEqual(segmenter.call_args.args[1], [[0, 1, 4, 6]])
        self.assertEqual([r["accepted"] for r in result["detections"]], [False] * 4 + [True])
        self.assertIsNone(result["detections"][2]["detector_box_xyxy"][0])

    def test_low_quality_mask_is_excluded_from_union(self):
        result, _, _ = self.initialize(
            [self.detection(), self.detection(box=(5, 1, 7, 4))],
            [(self.mask(1, 3), 0.2), (self.mask(5, 7), 0.8)],
        )
        self.assertEqual(result["mask_pixels"], 6)
        self.assertEqual(result["detections"][0]["rejection_reason"], "invalid_or_low_mask_score")
        self.assertTrue(result["detections"][1]["accepted"])

    def test_no_detection_records_failure_without_mask_or_later_frame_search(self):
        with patch.object(auto, "detect_objects", return_value=[]) as detector, \
             patch.object(auto, "segment_boxes") as segmenter:
            with self.assertRaises(auto.InitializationError) as error:
                auto.initialize_mask(self.frame, "onion", self.mask_path, self.provenance_path,
                                     sam2_checkpoint=self.checkpoint)
        detector.assert_called_once()
        segmenter.assert_not_called()
        self.assertFalse(self.mask_path.exists())
        self.assertEqual(json.loads(self.provenance_path.read_text())["status"], "failed")
        self.assertEqual(error.exception.provenance["frame_policy"], "exact_supplied_frame_no_temporal_search")

    def test_empty_full_nonfinite_and_wrong_size_masks_fail_without_fallback(self):
        invalid = [np.zeros((6, 8)), np.ones((6, 8)), np.full((6, 8), np.nan), np.zeros((2, 2))]
        for index, mask in enumerate(invalid):
            with self.subTest(index=index), \
                 patch.object(auto, "detect_objects", return_value=[self.detection()]), \
                 patch.object(auto, "segment_boxes", return_value=[(mask, 0.9)]):
                out = self.root / f"bad-{index}.png"
                with self.assertRaises(auto.InitializationError):
                    auto.initialize_mask(self.frame, "bread", out, self.root / f"bad-{index}.json",
                                         sam2_checkpoint=self.checkpoint)
                self.assertFalse(out.exists())

    def test_model_error_is_recorded_and_does_not_create_a_rectangle_mask(self):
        with patch.object(auto, "detect_objects", side_effect=RuntimeError("model unavailable")):
            with self.assertRaisesRegex(auto.InitializationError, "model unavailable"):
                auto.initialize_mask(self.frame, "bread", self.mask_path, self.provenance_path,
                                     sam2_checkpoint=self.checkpoint)
        self.assertFalse(self.mask_path.exists())
        self.assertIn("model unavailable", json.loads(self.provenance_path.read_text())["error"])

    def test_existing_mask_is_not_overwritten(self):
        self.mask_path.write_bytes(b"existing mask")
        with patch.object(auto, "detect_objects") as detector:
            with self.assertRaises(FileExistsError):
                auto.initialize_mask(self.frame, "bread", self.mask_path, self.provenance_path,
                                     sam2_checkpoint=self.checkpoint)
        detector.assert_not_called()
        self.assertEqual(self.mask_path.read_bytes(), b"existing mask")

    def test_invalid_thresholds_fail_before_loading_models(self):
        with patch.object(auto, "detect_objects") as detector:
            with self.assertRaisesRegex(ValueError, "finite"):
                auto.initialize_mask(self.frame, "bread", self.mask_path, self.provenance_path,
                                     sam2_checkpoint=self.checkpoint, min_mask_score=float("nan"))
        detector.assert_not_called()


if __name__ == "__main__":
    unittest.main()
