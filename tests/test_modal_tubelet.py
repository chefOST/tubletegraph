from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from modal_tubelet import _run_name, _safe_component, load_batch


class ModalTubeletTests(unittest.TestCase):
    def test_safe_components_accept_expected_ids(self) -> None:
        self.assertEqual(
            _safe_component("S12_Sandwich_7150991-2470", "video_id"),
            "S12_Sandwich_7150991-2470",
        )

    def test_safe_components_reject_path_traversal(self) -> None:
        with self.assertRaises(ValueError):
            _safe_component("../overwrite", "run_name")

    def test_explicit_run_name_is_preserved(self) -> None:
        self.assertEqual(_run_name("trial-001"), "trial-001")


    def _batch(self, payload: dict) -> list[dict]:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "batch.json"
            path.write_text(json.dumps(payload))
            return load_batch(str(path))

    def test_batch_merges_defaults_and_disables_frame_count_check(self) -> None:
        specs = self._batch({
            "defaults": {"fps": 30, "frame_stride": 6, "dataset": "CMU"},
            "videos": [
                {"video_id": "S12_Sandwich_7150991-2470", "object_name": "bread slices",
                 "r2_path": "CMU/S12.mp4", "clip_start_global": 2824},
                {"video_id": "S07_Brownie", "object_name": "egg", "fps": 15, "frame_stride": 3},
            ],
        })
        self.assertEqual(specs[0]["frame_stride"], 6)
        self.assertEqual(specs[0]["expected_frames"], 0)
        self.assertEqual(specs[1]["fps"], 15)
        self.assertEqual(specs[0]["mask_name"], "masks/bread-slices.png")

    def test_batch_rejects_unknown_keys_duplicates_and_unsafe_ids(self) -> None:
        base = {"video_id": "v1", "object_name": "egg"}
        with self.assertRaises(ValueError):
            self._batch({"videos": [{**base, "smoke_test": True}]})
        with self.assertRaises(ValueError):
            self._batch({"videos": [base, base]})
        with self.assertRaises(ValueError):
            self._batch({"videos": [{"video_id": "../x", "object_name": "egg"}]})
        with self.assertRaises(ValueError):
            self._batch({"videos": []})


if __name__ == "__main__":
    unittest.main()
