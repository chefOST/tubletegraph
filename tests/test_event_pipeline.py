from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from scripts.build_event_draft import build_draft
from scripts.map_event_states import (
    canonical_states,
    map_events,
    normalize_candidate,
    semantic_evidence,
)


class EventPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {
            "video_id": "S12_Sandwich_7150991-2470",
            "target_object": "bread slices",
            "clip_start_global": 2824,
            "clip_end_global": 2834,
        }

    def test_draft_consolidates_nodes_at_the_same_frame(self) -> None:
        prediction = {
            "obj_info": {
                "0": {"desc": "bread slices"},
                "2": {
                    "desc": "bread piece",
                    "prior_desc": "bread slice",
                    "action": "spread",
                    "analysis_frame_idx": 5,
                    "object_start_frame_idx": 4,
                },
                "1": {
                    "desc": "bread piece",
                    "prior_desc": "bread slice",
                    "action": "spread",
                    "analysis_frame_idx": 4,
                    "object_start_frame_idx": 4,
                },
            }
        }
        draft = build_draft(prediction, self.config)
        self.assertEqual(draft["initial_state"], None)
        self.assertEqual(len(draft["transitions"]), 1)
        self.assertEqual(draft["transitions"][0]["local_frame"], 4)
        self.assertEqual(
            [
                node["tubelet_object_id"]
                for node in draft["transitions"][0]["raw_nodes"]
            ],
            ["1", "2"],
        )

    def test_candidate_normalizes_known_alias(self) -> None:
        state_dict = {
            "s2i": {"on dish": 0, "topped": 1},
            "all2one": {"on plate": "on dish"},
        }
        self.assertEqual(
            normalize_candidate("on plate", state_dict, ["on dish", "topped"]),
            "on dish",
        )

    def test_candidate_accepts_json_wrapper(self) -> None:
        state_dict = {"s2i": {"on dish": 0}, "all2one": {}}
        self.assertEqual(
            normalize_candidate('{"state": "on dish"}', state_dict, ["on dish"]),
            "on dish",
        )

    def test_object_candidates_are_validated_and_normalized(self) -> None:
        state_dict = {
            "s2i": {"on dish": 0, "placed": 1, "unrelated": 2},
            "all2one": {},
        }
        self.assertEqual(
            canonical_states(state_dict, ["Placed", "on dish"]),
            ["on dish", "placed"],
        )
        with self.assertRaises(ValueError):
            canonical_states(state_dict, ["invented"])

    def test_missing_tubelet_text_abstains_without_model_call(self) -> None:
        class RefusingClient:
            @property
            def chat(self):
                raise AssertionError("Model must not be called without semantic text")

        draft = {
            "object": "bread slices",
            "num_local_frames": 3,
            "initial_raw": {"description": None},
            "transitions": [],
            "provenance": {},
        }
        state_dict = {"s2i": {"on dish": 0}, "all2one": {}, "one2all": {}}
        mapped = map_events(
            draft,
            state_dict,
            Path("unused"),
            RefusingClient(),
            "unused",
            allowed_states=["on dish"],
        )
        self.assertIsNone(mapped["initial_state"])
        self.assertEqual(
            mapped["initial_mapping"]["status"],
            "abstained_no_tubelet_text",
        )
        self.assertEqual(
            mapped["provenance"]["state_mapping_status"],
            "abstained_no_tubelet_text",
        )
        self.assertFalse(mapped["provenance"]["video_frames_used_for_mapping"])

    def test_adapter_uses_image_and_text_without_identifiers_or_frame_numbers(self) -> None:
        calls = []

        class Completions:
            def create(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="on dish"))]
                )

        client = SimpleNamespace(
            chat=SimpleNamespace(completions=Completions())
        )
        draft = {
            "video_id": "S12_Sandwich_7150991-2470",
            "object": "bread slices",
            "clip_start_global": 2824,
            "num_local_frames": 3,
            "initial_raw": {"description": "bread resting on a dish"},
            "transitions": [],
            "provenance": {},
        }
        state_dict = {
            "s2i": {"on dish": 0, "placed": 1},
            "all2one": {},
            "one2all": {"on dish": ["on dish", "on plate"], "placed": ["placed"]},
        }
        with TemporaryDirectory() as directory:
            frames_dir = Path(directory)
            (frames_dir / "0000000.jpg").write_bytes(b"synthetic-jpeg")
            mapped = map_events(
                draft,
                state_dict,
                frames_dir,
                client,
                "gpt-4.1",
                allowed_states=["on dish", "placed"],
            )

        self.assertEqual(mapped["initial_state"], "on dish")
        content = calls[0]["messages"][1]["content"]
        prompt = content[0]["text"]
        self.assertNotIn("S12_Sandwich_7150991-2470", prompt)
        self.assertNotIn("2824", prompt)
        self.assertNotIn("0000000", prompt)
        self.assertIn("bread resting on a dish", prompt)
        self.assertEqual(content[1]["type"], "image_url")
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))
        self.assertEqual(content[1]["image_url"]["detail"], "low")
        self.assertTrue(mapped["provenance"]["video_frames_used_for_mapping"])
        self.assertFalse(mapped["provenance"]["video_identifiers_used_in_prompt"])
        self.assertFalse(mapped["provenance"]["frame_numbers_used_in_prompt"])

    def test_adapter_prompt_hides_aliases_and_specificity_bonus(self) -> None:
        calls = []

        class Completions:
            def create(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content=(
                                    '{"observation": "two plain slices on a plate", '
                                    '"state": "on dish"}'
                                )
                            )
                        )
                    ]
                )

        client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
        draft = {
            "video_id": "S12_Sandwich_7150991-2470",
            "object": "bread slices",
            "clip_start_global": 2824,
            "num_local_frames": 3,
            "initial_raw": {"description": "Stacked sandwich bread"},
            "transitions": [],
            "provenance": {},
        }
        state_dict = {
            "s2i": {"on dish": 0, "pressed-together": 1},
            "all2one": {"sandwiched": "pressed-together"},
            "one2all": {
                "on dish": ["on dish", "on plate", "plated"],
                "pressed-together": ["pressed-together", "sandwiched", "pressed"],
            },
        }
        definitions = {
            "on dish": "plain slices resting on a plate",
            "pressed-together": "two slices closed face to face",
        }
        with TemporaryDirectory() as directory:
            frames_dir = Path(directory)
            (frames_dir / "0000000.jpg").write_bytes(b"synthetic-jpeg")
            mapped = map_events(
                draft,
                state_dict,
                frames_dir,
                client,
                "gpt-4.1",
                allowed_states=["on dish", "pressed-together"],
                visual_definitions=definitions,
            )

        prompt = calls[0]["messages"][1]["content"][0]["text"]
        # Change 1: alias lists never reach the model.
        self.assertNotIn("sandwiched", prompt)
        self.assertNotIn("plated", prompt)
        self.assertNotIn("ALIASES", prompt)
        self.assertIn("two slices closed face to face", prompt)
        # Change 3: the specificity bonus clause is gone.
        self.assertNotIn("more specific", prompt)
        # Change 5: observation is requested and recorded.
        self.assertIn('"observation"', prompt)
        self.assertEqual(mapped["initial_state"], "on dish")
        self.assertEqual(
            mapped["initial_mapping"]["observation"], "two plain slices on a plate"
        )
        self.assertFalse(mapped["provenance"]["adapter_prompt_shows_aliases"])

    def test_semantic_evidence_ignores_nonsemantic_metadata(self) -> None:
        self.assertEqual(
            semantic_evidence(
                {
                    "tubelet_object_id": "2",
                    "raw_nodes": [
                        {"description": "bread on a plate", "analysis_frame_idx": 4}
                    ],
                }
            ),
            ["bread on a plate"],
        )


if __name__ == "__main__":
    unittest.main()
