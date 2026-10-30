from __future__ import annotations

import unittest

from scripts.log_wandb import metric_payload, run_tags


class WandbLoggingTests(unittest.TestCase):
    def test_primary_metrics_use_word_accuracy_namespace(self) -> None:
        metrics = metric_payload(
            {
                "precision": 0.7,
                "accuracy": 0.6,
                "f1": 0.65,
                "f1_max": 0.8,
                "f1_max_threshold": 0.4,
                "evaluated_frames": 10,
            }
        )
        self.assertEqual(
            set(metrics),
            {
                "word_accuracy/precision",
                "word_accuracy/accuracy",
                "word_accuracy/f1",
                "word_accuracy/f1_max",
                "word_accuracy/f1_max_threshold",
                "word_accuracy/evaluated_frames",
                "word_accuracy/coverage",
                "word_accuracy/abstained_frames",
            },
        )

    def test_missing_primary_metric_is_rejected(self) -> None:
        with self.assertRaises(KeyError):
            metric_payload({"precision": 1.0})

    def test_run_tags_describe_the_run(self) -> None:
        tags = run_tags(
            eval_kind="full-eval",
            video_id="S12_Sandwich_7150991-2470",
            object_name="bread slices",
            manifest={
                "fps": 5,
                "frame_stride": 6,
                "later_object_tracking": False,
                "adapter_input_mode": "tubelet_semantic_text_plus_single_event_frame",
                "tubeletgraph_commit": "fdb05b6fbd7f4644aea990bf967cc18d82bf291b",
                "smoke_test": False,
            },
            extra=["prompt v2", "full-eval"],
        )
        self.assertEqual(
            tags,
            [
                "tubeletgraph",
                "moscato",
                "cmu",
                "evaluation",
                "full-eval",
                "video:S12_Sandwich_7150991-2470",
                "session:S12",
                "object:bread-slices",
                "fps:5",
                "stride:6",
                "later-tracking:off",
                "adapter:tubelet_semantic_text_plus_single_event_frame",
                "tubeletgraph-commit:fdb05b6",
                "prompt-v2",
            ],
        )
        self.assertTrue(all(len(tag) <= 64 for tag in tags))


if __name__ == "__main__":
    unittest.main()
