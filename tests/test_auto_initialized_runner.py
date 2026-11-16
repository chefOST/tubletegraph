from contextlib import nullcontext, redirect_stderr, redirect_stdout
import copy
import io
import itertools
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts import run_auto_initialized as runner


class AutoInitializedRunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.plan = {"provenance": {"alignment_policy": "infer_from_duration"}, "videos": [{
            "video_id": "v1", "dataset": "CMU", "r2_path": "CMU/v1.mp4",
            "video_sha256": "a" * 64, "source_fps": 30, "source_frame_count": 300,
            "duration_seconds": 10, "annotation_frame_count": 300,
            "alignment": {"status": "inferred_from_duration"}, "ingredients": [
                {"id": "a", "object_name": "egg", "seed_source_frame": 5,
                 "first_visible_annotation_index": 4, "seed_annotation_index": 5,
                 "status": "ready", "skip_reason": None},
                {"id": "b", "object_name": "bread", "seed_source_frame": 11,
                 "first_visible_annotation_index": 10, "seed_annotation_index": 11,
                 "status": "ready", "skip_reason": None},
            ],
        }]}
        self.path = self.root / "plan.json"
        self.path.write_text(json.dumps(self.plan))
        self.runtime = SimpleNamespace(
            app=Mock(), prepare_auto_video=Mock(), run_auto_initialized=Mock(),
            publish_auto_result=Mock(),
        )
        self.runtime.app.run.return_value = nullcontext()
        self.runtime.prepare_auto_video.remote.return_value = {"frames": "cloud"}
        self.call = Mock()
        self.call.get.return_value = {"status": "succeeded"}
        self.runtime.run_auto_initialized.with_options.return_value.spawn.return_value = self.call
        self.runtime.publish_auto_result.remote.return_value = {"wandb_url": "https://wandb.ai/example"}

    def execute(self, **kwargs):
        with patch.object(runner, "import_module", return_value=self.runtime), \
                patch.object(runner.time, "monotonic", side_effect=itertools.count()):
            return runner.execute_plan(self.plan, budget_usd=10, reserve_usd=2,
                                       max_job_seconds=60, output_dir=self.root / "progress", **kwargs)

    def test_preview_makes_no_cloud_calls_or_progress_files(self):
        with patch.object(runner, "import_module") as cloud, redirect_stdout(io.StringIO()) as output:
            runner.main(["--plan", str(self.path)])
        cloud.assert_not_called()
        self.assertEqual(json.loads(output.getvalue())["ready"], 2)
        self.assertEqual(list(self.root.iterdir()), [self.path])

    def test_execute_requires_explicit_budget_before_cloud_import(self):
        with patch.object(runner, "import_module") as cloud, self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
            runner.main(["--plan", str(self.path), "--execute"])
        cloud.assert_not_called()

    def test_too_small_budget_never_imports_cloud(self):
        with patch.object(runner, "import_module") as cloud, self.assertRaises(ValueError):
            runner.execute_plan(self.plan, budget_usd=2.01, reserve_usd=2,
                                max_job_seconds=60, output_dir=self.root / "progress")
        cloud.assert_not_called()
        self.assertFalse((self.root / "progress").exists())

    def test_prepares_once_runs_sequentially_and_publishes_raw_outcomes(self):
        events = []
        self.runtime.prepare_auto_video.remote.side_effect = lambda **kw: events.append("prepare") or {"frames": "cloud"}
        self.runtime.run_auto_initialized.with_options.return_value.spawn.side_effect = lambda **kw: events.append("spawn") or self.call
        self.call.get.side_effect = lambda **kw: events.append("finish") or {"status": "succeeded"}
        self.runtime.publish_auto_result.remote.side_effect = lambda **kw: events.append("publish") or {"wandb_url": "https://wandb.ai/example"}
        result = self.execute()
        self.assertEqual(events, ["prepare", "spawn", "finish", "publish", "spawn", "finish", "publish"])
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["provenance"]["gt_visibility_assisted"])
        self.assertFalse(result["provenance"]["ground_truth_states_used"])
        self.assertEqual(set(path.name for path in (self.root / "progress").iterdir()), {"progress.json"})
        options = self.runtime.run_auto_initialized.with_options.call_args.kwargs
        self.assertEqual(options["retries"], 0)
        self.assertEqual(options["max_containers"], 1)

    def test_initialization_failure_is_published_and_next_ingredient_runs(self):
        self.call.get.side_effect = [{"status": "initialization_failed", "error": "No accepted masks"},
                                     {"status": "succeeded"}]
        result = self.execute()
        self.assertEqual(result["status"], "completed_with_failures")
        self.assertEqual(result["jobs"]["a"]["status"], "initialization_failed")
        self.assertEqual(self.runtime.publish_auto_result.remote.call_count, 2)

    def test_prepare_failure_is_published_without_gpu_work(self):
        self.runtime.prepare_auto_video.remote.side_effect = ValueError("Invalid video hash")
        result = self.execute()
        self.runtime.run_auto_initialized.with_options.assert_not_called()
        self.assertEqual(self.runtime.publish_auto_result.remote.call_count, 2)
        self.assertTrue(all(record["status"] == "prepare_failed" for record in result["jobs"].values()))

    def test_budget_timeout_cancels_gpu_and_returns_publishable_failure(self):
        with patch.object(runner.time, "monotonic", side_effect=[0, 181, 182]):
            result, elapsed, fatal = runner._run_job(
                self.runtime, self.plan["videos"][0], self.plan["videos"][0]["ingredients"][0],
                {}, "trial", 60, 300,
            )
        self.call.cancel.assert_called_once_with(terminate_containers=True)
        self.call.get.assert_not_called()
        self.assertEqual(result["status"], "timed_out")
        self.assertEqual(result["volume_path"], "/data/v1/runs/trial")
        self.assertEqual(elapsed, 182)
        self.assertIsNone(fatal)

    def test_transport_failure_preserves_expected_artifact_location(self):
        self.call.get.side_effect = RuntimeError("Connection interrupted")
        result = self.execute()
        self.assertEqual(result["jobs"]["a"]["status"], "failed")
        self.assertIn("/data/v1/runs/auto-", result["jobs"]["a"]["volume_path"])
        self.assertEqual(self.runtime.publish_auto_result.remote.call_count, 2)

    def test_cancellation_failure_does_not_lose_job_failure_record(self):
        self.call.get.side_effect = RuntimeError("Connection interrupted")
        self.call.cancel.side_effect = RuntimeError("Cancellation unavailable")
        with self.assertRaisesRegex(RuntimeError, "cancellation could not be confirmed"):
            self.execute()
        result = json.loads((self.root / "progress" / "progress.json").read_text())
        self.assertIn("Cancellation unavailable", result["jobs"]["a"]["cancellation_error"])
        self.assertEqual(self.runtime.publish_auto_result.remote.call_count, 1)

    def test_timestamp_seed_is_allowed_for_pts_lookup(self):
        plan = copy.deepcopy(self.plan)
        plan["videos"][0]["ingredients"][0].update(seed_source_frame=None, seed_timestamp_seconds=0.5)
        self.path.write_text(json.dumps(plan))
        self.assertEqual(runner.load_plan(self.path), plan)


if __name__ == "__main__":
    unittest.main()
