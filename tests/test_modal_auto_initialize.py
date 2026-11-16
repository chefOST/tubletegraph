from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, Mock, patch

import modal_auto_initialize as worker


class ModalAutoInitializationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / "data"
        self.repository = self.root / "TubeletGraph"
        (self.repository / "configs").mkdir(parents=True)
        self.video = {
            "video_id": "v1", "dataset": "CMU", "r2_path": "CMU/v1.mp4",
            "video_sha256": "a" * 64, "source_fps": 10,
            "source_frame_count": 20, "annotation_frame_count": 20, "duration_seconds": 2,
            "frame_index_basis": "source_frame_indices",
        }
        self.ingredient = {
            "id": "ingredient1", "object_name": "egg", "status": "ready",
            "first_visible_annotation_index": 1, "seed_annotation_index": 2,
            "seed_source_frame": 2,
        }
        self.inputs = self.data / "auto_inputs" / "v1" / self.video["video_sha256"]
        (self.inputs / "frames").mkdir(parents=True)
        for index in range(20):
            (self.inputs / "frames" / f"{index:07d}.jpg").write_bytes(f"source-frame-{index}".encode())
        media = {**self.video, "timestamps_seconds": [index / 10 for index in range(20)],
                 "width": 3, "height": 2}
        (self.inputs / "source.json").write_text(json.dumps(media))
        self.prepared = {"root": str(self.inputs), "metadata": str(self.inputs / "source.json")}
        self.run_dir = self.data / "v1" / "runs" / "trial"
        self.volume = Mock()
        self.initialization_fails = False
        self.native_mode = "success"
        self.frame_bytes = []

    def command(self, command, log_path, *, env=None):
        log_path.write_text("captured model output\n")
        if command[1].endswith("auto_initialize.py"):
            frame = Path(command[command.index("--frame") + 1])
            self.assertEqual(frame.read_bytes(), b"source-frame-2")
            self.assertEqual(command[command.index("--object-name") + 1], "egg")
            provenance = Path(command[command.index("--provenance-out") + 1])
            provenance.write_text(json.dumps({"status": "failed" if self.initialization_fails else "succeeded",
                                              "frame": frame.name, "detections": []}))
            if self.initialization_fails:
                raise subprocess.CalledProcessError(1, command)
            Path(command[command.index("--mask-out") + 1]).write_bytes(b"indexed-mask-placeholder")
            return
        self.assertTrue(command[1].endswith("run_native_inference.py"))
        self.assertEqual(command[command.index("--fps") + 1], "5")
        self.assertEqual(env["TUBELETGRAPH_SHARE_SAM2"], "1")
        frames = Path(command[command.index("--input-dir") + 1])
        paths = sorted(frames.glob("*.jpg"))
        self.assertEqual([path.name for path in paths], [f"{i:07d}.jpg" for i in range(len(paths))])
        self.frame_bytes = [path.read_bytes() for path in paths]
        raw = self.repository / "_pred_out" / "custom-v1-Ours"
        raw.mkdir(parents=True)
        (raw / "v1_1.json").write_text(json.dumps({"prediction": {str(i): [] for i in range(len(paths))}}))
        (self.repository / "configs" / "custom_v1.yaml").write_text("dataset: custom\n")
        if self.native_mode == "failure":
            raise subprocess.CalledProcessError(1, command)
        semantic = self.repository / "_pred_out" / "custom-v1-Ours_gpt-4.1"
        semantic.mkdir()
        count = len(paths) - (self.native_mode == "wrong_count")
        (semantic / "v1_1.json").write_text(json.dumps({
            "prediction": {str(i): [] for i in range(count)}, "obj_info": {"0": {"desc": "raw egg"}},
        }))
        if self.native_mode != "missing_html":
            (semantic / "v1_1.html").write_text("<html>Native prompt transcript</html>")

    def invoke(self, *, measure_gpu=False):
        with patch.object(worker, "DATA_ROOT", self.data), \
                patch.object(worker, "REPOSITORY", self.repository), \
                patch.object(worker, "data_volume", self.volume), \
                patch.object(worker, "_command", side_effect=self.command) as command:
            result = worker.run_auto_initialized.local(
                self.video, self.ingredient, self.prepared, "trial",
                measure_gpu=measure_gpu,
            )
        self.commands = command.call_args_list
        self.manifest = json.loads((self.run_dir / "run_manifest.json").read_text())
        return result

    def test_saves_native_outputs_and_tracks_from_exact_seed_through_video_end(self):
        result = self.invoke()
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(len(self.commands), 2)
        self.assertEqual(self.frame_bytes, [f"source-frame-{i}".encode() for i in range(2, 20, 2)])
        mapping = json.loads((self.run_dir / "frames.json").read_text())["frames"]
        self.assertEqual([row["source_frame"] for row in mapping], list(range(2, 20, 2)))
        self.assertEqual(self.manifest["clip_policy"], "seed_through_video_end")
        self.assertEqual(self.manifest["sampled_frames"], 9)
        self.assertEqual(self.manifest["semantic_entries"], 1)
        self.assertTrue(self.manifest["gt_visibility_assisted"])
        self.assertFalse(self.manifest["ground_truth_state_values_used"])
        self.assertIn("started_at", self.manifest)
        self.assertIn("finished_at", self.manifest)
        self.assertNotIn("elapsed_seconds", self.manifest)
        self.assertNotIn("stages_seconds", self.manifest)
        self.assertNotIn("gpu_probe", self.manifest)
        self.assertFalse(self.manifest["measure_gpu"])
        self.assertFalse((self.run_dir / "gpu.csv").exists())
        for relative in ["initial_frame.jpg", "initial_mask.png", "initialization.json", "initialization.log",
                         "tubeletgraph.log", "input.json", "generated_configs/custom_v1.yaml",
                         "predictions/custom-v1-Ours/v1_1.json",
                         "predictions/custom-v1-Ours_gpt-4.1/v1_1.json",
                         "predictions/custom-v1-Ours_gpt-4.1/v1_1.html"]:
            self.assertTrue((self.run_dir / relative).is_file(), relative)
        self.volume.reload.assert_called_once()
        self.assertGreaterEqual(self.volume.commit.call_count, 3)

    def test_no_detection_preserves_failure_files_and_never_runs_tubeletgraph(self):
        self.initialization_fails = True
        result = self.invoke()
        self.assertEqual(result["status"], "initialization_failed")
        self.assertEqual(len(self.commands), 1)
        self.assertFalse((self.run_dir / "initial_mask.png").exists())
        self.assertFalse((self.run_dir / "tubeletgraph.log").exists())
        self.assertTrue((self.run_dir / "initialization.log").is_file())
        self.assertEqual(json.loads((self.run_dir / "initialization.json").read_text())["status"], "failed")
        self.assertIn("CalledProcessError", result["error"])
        self.assertEqual(self.manifest["status"], "initialization_failed")
        self.volume.commit.assert_called()

    def test_requested_gpu_probe_records_only_opt_in_outputs_and_named_stages(self):
        probe = MagicMock()
        probe.__enter__.return_value = probe
        probe.stage.side_effect = lambda name: nullcontext()
        probe.summary.return_value = {
            "samples": 2, "gpu_utilization_p95_pct": 80, "peak_memory_used_mib": 10000,
            "stages": {"initialization": {"elapsed_seconds": 1},
                       "native_inference": {"elapsed_seconds": 2}},
        }
        def close(*args):
            (self.run_dir / "gpu.csv").write_text("requested GPU samples\n")
            return False
        probe.__exit__.side_effect = close
        with patch.object(worker, "GpuProbe", return_value=probe) as create:
            result = self.invoke(measure_gpu=True)
        self.assertEqual(result["status"], "succeeded")
        create.assert_called_once_with(self.run_dir / "gpu.csv")
        self.assertEqual([call.args[0] for call in probe.stage.call_args_list],
                         ["initialization", "native_inference"])
        self.assertTrue(self.manifest["measure_gpu"])
        self.assertEqual(self.manifest["gpu_probe"], probe.summary.return_value)
        self.assertEqual(json.loads((self.run_dir / "gpu_summary.json").read_text()), probe.summary.return_value)

    def test_default_run_never_starts_gpu_probe(self):
        with patch.object(worker, "GpuProbe") as create:
            self.invoke()
        create.assert_not_called()

    def test_native_failure_preserves_partial_predictions_and_logs(self):
        self.native_mode = "failure"
        result = self.invoke()
        self.assertEqual(result["status"], "inference_failed")
        self.assertTrue((self.run_dir / "predictions/custom-v1-Ours/v1_1.json").is_file())
        self.assertTrue((self.run_dir / "tubeletgraph.log").is_file())
        self.assertTrue((self.run_dir / "initial_mask.png").is_file())
        self.assertTrue((self.run_dir / "initialization.json").is_file())
        self.assertIn("CalledProcessError", self.manifest["error"])

    def test_missing_native_transcript_is_not_reported_as_success(self):
        self.native_mode = "missing_html"
        result = self.invoke()
        self.assertEqual(result["status"], "inference_failed")
        self.assertIn("prompt transcript is missing", result["error"])

    def test_incomplete_native_frame_predictions_are_not_reported_as_success(self):
        self.native_mode = "wrong_count"
        result = self.invoke()
        self.assertEqual(result["status"], "inference_failed")
        self.assertIn("frame count does not match", result["error"])

    def test_publisher_preserves_native_artifacts_and_disables_system_stats(self):
        import wandb

        result = self.invoke()
        expected_artifacts = {path.relative_to(self.run_dir).as_posix(): path.read_bytes()
                              for path in self.run_dir.rglob("*") if path.is_file()}
        settings_path = self.root / "settings.json"
        settings_path.write_text(json.dumps({
            "entity": "chefOST", "projects": {"experiments": "tubeletgraph-experiments"},
        }))
        run = MagicMock()
        run.__enter__.return_value = run
        run.entity, run.project = "chefOST", "tubeletgraph-experiments"
        run.url = "https://wandb.ai/chefOST/tubeletgraph-experiments/runs/test"
        run.summary = {}
        artifact = Mock()
        uploaded = {}
        artifact.add_file.side_effect = lambda path, name: uploaded.update({name: Path(path).read_bytes()})

        def resolve_path(path):
            return settings_path if path == "/opt/wandb-settings.json" else Path(path)

        with patch.object(worker, "DATA_ROOT", self.data), \
                patch.object(worker, "Path", side_effect=resolve_path), \
                patch.object(worker, "data_volume", self.volume), \
                patch.object(wandb, "init", return_value=run) as init, \
                patch.object(wandb, "Artifact", return_value=artifact):
            worker.publish_auto_result.local(self.video, self.ingredient, result, "trial", {})
        self.assertTrue(init.call_args.kwargs["settings"].x_disable_stats)
        for name, contents in expected_artifacts.items():
            self.assertEqual(uploaded[name], contents)
        published_manifest = json.loads(uploaded["run_manifest.json"])
        self.assertEqual(published_manifest, self.manifest)
        self.assertIn("result.json", uploaded)


if __name__ == "__main__":
    unittest.main()
