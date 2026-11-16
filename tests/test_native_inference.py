from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts import run_native_inference as runner


class NativeInferenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.frames = Path(self.tmp.name) / "frames"
        self.frames.mkdir()
        self.mask = Path(self.tmp.name) / "mask.png"
        self.mask.touch()
        self.argv = ["--input-dir", str(self.frames), "--input-mask", str(self.mask),
                     "--fps", "5"]
        self.build = Mock(return_value=("configs/custom_clip.yaml", "clip"))

    def invoke(self, process):
        with patch.object(runner, "import_module", return_value=SimpleNamespace(
            create_custom_dataset=self.build
        )) as load, patch.object(runner.subprocess, "run", process), \
                patch.object(sys, "path", list(sys.path)):
            runner.main(self.argv)
            load.assert_called_once_with("eval.build_custom_dataset")
            self.assertEqual(sys.path[0], str(Path.cwd()))

    def test_runs_native_pipeline_with_gpt_and_without_preview_renderer(self):
        process = Mock()
        self.invoke(process)
        self.build.assert_called_once_with(
            "configs/default.yaml", str(self.frames), str(self.mask), fps=5
        )
        process.assert_called_once_with(
            [sys.executable, "TubeletGraph/run.py", "-c", "configs/custom_clip.yaml",
             "-d", "custom", "-s", "clip", "-m", "Ours"], check=True,
        )
        self.assertNotIn("--skip_vlm", process.call_args.args[0])

    def test_native_pipeline_failure_propagates(self):
        failure = subprocess.CalledProcessError(1, "TubeletGraph/run.py")
        with self.assertRaises(subprocess.CalledProcessError) as caught:
            self.invoke(Mock(side_effect=failure))
        self.assertIs(caught.exception, failure)


if __name__ == "__main__":
    unittest.main()
