from contextlib import nullcontext
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import modal
import run_budgeted_video as runner


class BudgetedVideoTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.previous = Path.cwd()
        os.chdir(self.tmp.name)
        Path('weights_and_biases').mkdir()
        Path('weights_and_biases/settings.json').write_text(json.dumps({
            'entity': 'chefOST',
            'projects': {'full-evals': 'tubeletgraph - full evals'},
        }))
        self.specs = [{'video_id': 'v1', 'object_name': 'bowl', 'mask_name': 'masks/bowl.png',
                       'fps': 30, 'frame_stride': 6}]
        self.run = MagicMock()
        self.run.__enter__.return_value = self.run
        self.run.entity = 'chefOST'
        self.run.project = 'tubeletgraph - full evals'
        self.run.url = 'https://wandb.ai/chefOST/test/runs/example'
        self.run.path = 'chefOST/test/example'
        self.run.summary = {}
        self.volume = MagicMock()
        self.volume.iterdir.return_value = [SimpleNamespace(path='v1/masks/bowl.png')]
        self.gpu = MagicMock()
        self.call = self.gpu.with_options.return_value.spawn.return_value
        self.call.get.return_value = {'status': 'succeeded'}

    def tearDown(self):
        os.chdir(self.previous)
        self.tmp.cleanup()

    def invoke(self, times=(0, 1, 2)):
        argv = ['runner', '--batch-file', 'batch.json', '--project',
                'tubeletgraph - full evals', '--budget-usd', '70']
        with patch('sys.argv', argv), patch.object(runner, 'load_batch', return_value=self.specs), \
             patch.object(runner, 'data_volume', self.volume), patch.object(runner, 'run_tubelet', self.gpu), \
             patch.object(runner.wandb, 'init', return_value=self.run), \
             patch.object(runner.wandb, 'Artifact'), \
             patch.object(runner.app, 'run', return_value=nullcontext()), \
             patch.object(runner.modal, 'enable_output', return_value=nullcontext()), \
             patch.object(runner.time, 'monotonic', side_effect=times):
            runner.main()

    def test_wrong_destination_never_spawns_gpu(self):
        self.run.entity = 'personal-workspace'
        with self.assertRaisesRegex(RuntimeError, 'Unexpected W&B destination'):
            self.invoke()
        self.gpu.with_options.assert_not_called()

    def test_missing_mask_never_spawns_gpu(self):
        self.volume.iterdir.return_value = []
        with self.assertRaisesRegex(FileNotFoundError, 'Missing initial mask'):
            self.invoke()
        self.gpu.with_options.assert_not_called()

    def test_time_limit_cancels_container_and_marks_stopped(self):
        with self.assertRaisesRegex(RuntimeError, 'time-based budget limit'):
            self.invoke(times=(0, 7400, 7401))
        self.call.cancel.assert_called_with(terminate_containers=True)
        self.assertEqual(self.run.summary['inference/status'], 'stopped')
        self.assertEqual(self.run.summary['inference/completed_objects'], 0)

    def test_success_publishes_raw_artifacts_without_metric_claims(self):
        def files(path):
            if path == 'v1':
                return [SimpleNamespace(path='v1/masks/bowl.png')]
            return [SimpleNamespace(path=f'{path}/run_manifest.json', type=modal.volume.FileEntryType.FILE)]
        self.volume.iterdir.side_effect = files
        self.volume.read_file.return_value = [b'{"status": "succeeded"}']
        self.invoke()
        self.assertEqual(self.run.summary['inference/status'], 'completed')
        self.assertEqual(self.run.summary['inference/completed_objects'], 1)
        self.assertEqual(self.run.summary['evaluation/status'], 'not_scored')
        self.assertEqual(self.run.log_artifact.call_count, 2)
        self.assertEqual(len(list(Path('results').rglob('run_manifest.json'))), 1)


if __name__ == '__main__':
    unittest.main()
