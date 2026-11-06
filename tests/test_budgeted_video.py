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

    def invoke(self, times=(0, 1, 2), extra_args=()):
        argv = ['runner', '--batch-file', 'batch.json', '--project',
                'tubeletgraph - full evals', '--budget-usd', '70', *extra_args]
        with patch('sys.argv', argv), patch.object(runner, 'load_batch', return_value=self.specs), \
             patch.object(runner, 'data_volume', self.volume), patch.object(runner, 'run_tubelet', self.gpu), \
             patch.object(runner.wandb, 'init', return_value=self.run), \
             patch.object(runner.wandb, 'Artifact'), \
             patch.object(runner.app, 'run', return_value=nullcontext()), \
             patch.object(runner.modal, 'enable_output', return_value=nullcontext()), \
             patch.object(runner.time, 'monotonic', side_effect=times):
            runner.main()
            self.init_kwargs = runner.wandb.init.call_args.kwargs

    def test_resume_skips_completed_objects_and_reuses_wandb_id(self):
        state = Path('saved')
        state.mkdir()
        (state / 'run.json').write_text(json.dumps({
            'video_id': 'v1', 'run_name': 'previous',
            'wandb_path': 'chefOST/tubeletgraph - full evals/example',
        }))
        (state / 'objects.json').write_text(json.dumps([
            {'object': 'bowl', 'run_name': 'previous-bowl', 'status': 'succeeded'},
        ]))
        self.invoke(times=(), extra_args=['--resume-dir', str(state),
                                        '--budget-usd', '39', '--prior-compute-usd', '3'])
        self.gpu.with_options.return_value.spawn.assert_not_called()
        kwargs = self.init_kwargs
        self.assertEqual(kwargs['id'], 'example')
        self.assertEqual(kwargs['resume'], 'must')
        self.assertEqual(kwargs['config']['compute_allowance_usd'], 21)
        self.assertEqual(self.run.summary['inference/completed_objects'], 1)
        self.assertEqual(self.run.summary['budget/estimated_compute_usd'], 3)

    def test_prior_spend_exhausts_budget_before_gpu_launch(self):
        with self.assertRaisesRegex(ValueError, 'Budget cannot accommodate'):
            self.invoke(extra_args=['--budget-usd', '39', '--prior-compute-usd', '25'])
        self.gpu.with_options.assert_not_called()

    def test_remote_coordinator_publishes_and_refreshes_volume(self):
        with patch.object(runner.modal, 'is_local', return_value=False):
            self.test_success_publishes_raw_artifacts_without_metric_claims()
        self.volume.reload.assert_called_once()
        self.assertGreaterEqual(self.volume.commit.call_count, 2)

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
