from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import modal
import run_budgeted_video as runner
import resume_budgeted_video as resumer


class BudgetedVideoTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.previous = Path.cwd()
        os.chdir(self.tmp.name)
        Path('weights_and_biases').mkdir()
        Path('weights_and_biases/settings.json').write_text(json.dumps({
            'entity': 'chefOST',
            'projects': {'experiments': 'tubeletgraph-experiments',
                         'full-evals': 'tubeletgraph - full evals'},
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
        self.volume.iterdir.side_effect = lambda path: [SimpleNamespace(path='v1/masks/bowl.png')] if path == 'v1' else []
        self.gpu = MagicMock()
        self.call = self.gpu.with_options.return_value.spawn.return_value
        self.call.get.return_value = {'status': 'succeeded'}

    def tearDown(self):
        os.chdir(self.previous)
        self.tmp.cleanup()

    def invoke(self, times=(0, 1, 2), extra_args=(), project='tubeletgraph - full evals'):
        argv = ['runner', '--batch-file', 'batch.json', '--budget-usd', '70']
        if project is not None:
            argv.extend(['--project', project])
        argv.extend(extra_args)
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

    def test_budget_and_reserve_are_configurable(self):
        self.invoke(extra_args=['--budget-usd', '5', '--reserve-usd', '1'])
        self.assertEqual(self.init_kwargs['config']['compute_allowance_usd'], 4)
        self.assertEqual(self.init_kwargs['config']['reserve_usd'], 1)

    def test_invalid_budget_values_never_spawn_gpu(self):
        for option, value in [('--budget-usd', 'nan'), ('--budget-usd', '0'),
                              ('--prior-compute-usd', 'inf'), ('--reserve-usd', '-1')]:
            with self.subTest(option=option, value=value):
                with self.assertRaisesRegex(ValueError, 'finite'):
                    self.invoke(extra_args=[option, value])
        self.gpu.with_options.assert_not_called()

    def test_project_defaults_to_shared_experiments(self):
        self.run.project = 'tubeletgraph-experiments'
        self.invoke(project=None)
        self.assertEqual(self.init_kwargs['project'], 'tubeletgraph-experiments')

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
        self.volume.iterdir.side_effect = None
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
        self.assertFalse(any(key.startswith('evaluation/') for key in self.run.summary))
        self.assertEqual(self.run.log_artifact.call_count, 2)
        self.assertEqual(len(list(Path('results').rglob('run_manifest.json'))), 1)
        self.assertNotIn('map_states', self.gpu.with_options.return_value.spawn.call_args.kwargs)
        batch = json.loads(next(Path('results').rglob('batch.json')).read_text())
        self.assertEqual(batch, {'videos': self.specs})
        identity = json.loads(next(Path('results').rglob('run.json')).read_text())
        self.assertEqual(identity['estimated_compute_usd'], self.run.summary['budget/estimated_compute_usd'])


class ResumeVideoTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.state = Path(self.tmp.name)
        self.identity = {
            'video_id': 'another-video', 'run_name': 'previous',
            'wandb_path': 'chefOST/tubeletgraph-experiments/example',
            'wandb_url': 'https://wandb.ai/chefOST/tubeletgraph-experiments/runs/example',
            'budget_usd': 5.0, 'reserve_usd': 1.0, 'estimated_compute_usd': 0.5,
        }
        self.batch = {'videos': [{'video_id': 'another-video', 'object_name': 'cup'}]}
        (self.state / 'run.json').write_text(json.dumps(self.identity))
        (self.state / 'objects.json').write_text('[]')
        (self.state / 'batch.json').write_text(json.dumps(self.batch))

    def tearDown(self):
        self.tmp.cleanup()

    def test_resume_uses_saved_batch_and_budget(self):
        with patch.object(resumer.coordinate, 'spawn', return_value=SimpleNamespace(object_id='fc-example')) as spawn:
            resumer.resume(str(self.state))
        spawn.assert_called_once_with(self.identity, [], self.batch, None, 5.0, 0.5, 1.0)

    def test_missing_saved_batch_requires_explicit_batch_file(self):
        (self.state / 'batch.json').unlink()
        with patch.object(resumer.coordinate, 'spawn') as spawn:
            with self.assertRaisesRegex(FileNotFoundError, 'pass --batch-file'):
                resumer.resume(str(self.state))
        spawn.assert_not_called()

    def test_legacy_state_requires_explicit_prior_spend(self):
        del self.identity['estimated_compute_usd']
        (self.state / 'run.json').write_text(json.dumps(self.identity))
        with patch.object(resumer.coordinate, 'spawn') as spawn:
            with self.assertRaisesRegex(ValueError, 'pass --prior-compute-usd'):
                resumer.resume(str(self.state))
        spawn.assert_not_called()

    def test_coordinator_preserves_project_and_optional_initialization(self):
        seen = []
        def local_path(path):
            return self.state if path == '/data' else Path(path)
        with patch.object(resumer, 'Path', side_effect=local_path), \
             patch('os.chdir'), patch('sys.argv', []), \
             patch.object(resumer.data_volume, 'commit'), \
             patch.object(runner, 'main', side_effect=lambda: seen.append(list(sys.argv))):
            resumer.coordinate.local(self.identity, [], self.batch, None, 5.0, 0.5, 1.0)
        argv = seen[0]
        self.assertEqual(argv[argv.index('--project') + 1], 'tubeletgraph-experiments')
        self.assertNotIn('--initialization', argv)
        self.assertEqual(json.loads(Path(argv[argv.index('--batch-file') + 1]).read_text()), self.batch)


if __name__ == '__main__':
    unittest.main()
