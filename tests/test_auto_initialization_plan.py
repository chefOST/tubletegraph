import copy
from fractions import Fraction
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.plan_auto_initialization import alignment_for, build_plan


class AutoInitializationPlanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.annotations = self.root / 'MOSCATO/annotations/ground_truth/CMU/gt_annotations_cmu.json'
        self.annotations.parent.mkdir(parents=True)
        vocab = self.root / 'MOSCATO/vocabulary/CMU/object_dict.json'
        vocab.parent.mkdir(parents=True)
        vocab.write_text(json.dumps({'o2i': {'egg': 0, 'oil': 1, 'bowl': 2},
                                     'all2one': {'eggs': 'egg'}}))
        self.policy = {'scope': 'ingredients', 'definition': 'Food, excluding vessels',
                       'datasets': {'CMU': {'include': ['egg', 'oil'], 'exclude': ['bowl']}}}
        self.manifest = {'entries': [self.entry('video-1')]}
        self.write({'video-1': {'object': [[], ['eggs'], ['egg'], []],
                               'state': {'egg': [[], [], [], ['late-secret-state']],
                                         'oil': [['never-visible-state']]}}})
        self.revision = patch('scripts.plan_auto_initialization._data_revision', return_value='data-commit')
        self.revision.start()
        self.addCleanup(self.revision.stop)

    @staticmethod
    def entry(video, count=4, fps='30/1'):
        return {'dataset': 'CMU', 'ground_truth_video_id': video,
                'download_status': 'downloaded_verified_identity', 'local_path': f'CMU/{video}.mp4',
                'sha256': 'video-hash', 'media': {'streams': [
                    {'width': 800, 'height': 600, 'avg_frame_rate': fps, 'nb_frames': str(count),
                     'duration': str(float(count / Fraction(fps)))}]}}

    def write(self, value):
        self.annotations.write_text(json.dumps(value))

    def plan(self):
        return build_plan(self.root, self.manifest, self.policy)

    def test_presence_drives_seed_and_state_values_never_leave_planner(self):
        plan = self.plan()
        egg, oil = plan['videos'][0]['ingredients']
        self.assertEqual((egg['first_visible_annotation_index'], egg['seed_annotation_index'],
                          egg['seed_source_frame'], egg['status']), (1, 2, 2, 'ready'))
        self.assertEqual(oil['skip_reason'], 'no_visibility_annotation')
        self.assertNotIn('secret-state', json.dumps(plan))
        annotation = json.loads(self.annotations.read_text())
        annotation['video-1']['state']['egg'] = {'arbitrary': 'changed-state-values'}
        self.write(annotation)
        self.assertEqual(self.plan()['videos'], plan['videos'])

    def test_matching_counts_preserve_raw_indices_and_record_inference(self):
        plan = build_plan(self.root, self.manifest, self.policy)
        alignment = plan['videos'][0]['alignment']
        self.assertEqual(plan['videos'][0]['ingredients'][0]['status'], 'ready')
        self.assertEqual(alignment['status'], 'inferred_from_duration')
        self.assertEqual(alignment['index_basis'], 'source_frame_indices')
        self.assertEqual(alignment['annotation_fps'], 30)
        self.assertNotIn('declared_annotation_fps', alignment)
        self.assertFalse(alignment['independently_verified'])

    def test_other_uniform_rates_use_duration_and_timestamp_not_raw_index(self):
        self.manifest['entries'][0] = self.entry('video-1', count=24, fps='24/1')
        plan = self.plan()
        self.assertEqual(plan['videos'][0]['alignment']['index_basis'], 'uniform_annotation_timeline')
        self.assertEqual(plan['videos'][0]['alignment']['annotation_fps'], 4)
        self.assertIsNone(plan['videos'][0]['ingredients'][0]['seed_source_frame'])
        self.assertEqual(plan['videos'][0]['ingredients'][0]['seed_timestamp_seconds'], 0.5)

    def test_resampled_mapping_uses_timestamp_for_actual_pts_lookup(self):
        annotation = {'object': [[], ['egg'], ['egg']] + [[]] * 7, 'state': {'egg': []}}
        self.write({'video-1': annotation})
        self.manifest['entries'][0] = self.entry('video-1', count=24, fps='24/1')
        video = self.plan()['videos'][0]
        egg = video['ingredients'][0]
        self.assertEqual(egg['seed_annotation_index'], 2)
        self.assertIsNone(egg['seed_source_frame'])
        self.assertEqual(egg['seed_timestamp_seconds'], 0.2)
        self.assertEqual(egg['status'], 'ready')
        self.assertEqual(video['alignment']['index_basis'], 'uniform_annotation_timeline')
        # Source-rate arrays must not be interpreted as 10 fps merely because
        # the dataset's general documentation mentions that rate.
        self.assertEqual(alignment_for(self.entry('video-1', 240, '24/1'), 240)
                         ['index_basis'], 'source_frame_indices')

    def test_exact_next_index_cannot_jump_to_later_visibility(self):
        self.write({'video-1': {'object': [['egg'], [], ['egg'], ['egg']], 'state': {'egg': []}}})
        egg = self.plan()['videos'][0]['ingredients'][0]
        self.assertEqual(egg['seed_annotation_index'], 1)
        self.assertEqual(egg['skip_reason'], 'not_visible_at_seed')
        self.assertEqual(egg['status'], 'skipped')

    def test_missing_videos_and_final_frame_seeds_remain_explicit(self):
        self.write({'video-1': {'object': [[], [], [], ['egg']], 'state': {'egg': []}},
                    'unavailable': {'object': [['oil'], ['oil']], 'state': {'oil': []}}})
        plan = self.plan()
        self.assertEqual(plan['summary']['videos'], 2)
        items = {v['video_id']: v['ingredients'][0] for v in plan['videos']}
        self.assertEqual(items['unavailable']['skip_reason'], 'missing_r2_video')
        self.assertEqual(items['video-1']['skip_reason'], 'seed_after_annotation_end')

    def test_identity_is_stable_when_media_timing_is_missing(self):
        raw = self.plan()['videos'][0]['ingredients'][0]
        self.manifest['entries'][0]['media']['streams'][0].pop('duration')
        blocked = self.plan()['videos'][0]['ingredients'][0]
        self.assertEqual(blocked['skip_reason'], 'missing_media_timing')
        self.assertEqual(raw['id'], blocked['id'])
        self.assertEqual(raw['mask_name'], blocked['mask_name'])

    def test_policy_must_classify_objects_and_manifest_ids_must_be_unique(self):
        self.policy['datasets']['CMU']['exclude'] = []
        with self.assertRaisesRegex(ValueError, 'classify every'):
            self.plan()
        self.policy['datasets']['CMU']['exclude'] = ['bowl']
        self.manifest['entries'].append(copy.deepcopy(self.manifest['entries'][0]))
        with self.assertRaisesRegex(ValueError, 'Duplicate manifest'):
            self.plan()


if __name__ == '__main__':
    unittest.main()
