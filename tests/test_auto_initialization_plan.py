import copy
from fractions import Fraction
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.plan_auto_initialization import build_plan


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

    def test_video_frame_rate_is_metadata_and_indices_are_direct(self):
        plan = build_plan(self.root, self.manifest, self.policy)
        video = plan['videos'][0]
        self.assertEqual(video['ingredients'][0]['status'], 'ready')
        self.assertEqual(video['frame_index_basis'], 'source_frame_indices')
        self.assertEqual(plan['policy']['frame_index_basis'], 'source_frame_indices')
        self.assertEqual(video['source_fps'], 30)
        self.assertNotIn('alignment', video)

    def test_annotation_length_and_duration_do_not_remap_indices(self):
        self.manifest['entries'][0] = self.entry('video-1', count=24, fps='24/1')
        self.manifest['entries'][0]['media']['streams'][0]['duration'] = '999'
        plan = self.plan()
        video = plan['videos'][0]
        self.assertEqual(video['source_fps'], 24)
        self.assertEqual(video['ingredients'][0]['seed_source_frame'], 2)
        self.assertEqual(video['ingredients'][0]['status'], 'ready')

    def test_duration_is_optional_and_never_required_for_seeding(self):
        self.manifest['entries'][0]['media']['streams'][0].pop('duration')
        video = self.plan()['videos'][0]
        self.assertIsNone(video['duration_seconds'])
        self.assertEqual(video['ingredients'][0]['seed_source_frame'], 2)
        self.assertEqual(video['ingredients'][0]['status'], 'ready')

    def test_seed_must_still_be_within_source_video(self):
        self.manifest['entries'][0] = self.entry('video-1', count=2)
        egg = self.plan()['videos'][0]['ingredients'][0]
        self.assertEqual(egg['skip_reason'], 'seed_after_video_end')
        self.assertEqual(egg['seed_source_frame'], 2)

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

    def test_identity_is_stable_when_media_metadata_is_missing(self):
        raw = self.plan()['videos'][0]['ingredients'][0]
        self.manifest['entries'][0]['media']['streams'][0].pop('avg_frame_rate')
        blocked = self.plan()['videos'][0]['ingredients'][0]
        self.assertEqual(blocked['skip_reason'], 'missing_media_metadata')
        self.assertEqual(blocked['seed_source_frame'], 2)
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
