import unittest

from scripts.auto_video import resolve_seed, sampled_frames


class AutoVideoTests(unittest.TestCase):
    def setUp(self):
        self.video = {"source_frame_count": 300, "source_fps": 30.0, "duration_seconds": 10,
                      "annotation_frame_count": 300, "frame_index_basis": "source_frame_indices"}
        self.media = {"source_frame_count": 300, "source_fps": 30.0, "duration_seconds": 10,
                      "timestamps_seconds": [i / 30 for i in range(300)]}
        self.ingredient = {"status": "ready", "first_visible_annotation_index": 100,
                           "seed_annotation_index": 101, "seed_source_frame": 101}

    def test_source_frame_indices_do_not_assume_ten_fps(self):
        self.assertEqual(resolve_seed(self.video, self.ingredient, self.media), 101)

    def test_timing_metadata_does_not_reinterpret_annotation_indices(self):
        media = {**self.media, "source_fps": 24, "duration_seconds": 50,
                 "source_frame_count": 200}
        self.assertEqual(resolve_seed(self.video, self.ingredient, media), 101)

    def test_invalid_seed_fails(self):
        with self.assertRaisesRegex(ValueError, "immediately after"):
            resolve_seed(self.video, {**self.ingredient, "seed_annotation_index": 102}, self.media)
        with self.assertRaisesRegex(ValueError, "outside the video"):
            resolve_seed(self.video, self.ingredient, {**self.media, "source_frame_count": 100})

    def test_twenty_four_fps_sampling_preserves_seed_and_video_extent(self):
        times = [i / 24 for i in range(240)]
        rows = sampled_frames(times, 13, 5)
        self.assertEqual(rows[0], {"local_frame": 0, "source_frame": 13, "timestamp_seconds": 13 / 24})
        self.assertLess(times[-1] - rows[-1]["timestamp_seconds"], .2 + 1 / 24)
        self.assertEqual(len({row["source_frame"] for row in rows}), len(rows))
        self.assertTrue(all(row["timestamp_seconds"] == times[row["source_frame"]] for row in rows))

    def test_irregular_pts_are_used_instead_of_nominal_fps(self):
        rows = sampled_frames([0, .05, .11, .24, .39, .65], 1, 5)
        self.assertEqual([row["source_frame"] for row in rows], [1, 4, 5])
        with self.assertRaises(ValueError):
            sampled_frames([0, .1, .1], 0)


if __name__ == "__main__":
    unittest.main()
