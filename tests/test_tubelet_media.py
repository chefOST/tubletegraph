import tempfile
from pathlib import Path
import unittest

import numpy as np
from PIL import Image
from pycocotools.mask import encode

from scripts.log_tubelet_media import render_track, track_samples


class TubeletMediaTests(unittest.TestCase):
    def test_samples_follow_one_tracks_lifetime(self):
        frames = {'0': {'a': {}}, '1': {'b': {}}, '3': {'a': {}}, '8': {'a': {}}}
        self.assertEqual(track_samples(frames, 'a'), [0, 3, 8])
        self.assertEqual(track_samples(frames, 'missing'), [])

    def test_renderer_uses_source_offset_stride_and_one_track(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            Image.new('RGB', (320, 240), (0, 0, 0)).save(root / '0000106.jpg')
            mask = np.zeros((240, 320), dtype=np.uint8)
            mask[100:140, 100:140] = 1
            other = np.zeros_like(mask)
            other[100:140, 200:240] = 1
            frames = {'2': {'a': encode(np.asfortranarray(mask)),
                            'b': encode(np.asfortranarray(other))}}
            result = render_track(frames, 'a', {'clip_start_global': 100, 'frame_stride': 3},
                                  root, root / 'preview.jpg', 'track a')
            pixels = np.array(Image.open(result))
            self.assertGreater(pixels[144, 120, 1], 80)
            self.assertLess(pixels[144, 220, 1], 20)


if __name__ == '__main__':
    unittest.main()
