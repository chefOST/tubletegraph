import base64
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock

from scripts.watch_tubelets import upload_snapshot


class TubeletUploadTests(unittest.TestCase):
    def test_unchanged_snapshot_uploads_nothing(self):
        run = MagicMock()
        self.assertEqual(upload_snapshot(run, {'changed': False}, Path('/unused'), 'knife', 'ta-example'), [])
        run.upload_file.assert_not_called()

    def test_snapshot_and_preview_are_attached_to_existing_run(self):
        raw = b'{"all_tracks": {"0": {}}}'
        payload = {'changed': True, 'digest': 'abc', 'sources': ['tubelets/video.json'],
                   'snapshot': base64.b64encode(gzip.compress(raw)).decode(),
                   'previews': {'target_tracks.jpg': base64.b64encode(b'preview').decode()}}
        run = MagicMock()
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            paths = upload_snapshot(run, payload, output, 'knife', 'ta-example')
            self.assertEqual(len(paths), 3)
            self.assertEqual(run.upload_file.call_count, 3)
            snapshot = next(output.rglob('tubelets.json.gz'))
            self.assertEqual(gzip.decompress(snapshot.read_bytes()), raw)
            metadata = json.loads(next(output.rglob('metadata.json')).read_text())
            self.assertEqual(metadata['object'], 'knife')
            self.assertEqual(metadata['container_id'], 'ta-example')
            for call in run.upload_file.call_args_list:
                self.assertEqual(call.kwargs['root'], str(output))
                self.assertTrue(Path(call.args[0]).is_relative_to(output / 'tubelet_snapshots'))


if __name__ == '__main__':
    unittest.main()
