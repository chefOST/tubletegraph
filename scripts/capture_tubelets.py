"""Read a bounded snapshot of intermediate tubelets inside an inference container."""
from pathlib import Path
import base64
import gzip
import hashlib
import io
import json
import sys


def capture(video_id, previous_digest='', completed_run=''):
    import numpy as np
    from PIL import Image, ImageDraw
    from pycocotools.mask import decode

    root = Path('/opt/TubeletGraph')
    manifest = {}
    directories = ('_interm_out', '_pred_out')
    if completed_run:
        root = Path('/data') / video_id / 'runs' / completed_run
        manifest = json.loads((root / 'run_manifest.json').read_bytes())
        directories = ('predictions',)
    sources = {}
    for directory in directories:
        for path in sorted((root / directory).rglob('*.json')):
            if video_id not in path.name:
                continue
            if directory == '_interm_out' and not any(
                part.startswith(('tubelets_', 'pred_custom_')) for part in path.parts
            ):
                continue
            if path.stat().st_size > 25_000_000:
                continue
            try:
                sources[str(path.relative_to(root))] = json.loads(path.read_bytes())
            except (OSError, ValueError):
                continue  # A producer may currently be replacing this file.
    raw = json.dumps(sources, sort_keys=True, separators=(',', ':')).encode()
    digest = hashlib.sha256(raw).hexdigest()
    if not sources or digest == previous_digest:
        return {'digest': digest, 'changed': False}
    previews = {}
    for kind, key in (('candidate_tubelets', 'all_tracks'), ('target_tracks', 'prediction')):
        matches = [(name, data[key]) for name, data in sources.items()
                   if isinstance(data, dict) and isinstance(data.get(key), dict)]
        if not matches:
            continue
        name, frames = max(matches, key=lambda item: ('_init/' not in item[0], item[0]))
        indices = sorted((int(k) for k in frames if k.isdigit()))
        if not indices:
            continue
        selected = [indices[round(i * (len(indices) - 1) / 7)] for i in range(8)]
        sheet = Image.new('RGB', (1280, 520), 'white')
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 4), f'{kind}: {name}', fill='black')
        colors = [(255, 60, 60), (60, 220, 80), (70, 120, 255), (255, 200, 40), (220, 60, 255)]
        for slot, index in enumerate(selected):
            if completed_run:
                source_index = manifest['clip_start_global'] + index * manifest['frame_stride']
                image_path = Path('/data') / video_id / 'frames' / f'{source_index:07d}.jpg'
            else:
                image_path = Path('/tmp/tubelet_input') / video_id / f'{index:07d}.jpg'
            if not image_path.exists():
                continue
            pixels = np.array(Image.open(image_path).convert('RGB'))
            masks = frames[str(index)]
            if not isinstance(masks, dict):
                continue
            for order, (track_id, rle) in enumerate(sorted(masks.items())):
                if not isinstance(rle, dict) or 'counts' not in rle:
                    continue
                try:
                    mask = decode(rle).astype(bool)
                    if mask.shape != pixels.shape[:2]:
                        continue
                    color = np.array(colors[order % len(colors)])
                    pixels[mask] = (pixels[mask] * 0.55 + color * 0.45).astype('uint8')
                except (ValueError, TypeError):
                    continue
            tile = Image.fromarray(pixels).resize((320, 240))
            x, y = (slot % 4) * 320, 24 + (slot // 4) * 248
            sheet.paste(tile, (x, y))
            draw.text((x + 4, y + 4), f'sampled frame {index}', fill='white', stroke_width=1, stroke_fill='black')
        buffer = io.BytesIO()
        sheet.save(buffer, format='JPEG', quality=82)
        previews[f'{kind}.jpg'] = base64.b64encode(buffer.getvalue()).decode()
    return {'changed': True, 'digest': digest, 'sources': list(sources),
            'snapshot': base64.b64encode(gzip.compress(raw, mtime=0)).decode(),
            'previews': previews}


if __name__ == '__main__':
    import time
    encoded = json.dumps(capture(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else '', sys.argv[3] if len(sys.argv) > 3 else ''))
    for offset in range(0, len(encoded), 8192):
        print(encoded[offset:offset + 8192], flush=True)
        time.sleep(0.02)
    time.sleep(1)
