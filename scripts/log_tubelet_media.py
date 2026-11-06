"""Backfill per-track mask images into the existing W&B run's media panels."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
from pathlib import Path

import modal
import numpy as np
from PIL import Image, ImageDraw
from pycocotools.mask import decode
import wandb


def track_samples(frames, track_id, count=4):
    indices = sorted(int(k) for k, masks in frames.items() if track_id in masks)
    if not indices:
        return []
    return sorted({indices[round(i * (len(indices) - 1) / (count - 1))] for i in range(count)})


def render_track(frames, track_id, manifest, cache, output, label):
    indices = track_samples(frames, track_id)
    if not indices:
        return None
    sheet = Image.new('RGB', (320 * len(indices), 266), 'white')
    draw = ImageDraw.Draw(sheet)
    draw.text((6, 4), label, fill='black')
    for slot, index in enumerate(indices):
        source = manifest['clip_start_global'] + index * manifest['frame_stride']
        pixels = np.array(Image.open(cache / f'{source:07d}.jpg').convert('RGB'))
        rle = frames[str(index)][track_id]
        mask = decode(rle).astype(bool)
        if mask.shape != pixels.shape[:2]:
            raise ValueError(f'Mask/frame shape mismatch for {label}')
        pixels[mask] = (pixels[mask] * 0.5 + np.array([45, 220, 70]) * 0.5).astype('uint8')
        sheet.paste(Image.fromarray(pixels).resize((320, 240)), (320 * slot, 24))
        draw.text((320 * slot + 5, 27), f'local {index}; source {source}',
                  fill='white', stroke_width=1, stroke_fill='black')
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=86)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--snapshots', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    tasks = []
    manifests = {}
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or not (folder / 'run_manifest.json').exists():
            continue
        manifest = json.loads((folder / 'run_manifest.json').read_text())
        obj = manifest['target_object']
        manifests[obj] = manifest
        prediction = next(folder.glob('predictions/*Ours_gpt-4.1/*.json'))
        frames = json.loads(prediction.read_text())['prediction']
        for track_id in sorted({k for masks in frames.values() for k in masks}, key=int):
            tasks.append(('target', obj, track_id, frames, manifest))
    # Use the latest saved candidate graph per object. Candidates are proposal tracks,
    # not evidence that these regions belong to the initialized target object.
    latest = {}
    for snapshot in sorted(args.snapshots.rglob('tubelets.json.gz')):
        obj = snapshot.parent.parent.name
        data = json.loads(gzip.decompress(snapshot.read_bytes()))
        candidates = [(name, item['all_tracks']) for name, item in data.items()
                      if isinstance(item, dict) and 'all_tracks' in item]
        if candidates:
            latest[obj] = max(candidates, key=lambda item: ('_init/' not in item[0], item[0]))[1]
    for obj, frames in latest.items():
        for track_id in sorted({k for masks in frames.values() for k in masks}, key=int):
            tasks.append(('candidate', obj, track_id, frames, manifests[obj]))
    cache = root / 'source_frames'
    cache.mkdir(exist_ok=True)
    video_id = next(iter(manifests.values()))['video_id']
    required = {m['clip_start_global'] + i * m['frame_stride']
                for _, _, tid, frames, m in tasks for i in track_samples(frames, tid)}
    volume = modal.Volume.from_name('tubelet-data')

    def fetch(index):
        dest = cache / f'{index:07d}.jpg'
        if not dest.exists():
            dest.write_bytes(b''.join(volume.read_file(f'{video_id}/frames/{index:07d}.jpg')))
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(fetch, sorted(required)))
    print('FRAMES_READY', len(required), 'TUBELETS', len(tasks), flush=True)
    galleries = {}
    table = wandb.Table(columns=['kind', 'object', 'tubelet_id', 'first_local_frame',
                                 'last_local_frame', 'snapshot'])
    for kind, obj, tid, frames, manifest in tasks:
        label = f'{obj} / {kind} tubelet {tid}'
        path = render_track(frames, tid, manifest, cache,
                            root / 'images' / kind / obj / f'tubelet-{tid}.jpg', label)
        indices = track_samples(frames, tid)
        media = wandb.Image(str(path), caption=label)
        galleries.setdefault((kind, obj), []).append(media)
        table.add_data(kind, obj, tid, indices[0], indices[-1], media)
    run_path = 'chefOST/tubeletgraph - full evals/vie9lddr'
    original = wandb.Api().run(run_path)
    if original.state != 'finished':
        raise RuntimeError('Backfill requires the inference writer to have finished')
    with wandb.init(entity='chefOST', project='tubeletgraph - full evals',
                    id='vie9lddr', resume='must', job_type='inference') as run:
        if run.entity.lower() != 'chefost' or run.project != 'tubeletgraph - full evals':
            raise RuntimeError('Unexpected W&B destination')
        run.config.update({'tubelet_snapshot_interval_seconds': 60,
                           'tubelet_media_note': 'Post-inference backfill of saved outputs. Candidate tubelets are unvalidated proposals; green masks show one track at a time.'}, allow_val_change=True)
        del run.summary['tubelets/snapshot_interval_seconds']
        for (kind, obj), images in galleries.items():
            for offset in range(0, len(images), 32):
                key = f'tubelets/{obj}' if kind == 'target' else f'candidate_tubelets/{obj}_{offset // 32 + 1:02d}'
                run.log({key: images[offset:offset + 32]})
        for path in sorted(args.snapshots.rglob('target_tracks.jpg')):
            obj = path.parent.parent.name
            stamp = path.parent.name
            run.log({f'tubelet_progress/{obj}': wandb.Image(str(path), caption=f'{obj}; saved snapshot {stamp}; backfilled after inference')})
        run.log({'tubelet_gallery/all_tracks': table})
        run.summary['tubelet_media/target_tracks'] = sum(len(v) for (k, _), v in galleries.items() if k == 'target')
        run.summary['tubelet_media/candidate_tracks'] = sum(len(v) for (k, _), v in galleries.items() if k == 'candidate')
        print('MEDIA_LOGGED', {f'{k}/{o}': len(v) for (k, o), v in galleries.items()}, flush=True)
    print('FINISHED', run_path, flush=True)


if __name__ == '__main__':
    main()
