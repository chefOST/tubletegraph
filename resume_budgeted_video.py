"""Resume a budgeted run in a detached CPU coordinator on Modal."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from typing import Optional

import modal

from modal_tubelet import app, data_volume, orchestration_image, wandb_secret

coordinator_image = (
    orchestration_image
    .add_local_file('modal_tubelet.py', '/root/modal_tubelet.py', copy=True)
    .add_local_file('run_budgeted_video.py', '/root/run_budgeted_video.py', copy=True)
    .add_local_dir('configs', '/opt/resume/configs', copy=True)
    .add_local_dir('weights_and_biases', '/opt/resume/weights_and_biases', copy=True)
)


@app.function(
    image=coordinator_image,
    cpu=(0.25, 0.25), memory=(512, 512),
    timeout=7 * 3600, retries=0, max_containers=1, scaledown_window=2,
    volumes={'/data': data_volume}, secrets=[wandb_secret],
)
def coordinate(identity: dict, records: list, batch: dict, initialization: Optional[dict],
               budget_usd: float, prior_compute_usd: float, reserve_usd: float) -> dict:
    import os
    from datetime import datetime, timezone
    import run_budgeted_video

    os.chdir('/opt/resume')
    out = Path('/data') / identity['video_id'] / 'orchestration' / (identity['run_name'] + '-resume-' + datetime.now(timezone.utc).strftime('%H%M%S'))
    if (out / 'run.json').exists():
        raise RuntimeError('Coordinator already has saved state; inspect it before another launch')
    # Verify saved successful outputs before skipping any GPU work.
    specs = {s['object_name']: s for s in batch['videos']}
    for record in records:
        remote = Path('/data') / identity['video_id'] / 'runs' / record['run_name']
        manifest = json.loads((remote / 'run_manifest.json').read_text())
        spec = specs[record['object']]
        expected = {'status': 'succeeded', 'video_id': identity['video_id'],
                    'target_object': record['object'], 'frame_stride': spec['frame_stride'],
                    'clip_start_global': spec['clip_start_global']}
        if any(manifest.get(k) != v for k, v in expected.items()):
            raise RuntimeError(f'Completed output does not match batch: {record["object"]}')
        if not list((remote / 'predictions').rglob('*.json')):
            raise RuntimeError('Completed output has no prediction JSON')
    out.mkdir(parents=True)
    (out / 'run.json').write_text(json.dumps(identity, indent=2) + '\n')
    (out / 'objects.json').write_text(json.dumps(records, indent=2) + '\n')
    (out / 'batch.json').write_text(json.dumps(batch, indent=2) + '\n')
    if initialization is not None:
        (out / 'initialization.json').write_text(json.dumps(initialization, indent=2) + '\n')
    data_volume.commit()
    sys.argv = ['run_budgeted_video', '--batch-file', str(out / 'batch.json'),
                '--project', identity['wandb_path'].rsplit('/', 2)[1],
                '--budget-usd', str(budget_usd), '--reserve-usd', str(reserve_usd),
                '--prior-compute-usd', str(prior_compute_usd), '--resume-dir', str(out)]
    if initialization is not None:
        sys.argv.extend(['--initialization', str(out / 'initialization.json')])
    try:
        run_budgeted_video.main()
    finally:
        data_volume.commit()
    return {'wandb_url': identity['wandb_url'], 'results': str(out)}


@app.local_entrypoint()
def resume(resume_dir: str, budget_usd: Optional[float] = None,
           prior_compute_usd: Optional[float] = None, reserve_usd: Optional[float] = None,
           batch_file: str = ''):
    state = Path(resume_dir)
    identity = json.loads((state / 'run.json').read_text())
    records = json.loads((state / 'objects.json').read_text())
    batch_path = Path(batch_file) if batch_file else state / 'batch.json'
    if not batch_path.is_file():
        raise FileNotFoundError('Saved batch.json is missing; pass --batch-file for this run')
    batch = json.loads(batch_path.read_text())
    initialization = identity.get('initialization')
    if budget_usd is None:
        budget_usd = identity['budget_usd']
    if prior_compute_usd is None:
        if 'estimated_compute_usd' not in identity:
            raise ValueError('Saved state has no spend estimate; pass --prior-compute-usd')
        prior_compute_usd = identity['estimated_compute_usd']
    if reserve_usd is None:
        reserve_usd = identity.get('reserve_usd', 15.0)
    if not math.isfinite(budget_usd) or budget_usd <= 0:
        raise ValueError('Budget must be finite and positive')
    if any(not math.isfinite(value) or value < 0 for value in (prior_compute_usd, reserve_usd)):
        raise ValueError('Prior compute and reserve must be finite and nonnegative')
    call = coordinate.spawn(identity, records, batch, initialization,
                            budget_usd, prior_compute_usd, reserve_usd)
    print(json.dumps({'coordinator_call_id': call.object_id, 'wandb_url': identity['wandb_url'],
                      'budget_usd': budget_usd, 'prior_compute_usd': prior_compute_usd,
                      'reserve_usd': reserve_usd}), flush=True)
