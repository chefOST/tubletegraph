"""Resume a budgeted run in a detached CPU coordinator on Modal."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import modal

from modal_tubelet import app, data_volume, evaluation_image, wandb_secret

coordinator_image = (
    evaluation_image
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
def coordinate(identity: dict, records: list, batch: dict, initialization: dict,
               budget_usd: float, prior_compute_usd: float) -> dict:
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
    (out / 'initialization.json').write_text(json.dumps(initialization, indent=2) + '\n')
    data_volume.commit()
    sys.argv = ['run_budgeted_video', '--batch-file', str(out / 'batch.json'),
                '--initialization', str(out / 'initialization.json'),
                '--project', 'tubeletgraph - full evals', '--budget-usd', str(budget_usd),
                '--prior-compute-usd', str(prior_compute_usd), '--resume-dir', str(out)]
    try:
        run_budgeted_video.main()
    finally:
        data_volume.commit()
    return {'wandb_url': identity['wandb_url'], 'results': str(out)}


@app.local_entrypoint()
def resume(resume_dir: str, budget_usd: float = 39.0, prior_compute_usd: float = 3.0):
    state = Path(resume_dir)
    identity = json.loads((state / 'run.json').read_text())
    records = json.loads((state / 'objects.json').read_text())
    batch = json.loads(Path('configs/batch_s15_salad_visible_start.json').read_text())
    initialization = identity['initialization']
    call = coordinate.spawn(identity, records, batch, initialization, budget_usd, prior_compute_usd)
    print(json.dumps({'coordinator_call_id': call.object_id, 'wandb_url': identity['wandb_url'],
                      'budget_usd': budget_usd, 'prior_compute_usd': prior_compute_usd}), flush=True)
