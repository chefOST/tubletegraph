"""Run all configured objects in one video and publish raw artifacts to ChefOST W&B."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import modal
import wandb

from modal_tubelet import BATCH_RUN_KEYS, app, data_volume, load_batch, run_tubelet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-file', type=Path, required=True)
    parser.add_argument('--project', required=True)
    parser.add_argument('--budget-usd', type=float, required=True)
    parser.add_argument('--initialization', type=Path)
    parser.add_argument('--resume-dir', type=Path)
    parser.add_argument('--prior-compute-usd', type=float, default=0.0)
    args = parser.parse_args()
    destinations = json.loads(Path('weights_and_biases/settings.json').read_text())
    if args.project not in destinations['projects'].values():
        raise ValueError('Project must match weights_and_biases/settings.json')
    specs = load_batch(str(args.batch_file))
    video_ids = {s['video_id'] for s in specs}
    if len(video_ids) != 1:
        raise ValueError('This budgeted runner accepts exactly one video')
    if not 20 <= args.budget_usd <= 70:
        raise ValueError('This runner is authorized for a budget of $20–$70')
    video_id = next(iter(video_ids))
    initialization = None
    if args.initialization:
        initialization = json.loads(args.initialization.read_text())
        if initialization['video_id'] != video_id:
            raise ValueError('Initialization metadata belongs to a different video')
        for spec in specs:
            seed = initialization['objects'][spec['object_name']]
            if seed['initial_frame'] != spec['clip_start_global']:
                raise ValueError(f"Mask/frame mismatch for {spec['object_name']}")
    previous = {}
    results = []
    if args.resume_dir:
        out = args.resume_dir
        previous = json.loads((out / 'run.json').read_text())
        if previous['video_id'] != video_id or previous['wandb_path'].rsplit('/', 1)[0] != f"{destinations['entity']}/{args.project}":
            raise ValueError('Resume state belongs to a different video or W&B destination')
        name = previous['run_name']
        results = json.loads((out / 'objects.json').read_text())
        if any(r.get('status') != 'succeeded' for r in results):
            raise ValueError('Resume state contains unsuccessful objects')
    else:
        name = datetime.now(timezone.utc).strftime('one-video-%Y%m%dT%H%M%SZ')
        out = Path('results') / video_id / name
        out.mkdir(parents=True, exist_ok=False)
    # List prices checked 2026-11-06: A100 80GB, 8 cores, 64 GiB.
    hourly_rate = (0.000694 + 8 * 0.0000131 + 64 * 0.00000222) * 3600
    if args.prior_compute_usd < 0:
        raise ValueError('Prior compute must be nonnegative')
    compute_allowance = args.budget_usd - 15 - args.prior_compute_usd
    completed = {r['object'] for r in results}
    pending = [s for s in specs if s['object_name'] not in completed]
    gpu_seconds_limit = compute_allowance / hourly_rate * 3600
    timeout_seconds = min(7200, int(gpu_seconds_limit / max(1, len(pending))) - 120)
    if timeout_seconds < 60:
        raise ValueError('Budget cannot accommodate this batch')
    # Read-only input preflight happens before creating any GPU containers.
    existing = {e.path.lstrip('/') for e in data_volume.iterdir(video_id)}
    for spec in specs:
        path = f"{video_id}/{spec['mask_name']}"
        if path not in existing:
            raise FileNotFoundError(f'Missing initial mask: {path}')
    config = {
        'video_id': video_id,
        'objects': [s['object_name'] for s in specs],
        'object_runs': len(specs),
        'sampling_fps': specs[0]['fps'] / specs[0]['frame_stride'],
        'budget_usd': args.budget_usd,
        'compute_allowance_usd': compute_allowance,
        'prior_compute_usd': args.prior_compute_usd,
        'estimated_container_usd_per_hour': hourly_rate,
        'per_object_timeout_seconds': timeout_seconds,
        'parallel_gpu_jobs': 1,
        'automatic_retries': 0,
        'sam2_speedups': True,
        'later_tracking': True,
        'evaluation_status': 'not_scored',
        'output_kind': 'raw_tubeletgraph_inference',
        'metric_note': 'No legacy evaluator scores are presented as ChefOST precision/recall.',
        'budget_note': 'Time-based compute guard; reserve $15 for API/build/storage. Not an invoice.',
        'initialization': initialization,
    }
    cumulative_seconds = 0.0
    with wandb.init(
        entity=destinations['entity'], project=args.project, name=f'{video_id}-{name}',
        job_type='inference', group=video_id,
        tags=['tubeletgraph', 'moscato', 'cmu', 'one-video', f'budget-{args.budget_usd:g}', 'raw-inference'],
        config=config, allow_val_change=True,
        **({'id': previous['wandb_path'].rsplit('/', 1)[-1], 'resume': 'must'} if previous else {}),
    ) as run:
        if run.entity.lower() != destinations['entity'].lower() or run.project != args.project:
            raise RuntimeError(f'Unexpected W&B destination: {run.entity}/{run.project}')
        if previous:
            run.config.update(config, allow_val_change=True)
        identity = {'wandb_url': run.url, 'wandb_path': run.path, 'run_name': name, **config}
        (out / 'run.json').write_text(json.dumps(identity, indent=2) + '\n')
        print('VERIFIED_WANDB', run.url, flush=True)
        print('LOCAL_RESULTS', out, flush=True)
        inputs = wandb.Artifact(f'{video_id}-{name}-inputs', type='run-inputs')
        inputs.add_file(str(args.batch_file), name='batch.json')
        if args.initialization:
            inputs.add_file(str(args.initialization), name='initial_mask_polygons.json')
            for spec in specs:
                local_mask = Path('work') / video_id / spec['mask_name']
                if local_mask.is_file():
                    inputs.add_file(str(local_mask), name=spec['mask_name'])
                frame = Path('work') / video_id / 'preflight' / f"{spec['clip_start_global']:07d}.jpg"
                if frame.is_file():
                    inputs.add_file(str(frame), name=f"initial_frames/{spec['object_name'].replace(' ', '-')}.jpg")
            preview = Path('work') / video_id / 'preflight' / 'initial-mask-review.jpg'
            if preview.is_file():
                inputs.add_file(str(preview), name='initial_mask_review.jpg')
                run.log({'inputs/initial_masks': wandb.Image(str(preview))})
        run.log_artifact(inputs)
        run.summary['inference/status'] = 'starting'
        if run.summary.get('inference/error') is not None:
            del run.summary['inference/error']
        run.summary['inference/completed_objects'] = len(results)
        run.summary['budget/prior_compute_usd'] = args.prior_compute_usd
        try:
            with (modal.enable_output() if modal.is_local() else nullcontext()), (app.run(detach=True) if modal.is_local() else nullcontext()):
                gpu = run_tubelet.with_options(
                    cpu=(8.0, 8.0), memory=(65536, 65536),
                    timeout=timeout_seconds, retries=0,
                    max_containers=1, scaledown_window=2,
                )
                for spec in pending:
                    if (cumulative_seconds + timeout_seconds + 120) > gpu_seconds_limit:
                        raise RuntimeError('Stopping before the reserved compute budget is exhausted')
                    object_run_name = f"{name}-{spec['mask_name'].split('/')[-1].removesuffix('.png')}"
                    kwargs = {k: v for k, v in spec.items() if k in BATCH_RUN_KEYS}
                    kwargs.update(run_name=object_run_name, map_states=False, sam2_speedups=True)
                    start = time.monotonic()
                    call = gpu.spawn(**kwargs)
                    run.summary['inference/status'] = 'running'
                    run.summary['inference/current_object'] = spec['object_name']
                    run.summary['inference/modal_call_id'] = call.object_id
                    print('STARTED_OBJECT', spec['object_name'], call.object_id, flush=True)
                    try:
                        while True:
                            elapsed = time.monotonic() - start
                            if elapsed > timeout_seconds + 120 or cumulative_seconds + elapsed > gpu_seconds_limit:
                                call.cancel(terminate_containers=True)
                                raise RuntimeError('Cancelled GPU job at the time-based budget limit')
                            try:
                                result = call.get(timeout=30)
                                break
                            except TimeoutError:
                                run.log({'budget/estimated_compute_usd': args.prior_compute_usd + (cumulative_seconds + elapsed) / 3600 * hourly_rate})
                    except BaseException:
                        call.cancel(terminate_containers=True)
                        raise
                    finally:
                        cumulative_seconds += time.monotonic() - start
                    if not modal.is_local():
                        data_volume.reload()
                    record = {'object': spec['object_name'], 'run_name': object_run_name, **result}
                    results.append(record)
                    (out / 'objects.json').write_text(json.dumps(results, indent=2) + '\n')
                    if not modal.is_local():
                        data_volume.commit()
                    remote = f'{video_id}/runs/{object_run_name}'
                    local = out / object_run_name
                    artifact = wandb.Artifact(f'{video_id}-{object_run_name}', type='tubeletgraph-predictions')
                    for entry in data_volume.iterdir(remote):
                        path = entry.path.lstrip('/')
                        if entry.type != modal.volume.FileEntryType.FILE:
                            continue
                        relative = Path(path).relative_to(remote)
                        dest = local / relative
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        with dest.open('wb') as handle:
                            for chunk in data_volume.read_file(path):
                                handle.write(chunk)
                        artifact.add_file(str(dest), name=str(relative))
                    run.log_artifact(artifact).wait()
                    run.log({'inference/completed_objects': len(results), 'budget/estimated_compute_usd': args.prior_compute_usd + cumulative_seconds / 3600 * hourly_rate})
                    print('FINISHED_OBJECT', spec['object_name'], result.get('status'), flush=True)
                    if result.get('status') != 'succeeded':
                        raise RuntimeError(f"Object run failed: {spec['object_name']}")
            run.summary['inference/status'] = 'completed'
        except BaseException as error:
            run.summary['inference/status'] = 'stopped'
            run.summary['inference/error'] = f'{type(error).__name__}: {error}'
            raise
        finally:
            run.summary['inference/completed_objects'] = len(results)
            run.summary['budget/estimated_compute_usd'] = args.prior_compute_usd + cumulative_seconds / 3600 * hourly_rate
            run.summary['evaluation/status'] = 'not_scored'
            if not modal.is_local():
                data_volume.commit()
            (out / 'objects.json').write_text(json.dumps(results, indent=2) + '\n')
    print('COMPLETE', json.dumps({'wandb_url': identity['wandb_url'], 'objects': len(results), 'results': str(out)}), flush=True)


if __name__ == '__main__':
    main()
