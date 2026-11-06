"""Publish intermediate tubelet snapshots to an existing W&B run without another writer."""
from __future__ import annotations

import argparse
import base64
import fcntl
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import wandb


def upload_snapshot(run, payload, output, object_name, container_id):
    if not payload.get('changed'):
        return []
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    folder = output / 'tubelet_snapshots' / object_name / stamp
    folder.mkdir(parents=True, exist_ok=True)
    files = {'tubelets.json.gz': payload['snapshot'], **payload.get('previews', {})}
    metadata = {'captured_at': stamp, 'object': object_name, 'container_id': container_id,
                'sha256': payload['digest'], 'sources': payload['sources'],
                'note': 'Intermediate inference outputs; candidate tubelets are not validated target tracks. Frame numbers are sampled local indices.'}
    (folder / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    for name, encoded in files.items():
        (folder / name).write_bytes(base64.b64decode(encoded, validate=True))
    uploaded = []
    for path in sorted(folder.iterdir()):
        run.upload_file(str(path), root=str(output))
        uploaded.append(str(path.relative_to(output)))
    return uploaded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wandb-path', required=True)
    parser.add_argument('--modal-app', required=True)
    parser.add_argument('--interval', type=int, default=300)
    parser.add_argument('--max-hours', type=float, default=7)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.wandb_path.rsplit('/', 1)[0] != 'chefOST/tubeletgraph - full evals':
        raise ValueError('Unexpected W&B destination')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / 'watcher.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    capture_code = Path(__file__).with_name('capture_tubelets.py').read_text()
    modal_cmd = [sys.executable, '-m', 'modal']
    digests = {}
    deadline = time.monotonic() + args.max_hours * 3600
    while time.monotonic() < deadline:
        try:
            run = wandb.Api(timeout=45).run(args.wandb_path)
            object_name = run.summary.get('inference/current_object')
            if run.summary.get('inference/status') in ('completed', 'stopped'):
                print('STOP', run.summary.get('inference/status'), flush=True)
                break
            result = subprocess.run([*modal_cmd, 'container', 'list', '--json'],
                                    capture_output=True, text=True, check=True, timeout=45)
            containers = [c for c in json.loads(result.stdout) if c['app_id'] == args.modal_app]
            for container in containers:
                container_id = container['container_id']
                # The CPU coordinator has no TubeletGraph installation and returns no snapshot.
                code = 'from pathlib import Path\nif Path("/opt/TubeletGraph").is_dir():\n' + '\n'.join(
                    '    ' + line for line in capture_code.splitlines())
                result = subprocess.run([*modal_cmd, 'container', 'exec', '--no-pty', '--',
                                         container_id, 'python', '-c', code,
                                         run.config['video_id'], digests.get(container_id, '')],
                                        capture_output=True, text=True, timeout=90, check=True)
                if not result.stdout.strip():
                    continue
                payload = json.loads(''.join(result.stdout.splitlines()))
                uploaded = upload_snapshot(run, payload, output, object_name, container_id)
                if uploaded:
                    digests[container_id] = payload['digest']
                    print(json.dumps({'uploaded': uploaded, 'container_id': container_id}), flush=True)
        except Exception as error:
            print(f'RETRY {type(error).__name__}: {error}', flush=True)
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
