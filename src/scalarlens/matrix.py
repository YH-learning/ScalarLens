"""Run a resumable paper matrix, at most one child process per listed GPU slot."""
import argparse
import concurrent.futures
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from .backbones import BACKBONES
from .protocol import METHODS


def valid_result(path, task):
    if not path.exists():
        return False
    row = json.loads(path.read_text())
    identity = (row['dataset'], row['backbone'], row['method'], row['seed'])
    if identity != (task['dataset'], task['backbone'], task['method'], task['seed']):
        raise ValueError(f'Result identity mismatch: {path}')
    if not all(math.isfinite(float(row[m])) for m in ('val_auc', 'val_logloss', 'test_auc', 'test_logloss')):
        raise ValueError(f'Nonfinite result: {path}')
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True, help='Contains the dataset ID directories')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--datasets', nargs='+', default=['AutoML_A', 'AutoML_E', 'Criteo_x4_NumEmbRaw'])
    p.add_argument('--backbones', nargs='+', choices=BACKBONES, default=list(BACKBONES))
    p.add_argument('--methods', nargs='+', choices=METHODS, default=['scalarlens'])
    p.add_argument('--seeds', nargs='+', type=int, default=[2026, 2027, 2028])
    p.add_argument('--gpus', nargs='+', type=int, default=[-1], help='-1 means CPU')
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if len(a.gpus) != len(set(a.gpus)):
        p.error('Duplicate GPU slots would oversubscribe a device.')
    a.output_dir.mkdir(parents=True, exist_ok=True)
    tasks = [dict(dataset=d, backbone=b, method=m, seed=s) for d in a.datasets
             for b in a.backbones for m in a.methods for s in a.seeds]
    manifest = dict(tasks=tasks, data_root=str(a.data_root.resolve()), workers=a.workers,
                    protocol='paper-raw-auc-p2', output_schema=1)
    target = a.output_dir/'task_manifest.json'
    if target.exists() and json.loads(target.read_text()) != manifest:
        raise ValueError('Output directory belongs to a different matrix. Choose a new output directory.')
    if not a.dry_run:
        target.write_text(json.dumps(manifest, indent=2)+'\n')
    for task in tasks:
        task['output'] = str(a.output_dir / f"{task['dataset']}__{task['backbone']}__{task['method']}__s{task['seed']}")
    pending = [t for t in tasks if not valid_result(Path(t['output'])/'result.json', t)]
    print(f'Tasks: {len(tasks)}; completed: {len(tasks)-len(pending)}; pending: {len(pending)}', flush=True)
    if a.dry_run:
        print(json.dumps(pending, indent=2))
        return
    # A worker owns one GPU slot and consumes a static round-robin queue. No
    # automatic retries, overwrites of completed tasks, or process termination.
    def worker(gpu, queue):
        failures = []
        for task in queue:
            output = Path(task['output'])
            output.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, '-m', 'scalarlens.train', '--dataset-dir',
                       str(a.data_root/task['dataset']), '--output-dir', str(output),
                       '--backbone', task['backbone'], '--method', task['method'],
                       '--seed', str(task['seed']), '--gpu', str(gpu), '--workers', str(a.workers)]
            with (output/'console.log').open('a') as log:
                returncode = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
            passed = returncode == 0 and valid_result(output/'result.json', task)
            print(('DONE' if passed else 'FAILED') + ' ' + str(output), flush=True)
            if not passed:
                failures.append(dict(task, returncode=returncode))
        return failures
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(a.gpus)) as pool:
        futures = [pool.submit(worker, gpu, pending[i::len(a.gpus)]) for i, gpu in enumerate(a.gpus)]
        for future in futures:
            failures += future.result()
    (a.output_dir/'failures.json').write_text(json.dumps(failures, indent=2)+'\n')
    rows = [json.loads((Path(t['output'])/'result.json').read_text()) for t in tasks
            if valid_result(Path(t['output'])/'result.json', t)]
    if rows:
        with (a.output_dir/'per_seed.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if failures:
        raise SystemExit(f'{len(failures)} failures; inspect failures.json before resuming.')


if __name__ == '__main__':
    main()
