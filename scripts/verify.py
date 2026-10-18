"""Run local regression tests, paper reaggregation, and real CPU smoke training.

This is not a rerun of the real-data benchmark. Each invocation uses a fresh
temporary directory; no archived result or user dataset is modified.
"""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT/'outputs/validation'
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONPATH=str(ROOT/'src'))
    began = time.monotonic()
    report = dict(status='RUNNING', python=sys.version, platform=platform.platform(),
                  scope='CPU regression + synthetic training + archived paper reaggregation; not real-data retraining.',
                  packages={name: importlib.metadata.version(name) for name in
                            ('torch', 'numpy', 'pandas', 'pyarrow', 'polars', 'scikit-learn', 'PyYAML', 'rtdl-num-embeddings')},
                  checks=[], smoke_results=[])

    def run(name, args, cwd):
        path = output/(name+'.log')
        with path.open('w') as stream:
            code = subprocess.run([sys.executable, *args], cwd=cwd, env=env,
                                  stdout=stream, stderr=subprocess.STDOUT).returncode
        report['checks'].append(dict(name=name, returncode=code, log=path.name))
        if code:
            raise RuntimeError(f'{name} failed; inspect {path}')

    try:
        run('pip_check', ['-m', 'pip', 'check'], ROOT)
        run('publication_audit', [str(ROOT/'scripts/audit.py')], ROOT)
        run('pytest', ['-m', 'pytest', str(ROOT/'tests'), '-q'], ROOT)
        run('real_schema_equivalence', [str(ROOT/'scripts/check_schemas.py')], ROOT)
        # Run outside the repository, demonstrating no cwd/parent-project import dependency.
        with tempfile.TemporaryDirectory(prefix='scalarlens-check-') as work:
            work = Path(work)
            run('paper_reaggregation', ['-m', 'scalarlens.results', '--source', str(ROOT/'reproduction/paper'),
                                       '--output', str(ROOT/'outputs/paper')], work)
            data = work/'ScalarLensSmoke'
            run('make_smoke', ['-m', 'scalarlens.data', '--dataset-dir', str(data), '--make-smoke'], work)
            tasks = [('DCNv2', 'scalarlens', s) for s in (2026, 2027, 2028)]
            tasks += [('DeepFM', 'scalarlens', 2026), ('DNN', 'scalarlens', 2026)]
            tasks += [('DCNv2', m, 2026) for m in ('field', 'deer', 'daes_gate', 'narydis')]
            tasks += [('DCNv2', 'scalarlens', 2026)]  # repeated seed, fresh directory
            for i, (backbone, method, seed) in enumerate(tasks):
                name = f'smoke_{i:02d}_{backbone}_{method}_{seed}'
                destination = work/name
                run(name, ['-m', 'scalarlens.train', '--dataset-dir', str(data), '--output-dir', str(destination),
                           '--backbone', backbone, '--method', method, '--seed', str(seed), '--gpu', '-1',
                           '--workers', '0', '--epochs', '2', '--batch-size', '128'], work)
                row = json.loads((destination/'result.json').read_text())
                # Keep metrics, configs and raw console logs, never weights.
                row['checkpoint'] = '<temporary CPU smoke checkpoint; not distributed>'
                report['smoke_results'].append(row)
                shutil.copy2(destination/'config.json', output/(name+'_config.json'))
            first, repeated = report['smoke_results'][0], report['smoke_results'][-1]
            keys = ('val_auc', 'val_logloss', 'test_auc', 'test_logloss', 'best_epoch', 'stopped_epoch')
            report['cpu_seed_repeat_exact'] = all(first[k] == repeated[k] for k in keys)
            if not report['cpu_seed_repeat_exact']:
                raise AssertionError('Same-seed CPU training did not reproduce exactly')
        report['status'] = 'PASS'
    except Exception as exc:
        report['status'] = 'FAIL'
        report['error'] = repr(exc)
    finally:
        report['elapsed_seconds'] = time.monotonic()-began
        report['paper_audit'] = json.loads((ROOT/'outputs/paper/verification.json').read_text())
        (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        lines = ['# Source verification', '',
                 'Scope: encoder refactoring, integration, synthetic CPU training, and archived statistics. Not full real-data retraining or GPU speed verification.',
                 '', f"Status: {report['status']}. Python {platform.python_version()} / {platform.machine()} / torch {report['packages']['torch']}.", '',
                 '| Check | Exit code | Log |', '|---|---:|---|']
        lines += [f"| {r['name']} | {r['returncode']} | [{r['log']}]({r['log']}) |" for r in report['checks']]
        lines += ['', f"Exact repeated-seed CPU training: {report.get('cpu_seed_repeat_exact', False)}.", '',
                  'Paper: 1,539 seed records, 513 groups, 4,104 aggregate values and 1,539 table values checked; 54 stratified sample cells.', '',
                  '## Synthetic smoke results (not benchmark scores)', '',
                  '| Backbone | Method | Seed | Valid AUC | Test AUC | Test LL |', '|---|---|---:|---:|---:|---:|']
        lines += [f"| {r['backbone']} | {r['method']} | {r['seed']} | {r['val_auc']:.6f} | {r['test_auc']:.6f} | {r['test_logloss']:.6f} |" for r in report['smoke_results']]
        if report.get('error'):
            lines += ['', report['error']]
        (output/'report.md').write_text('\n'.join(lines)+'\n')
        curated = ROOT/'reproduction/checks'
        curated.mkdir(parents=True, exist_ok=True)
        (curated/'verification.json').write_text(json.dumps(report, indent=2)+'\n')
        audit_report = output/'audit.json'
        if audit_report.exists():
            shutil.copy2(audit_report, curated/audit_report.name)
        schema_report = output/'real_schema_equivalence.json'
        if schema_report.exists():
            shutil.copy2(schema_report, curated/schema_report.name)
        print(json.dumps(dict(status=report['status'], checks=len(report['checks']),
                              smoke_tasks=len(report['smoke_results']), report=str(output/'report.md')), indent=2))
    if report['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
