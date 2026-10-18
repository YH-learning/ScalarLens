"""Build a wheel and test its imports/training outside the source checkout."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from audit import public_files

ROOT = Path(__file__).resolve().parents[1]


def build_wheel(log):
    # Build from a temporary source copy so setuptools cannot clutter the root.
    with tempfile.TemporaryDirectory(prefix='scalarlens-build-') as directory:
        source = Path(directory)/'source'
        for path in public_files():
            destination = source/path.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        wheels = ROOT/'outputs/wheels'
        wheels.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, '-m', 'pip', 'wheel', '--no-deps', '--no-build-isolation',
                        str(source), '-w', str(wheels)], stdout=log, stderr=subprocess.STDOUT, check=True)
    return wheels/'scalarlens_ctr-0.1.0-py3-none-any.whl'


def main():
    target = ROOT/'outputs/validation/wheel.log'
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('w') as log:
        wheel = build_wheel(log)
        with tempfile.TemporaryDirectory(prefix='scalarlens-wheel-') as temporary:
            temporary = Path(temporary)
            installed = temporary/'installed'
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--target', str(installed), str(wheel)],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
            env = dict(os.environ, PYTHONPATH=str(installed), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
            def call(*args):
                subprocess.run([sys.executable, *args], cwd=temporary, env=env,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            call('-c', 'import scalarlens, fuxictr; '
                 'assert "installed" in scalarlens.__file__; '
                 'assert "installed" in fuxictr.__file__; '
                 'from scalarlens import ScalarLensEmbedding, StableCoordinates; '
                 'from scalarlens.protocol import training_params; '
                 'assert training_params("DCNv2")["numeric_embedding"] == "scalarlens"; '
                 'print(scalarlens.__file__)')
            for module in ('automl', 'criteo_reference', 'criteo_raw'):
                call('-m', 'scalarlens.data.'+module, '--help')
            data = temporary/'data'
            output = temporary/'trained'
            call('-m', 'scalarlens.data', '--dataset-dir', str(data), '--make-smoke')
            call('-m', 'scalarlens.train', '--dataset-dir', str(data), '--output-dir', str(output),
                 '--gpu', '-1', '--workers', '0', '--epochs', '2', '--batch-size', '128')
            row = json.loads((output/'result.json').read_text())
            row['checkpoint'] = '<temporary wheel smoke checkpoint; not distributed>'
            report = dict(status='PASS', wheel=wheel.name, scope='Installed wheel, outside checkout, CPU synthetic data',
                          result=row)
            (ROOT/'outputs/validation/wheel.json').write_text(json.dumps(report, indent=2)+'\n')
            curated = ROOT/'reproduction/checks'
            curated.mkdir(parents=True, exist_ok=True)
            (curated/'wheel.json').write_text(json.dumps(report, indent=2)+'\n')
            print(json.dumps(dict(status='PASS', wheel=wheel.name)))


if __name__ == '__main__':
    main()
