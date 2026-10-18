"""Audit public files for machine paths, undeclared imports, and release debris."""
import ast
import importlib.util
import json
from pathlib import Path
import re
import sys

try:
    import tomllib
except ImportError:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DIRS = ('src', 'scripts', 'tests', 'reproduction', '.github')
PUBLIC_FILES = ('README.md', 'LICENSE', 'NOTICE', 'pyproject.toml', 'requirements.txt', 'MANIFEST.in', '.gitignore')
SKIP = {'__pycache__', '.pytest_cache', 'build', 'dist'}
ALIASES = {'yaml': 'pyyaml', 'sklearn': 'scikit-learn', 'keras_preprocessing': 'keras-preprocessing',
           'rtdl_num_embeddings': 'rtdl-num-embeddings', 'PIL': 'pillow'}


def public_files():
    for name in PUBLIC_FILES:
        yield ROOT/name
    for directory in PUBLIC_DIRS:
        for path in sorted((ROOT/directory).rglob('*')):
            if path.is_file() and path.name != '.DS_Store' and not path.name.startswith('._') and not any(p in SKIP or p.endswith('.egg-info') for p in path.relative_to(ROOT).parts):
                yield path


def audit():
    config = tomllib.loads((ROOT/'pyproject.toml').read_text())
    dependencies = config['project']['dependencies']
    for group in config['project'].get('optional-dependencies', {}).values():
        dependencies += group
    normalized = lambda x: re.split(r'[<>=!~;\[ ]', x, maxsplit=1)[0].lower().replace('_', '-')
    declared = {normalized(item) for item in dependencies}
    internal = {'scalarlens', 'fuxictr'} | {p.stem for p in (ROOT/'scripts').glob('*.py')}
    imports = set()
    errors = []
    files = list(public_files())
    # Construct prefixes in parts so this auditor does not flag its own rules.
    private_prefixes = ('/'+'Users/', '/'+'home/', '/'+'mnt/nas/', '/'+'ossfs/', '/'+'tmp/CTR-NumEmb')
    for path in files:
        relative = str(path.relative_to(ROOT))
        if path.name.startswith('._') or path.name == '.DS_Store' or path.suffix in ('.pt', '.pth', '.model', '.ckpt', '.parquet', '.pyc'):
            errors.append(f'Unexpected payload: {relative}')
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            errors.append(f'Unexpected binary file: {relative}')
            continue
        for prefix in private_prefixes:
            if prefix in text:
                errors.append(f'Machine-specific path in {relative}')
                break
        if path.suffix != '.py':
            continue
        tree = ast.parse(text, filename=relative)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split('.')[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imports.add(node.module.split('.')[0])
    third_party = imports-set(sys.stdlib_module_names)-internal-{'tomllib'}
    missing = sorted(name for name in third_party if normalized(ALIASES.get(name, name)) not in declared)
    errors += [f'Undeclared third-party import: {name}' for name in missing]
    # Triton is deliberately optional on CPU/macOS.
    optional = {'triton'}
    if importlib.util.find_spec('tomllib') is not None:
        optional.add('tomli')
    unavailable = sorted(name for name in third_party if name not in optional and importlib.util.find_spec(name) is None)
    errors += [f'Missing installed dependency: {name}' for name in unavailable]
    result = dict(status='PASS' if not errors else 'FAIL', public_files=len(files),
                  third_party_imports=sorted(third_party), undeclared=missing,
                  unavailable_required=unavailable, errors=errors,
                  optional_triton_available=importlib.util.find_spec('triton') is not None)
    return result


def main():
    result = audit()
    destination = ROOT/'outputs/validation/audit.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    if result['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
