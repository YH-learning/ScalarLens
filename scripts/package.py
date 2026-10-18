"""Build an audited source archive from an explicit publication allowlist."""
import hashlib
import json
from pathlib import Path
import zipfile
from audit import audit, public_files

ROOT = Path(__file__).resolve().parents[1]


def main():
    report = audit()
    if report['status'] != 'PASS':
        raise RuntimeError(f'Publication audit failed: {report["errors"]}')
    manifest = ROOT/'reproduction/metadata/release_manifest.json'
    candidates = sorted(path for path in public_files() if path != manifest)
    entries = [dict(file=str(path.relative_to(ROOT)), bytes=path.stat().st_size,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in candidates]
    manifest.write_text(json.dumps(dict(files=entries, file_count=len(entries),
                        scope='Explicit source/documentation/reproduction allowlist; no generated outputs'), indent=2)+'\n')
    destination = ROOT/'outputs/ScalarLens-source.zip'
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in [*candidates, manifest]:
            archive.write(path, 'ScalarLens/'+str(path.relative_to(ROOT)))
    print(json.dumps(dict(archive=str(destination), bytes=destination.stat().st_size,
                          source_files=len(entries)+1, sha256=hashlib.sha256(destination.read_bytes()).hexdigest()), indent=2))


if __name__ == '__main__':
    main()
