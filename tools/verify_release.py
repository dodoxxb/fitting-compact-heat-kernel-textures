"""Verify immutable release-file hashes without importing GPU libraries."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / 'release_manifest.json').read_text())
    for name, expected in manifest['files'].items():
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError('Unsafe path: ' + name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('File hash mismatch: ' + name)
    print(json.dumps(dict(verified_files=len(manifest['files']), state=manifest['state'],
                         scope='Integrity only; no claim of experimental completeness or clean-environment reproducibility.')))


if __name__ == '__main__':
    main()
