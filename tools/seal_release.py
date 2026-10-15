"""Seal a reviewed file allowlist into deterministic source/report archives.

The allowlist is explicit: generated outputs and unlisted local artifacts are
never swept into a distribution. Review release_files.txt when adding a file.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, help='Optional .tar.gz output')
    parser.add_argument('--zip', type=Path, help='Optional .zip output')
    args = parser.parse_args()
    names = [line.strip() for line in (ROOT/'release_files.txt').read_text().splitlines() if line.strip() and not line.startswith('#')]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate allowlist entry')
    hashes = {}
    for name in names:
        p = ROOT/name
        if p.is_symlink() or not p.is_file() or not p.resolve().is_relative_to(ROOT):
            raise ValueError('Missing or unsafe path: ' + name)
        if p.suffix in {'.pt','.pth','.npz','.so','.glb','.obj','.ply'}:
            raise ValueError('Unexpected model/data/runtime binary: ' + name)
        raw = p.read_bytes()
        if p.suffix not in {'.png','.pdf'}:
            text = raw.decode('utf-8')
            if re.search(r'(?<![A-Za-z0-9_])/(?:data|home|mnt)/[^\s\x22\x27]+',text):
                raise ValueError('Machine-local absolute path: ' + name)
            if re.search(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',text):
                raise ValueError('Private-key marker: ' + name)
        elif not name.startswith('paper/'):
            raise ValueError('Report binaries only: ' + name)
        hashes[name] = hashlib.sha256(raw).hexdigest()
    (ROOT/'release_manifest.json').write_text(json.dumps(dict(state='ready_for_author_review',files=hashes,
        scope='Reviewed software, scalar evidence and report assets; no raw dataset/model/runtime binaries.'),indent=2)+'\n')
    names = sorted([*names,'release_manifest.json'])
    prefix = 'fitting-compact-heat-kernel-textures'
    if args.zip:
        args.zip.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(args.zip,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
            for name in names:
                info=zipfile.ZipInfo(prefix+'/'+name,date_time=(2026,10,14,0,0,0))
                info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o100644 << 16
                archive.writestr(info,(ROOT/name).read_bytes())
    if args.archive:
        args.archive.parent.mkdir(parents=True,exist_ok=True)
        def normalize(info):
            info.uid=info.gid=0;info.uname=info.gname='';info.mtime=0;info.mode=0o644
            return info
        with args.archive.open('wb') as dest:
            with gzip.GzipFile(filename='',mode='wb',fileobj=dest,mtime=0) as stream:
                with tarfile.open(fileobj=stream,mode='w') as archive:
                    for name in names:
                        archive.add(ROOT/name,arcname=prefix+'/'+name,recursive=False,filter=normalize)
    print(json.dumps(dict(state='sealed',files=len(hashes),zip=str(args.zip),archive=str(args.archive))))


if __name__ == '__main__':
    main()
