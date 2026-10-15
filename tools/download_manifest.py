"""Print a download plan or write a shell script; never download implicitly."""
import argparse
import json
from pathlib import Path
import shlex

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'data/objects.json')
    parser.add_argument('--cohort')
    parser.add_argument('--output', type=Path, default=Path('datasets'))
    parser.add_argument('--shell', type=Path, help='Write a reviewable download script; do not execute it.')
    args = parser.parse_args()
    rows = json.loads(args.manifest.read_text())['objects']
    rows = [r for r in rows if args.cohort is None or r['cohort'] == args.cohort]
    commands = ['#!/usr/bin/env bash', 'set -euo pipefail', '# Fetch original sources directly, without proxy inheritance.']
    plan = []
    for row in rows:
        suffix = '.tar.gz' if row['source'] == 'GSO_Kubric' else '.glb'
        target = args.output / row['source'] / (row['uid'] + suffix)
        plan.append(dict(uid=row['uid'], cohort=row['cohort'], url=row['url'],
                         license=row['license'], sha256=row['download_sha256'], target=str(target)))
        commands += [f'mkdir -p {shlex.quote(str(target.parent))}',
                     'curl --proxy \'\' --fail --location --retry 2 --max-time 180 '
                     + '--output ' + shlex.quote(str(target)) + ' ' + shlex.quote(row['url'])]
        if row['download_sha256']:
            line = row['download_sha256'] + '  ' + str(target)
            commands.append('printf \'%s\\n\' ' + shlex.quote(line) + ' | sha256sum --check -')
    if args.shell:
        args.shell.parent.mkdir(parents=True, exist_ok=True)
        args.shell.write_text('\n'.join(commands) + '\n')
    print(json.dumps(dict(objects=len(plan), plan=plan), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
