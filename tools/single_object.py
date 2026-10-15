"""Portable single-object entry using the frozen training/calibration routines.

The scientific recipe always uses 5,000 updates per fit. A parent-only run is
an installation check, not a replacement experiment. No automatic numerical
rescue or mesh repair is performed. Run from any working directory.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh', type=Path, required=True, help='Textured triangular GLB; other formats must be explicitly converted.')
    parser.add_argument('--gpu', default='0', help='Visible GPU ordinal or UUID; exactly one GPU.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--budget', type=int, default=3000)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, help='Optional compatible spectral .pt cache; no training checkpoint reuse.')
    parser.add_argument('--through', choices=['parent', 'fits', 'evaluation'], default='evaluation')
    parser.add_argument('--resume-after-parent', action='store_true', help='Continue a successfully completed --through parent run with the same mesh, seed and budget.')
    parser.add_argument('--plan', action='store_true', help='Validate options and print the stage plan without imports or GPU work.')
    args = parser.parse_args()
    if args.mesh.suffix.lower() != '.glb':
        parser.error('Use a textured GLB; the original loader infers format from asset.glb.')
    if args.budget < 1 or ',' in args.gpu:
        parser.error('A positive budget and exactly one GPU are required.')
    mesh = args.mesh.resolve()
    out = args.output.resolve()
    if args.plan:
        print(json.dumps(dict(mesh=str(mesh), gpu=args.gpu, seed=args.seed, budget=args.budget,
                              output=str(out), steps_per_fit=5000, through=args.through,
                              stages=['spectral_cache', 'parent', 'calibration', 'posthoc',
                                      'capped', 'warm', 'full_restart', 'capped_restart', 'evaluation'],
                              scope='No GPU work. Full plan depends on parent kernel-count eligibility.'), indent=2))
        return
    if not mesh.is_file():
        parser.error('Mesh does not exist.')
    previous = None
    if args.resume_after_parent:
        if not (out/'run.json').is_file():
            parser.error('No completed parent-only run found.')
        previous = json.loads((out/'run.json').read_text())
        if previous.get('state') != 'parent_only_complete' or previous.get('seed') != args.seed or previous.get('budget') != args.budget or previous.get('mesh_sha256') != sha(mesh):
            parser.error('Resume requires the same successfully completed parent-only recipe.')
        if args.through == 'parent':
            parser.error('Resume must request fits or evaluation.')
    elif out.exists() and any(out.iterdir()):
        parser.error('Output must be new or empty; previous runs are never overwritten.')
    native = ROOT / 'research/round15/geometry_diagnostic/boundary_check/build.json'
    if not native.is_file():
        parser.error('Build the corrected geodesic extension first: python tools/build_native.py')
    if not Path(json.loads(native.read_text())['library']).is_file():
        parser.error('Native library was moved or removed; rebuild it on this machine.')
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, CUDA_DEVICE_ORDER='PCI_BUS_ID',
               TORCH_COMPILE_DISABLE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8',
               OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
    env['PYTHONPATH'] = str(ROOT) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    stages = list(previous['stage_records']) if previous else []
    ledger = dict(state='running', seed=args.seed, budget=args.budget, steps_per_fit=5000,
                  mesh_sha256=sha(mesh), stage_records=stages,
                  scope='Single-object reproduction; training TF32/high, evaluation TF32 off/highest.')

    def save():
        (out / 'run.json').write_text(json.dumps(ledger, indent=2))

    def run(label, entry, argv):
        command = [sys.executable, str(ROOT / entry), *map(str, argv)]
        record = dict(stage=label, command=command, state='running')
        stages.append(record)
        save()
        started = time.monotonic()
        print('Starting', label, flush=True)
        with (out / f'{label}.log').open('w') as log:
            result = subprocess.run(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        record.update(state='complete' if result.returncode == 0 else 'failed',
                      returncode=result.returncode, seconds=time.monotonic()-started)
        save()
        if result.returncode:
            raise RuntimeError(f'{label} failed; inspect {out / (label + ".log")}')

    def setup(folder, cache):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'asset.glb').symlink_to(mesh)
        (folder / 'asset_eigen_albo_principal_curvatures.pt').symlink_to(cache)
        (folder / 'cache_provenance.json').write_text(json.dumps(dict(
            name=mesh.stem, split='single_object_reproduction', mesh=str(mesh),
            mesh_sha256=ledger['mesh_sha256'], cache=str(cache), cache_sha256=sha(cache))))

    def fit(name, cache, initializer=None):
        folder = out / name
        setup(folder, cache)
        argv = ['--teacher', folder, '--seed', args.seed, '--arm', name]
        if name == 'parent':
            argv[-1] = 'original'
        else:
            sys.path.insert(0, str(ROOT / 'research/round22'))
            from gen_hooks import render, SRC_SEED
            import numpy as np
            budget = len(np.load(initializer)['_kernel_colours']) if initializer else args.budget
            source = render(name, budget, 'oracle', 'warm' if initializer else 'sched', 5000, 'seed')
            if initializer:
                source = source.replace(SRC_SEED, 'SRC = ' + repr(str(initializer)))
            hooks = ROOT / 'research/round17/training/generated_hooks'
            hooks.mkdir(parents=True, exist_ok=True)
            hook = hooks / (hashlib.sha256(str(out).encode()).hexdigest()[:16] + '_' + name + '.py')
            if hook.exists():
                raise RuntimeError('Generated hook already exists; preserve prior output and use a new output path.')
            compile(source, str(hook), 'exec')
            hook.write_text(source)
            argv += ['--candidate-hook', hook]
        run(name, 'research/round22/train_nativefix.py', argv)
        import numpy as np
        with np.load(folder / 'model.npz', allow_pickle=False) as model:
            if not all(np.isfinite(v).all() for v in model.values()):
                raise RuntimeError(f'Nonfinite model: {name}; finite loss is insufficient.')
            count = len(model['_kernel_colours'])
        if initializer:
            with np.load(initializer, allow_pickle=False) as initial:
                if count != len(initial['_kernel_colours']):
                    raise RuntimeError('A fixed-count continuation changed the kernel count.')
        status = json.loads((folder / 'status.json').read_text())
        trace = json.loads((folder / 'loss_trace.json').read_text())
        if status.get('actual_updates') != 5000 or len(trace['loss']) != 5000:
            raise RuntimeError('Scientific fit did not complete exactly 5000 updates.')
        return folder / 'model.npz'

    try:
        save()
        if previous:
            cache = (out/'parent/asset_eigen_albo_principal_curvatures.pt').resolve()
            if sha(cache) != previous['cache_sha256']:
                raise RuntimeError('The parent spectral cache changed.')
            parent = out/'parent/model.npz'
            import numpy as np
            with np.load(parent, allow_pickle=False) as model:
                if not all(np.isfinite(v).all() for v in model.values()):
                    raise RuntimeError('The resumed parent is not finite.')
            if json.loads((out/'parent/status.json').read_text()).get('actual_updates') != 5000:
                raise RuntimeError('The resumed parent did not complete 5000 updates.')
            ledger['cache_scope'] = 'Cache preserved from the completed parent-only stage.'
        elif args.cache:
            cache = args.cache.resolve()
            if not cache.is_file():
                raise FileNotFoundError(cache)
            ledger['cache_scope'] = 'User-supplied compatible cache; its mesh/config correspondence is the caller\'s responsibility.'
        else:
            run('spectral_cache', 'research/round17/training/cache.py', ['--mesh', mesh, '--output', out / 'cache'])
            cache = (out / 'cache/asset_eigen_albo_principal_curvatures.pt').resolve()
        ledger['cache_sha256'] = sha(cache)
        if previous is None:
            parent = fit('parent', cache)
        if args.through == 'parent':
            ledger['state'] = 'parent_only_complete'
            save()
            return
        import numpy as np
        parent_n = len(np.load(parent)['_kernel_colours'])
        ledger['parent_sources'] = parent_n
        if parent_n <= args.budget:
            ledger['state'] = 'budget_ineligible'
            save()
            return
        run('export', 'tools/export_checkpoint.py', ['--teacher', parent.parent])
        queries = out / 'queries.npz'
        run('queries', 'research/round28/prepare_queries.py', ['--teacher', parent.parent, '--queries', queries])
        run('calibration', 'research/round22/calib_cache.py', ['--teacher', parent.parent, '--output', out / 'calibration'])
        run('compression', 'research/round22/coalesce.py', ['--cache', out / 'calibration/calib_cache.npz', '--model', parent, '--output', out / 'coalesced', '--targets', args.budget])
        initializer = out / f'coalesced/oracle_del{args.budget}_R.npz'
        capped = fit('capped', cache)
        warm = fit('warm', cache, initializer)
        restart = fit('full_restart', cache, parent)
        matched = fit('capped_restart', cache, capped)
        if args.through == 'evaluation':
            pairs = [('posthoc', initializer), ('capped', capped), ('warm', warm), ('full_restart', restart), ('capped_restart', matched)]
            run('evaluation', 'research/round22/evaluate_variants.py', ['--teacher', parent.parent, '--queries', queries,
                 '--output', out / 'evaluation', '--models', ','.join(f'{k}:{v}' for k,v in pairs)])
        ledger['state'] = 'complete'
    except Exception as exc:
        ledger.update(state='failed', error=repr(exc))
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
