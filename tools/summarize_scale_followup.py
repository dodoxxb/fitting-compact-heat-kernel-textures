"""Recompute the exploratory update-scale appendix from a compact CPU export.

The default command summarises recorded endpoints; it does not train models.
Authors may rebuild the compact export with --export-round31 ORIGINAL_DIRECTORY.
No meshes, model arrays, machine paths or GPU dependencies are redistributed.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('default', 'tenth', 'adaptive', 'direct')
COMPARISONS = (('adaptive', 'default'), ('adaptive', 'tenth'),
               ('adaptive', 'direct'), ('tenth', 'default'), ('default', 'direct'))


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(source):
    """Read the authors' original ledgers, retaining relative provenance only."""
    import numpy as np

    source = source.resolve()
    original_root = source.parents[1]
    raw = read(source / 'results.json')
    cases = {c['uid']: c for c in read(source / 'COHORT.json')}
    if raw['state'] != 'terminal' or raw['states'] != {'complete': 32}:
        raise ValueError('The frozen pilot requires all 32 completed endpoints.')
    provenance = {}
    for worker in sorted(source.glob('worker?.json')):
        metadata = read(worker)
        if metadata['state'] != 'complete':
            raise ValueError('Incomplete worker: ' + worker.name)
        for name, digest in metadata['source_hashes'].items():
            path = Path(name)
            relative = path.relative_to(original_root).as_posix()
            if sha(path) != digest:
                raise ValueError('Frozen source changed: ' + relative)
            provenance[relative] = digest
    for name in ('results.json', 'summarize.py'):
        path = source / name
        provenance[path.relative_to(original_root).as_posix()] = sha(path)
    rows, objects = [], []
    for uid, case in cases.items():
        directory = source / 'runs' / uid
        evaluation_path = directory / 'evaluation/results.json'
        evaluation = read(evaluation_path)
        parent = Path(case['base']) / 'parent/model.npz'
        if sha(parent) != evaluation['source_sha256']:
            raise ValueError('Evaluation reference changed: ' + uid)
        query = Path(case['queries'])
        if sha(query) != evaluation['query_sha256']:
            raise ValueError('Evaluation query changed: ' + uid)
        cache = read(directory / 'adaptive/cache_provenance.json')
        objects.append(dict(uid=uid, name=case['name'], cohort=case['cohort'],
                            seed=0, original_gpu_index=case['gpu'],
                            mesh_sha256=cache['mesh_sha256'],
                            cache_sha256=cache['cache_sha256'],
                            reference_model_sha256=sha(parent),
                            query_sha256=sha(query),
                            renderer_sha256=sha(directory / 'evaluation/renderer_snapshot.py'),
                            evaluation_sha256=sha(evaluation_path)))
        for arm in ARMS:
            candidates = [r for r in raw['rows'] if r['uid'] == uid and r['arm'] == arm]
            if len(candidates) != 1:
                raise ValueError('Missing or duplicate endpoint: ' + uid + '/' + arm)
            row = dict(candidates[0])
            run = directory / arm
            initializer = Path(case['base']) / ('capped3000/model.npz' if arm == 'direct'
                                               else 'coalesced/oracle_del3000_R.npz')
            hook = Path(read(run / 'candidate_hook.json')['hook'])
            if sha(initializer) != row['source_sha256'] or sha(hook) != row['hook_sha256']:
                raise ValueError('Initializer or hook changed: ' + uid + '/' + arm)
            status = read(run / 'status.json')
            sampling = read(run / 'sampling_contract.json')
            trace = read(run / 'scale_trace.json')
            if status['actual_updates'] != 5000 or sampling['mode'] != 'uniform':
                raise ValueError('Unexpected training contract: ' + uid + '/' + arm)
            if len(trace) != 5000 or [r['update'] for r in trace] != list(range(1, 5001)):
                raise ValueError('Incomplete scale trace: ' + uid + '/' + arm)
            model = run / 'model.npz'
            with np.load(model, allow_pickle=False) as arrays:
                if not all(np.isfinite(v).all() for v in arrays.values()):
                    raise ValueError('Nonfinite export: ' + uid + '/' + arm)
            if row['metrics'] != evaluation['methods'][arm]['render']:
                raise ValueError('Evaluation/report mismatch: ' + uid + '/' + arm)
            scales = [r['scale'] for r in trace]
            if not all(math.isfinite(s) and 0 <= s <= 1 for s in scales):
                raise ValueError('Invalid scale: ' + uid + '/' + arm)
            constant = 0.1 if arm == 'tenth' else 1.0
            if arm != 'adaptive' and any(s != constant for s in scales):
                raise ValueError('Unexpected control scale: ' + uid + '/' + arm)
            row.update(actual_updates=5000, model_sha256=sha(model),
                       scale_trace_sha256=sha(run / 'scale_trace.json'),
                       status_sha256=sha(run / 'status.json'),
                       sampling_contract_sha256=sha(run / 'sampling_contract.json'),
                       scale_reason_counts=dict(Counter(r['reason'] for r in trace)))
            if arm == 'adaptive':
                row['scale_below_one'] = [[r['update'], r['scale']] for r in trace if r['scale'] < 1]
            else:
                row['constant_scale'] = constant
            rows.append(row)
    return dict(schema_version=1, state='terminal_exploratory', planned_objects=8,
                planned_fits=32, objects=objects, records=rows,
                original_source_sha256=provenance,
                protocol=dict(
                    original_frozen_date='2026-09-27',
                    publication_scope='Separate exploratory appendix; no changes to Round28/30 confirmation scores or failure records.',
                    selection='First eligible seed-zero reference (>4000 kernels) in frozen order per original GPU, separately for new Objaverse and GSO; original outcomes were already observed.',
                    seed=0, budget=3000, additional_updates=5000,
                    initialization='Default, tenth and adaptive share the calibration-edited/recoloured reference; direct continues the independently fitted budget-limited initializer. Initial fits used 5000 updates.',
                    objective='Inherited source-UV fitting target; all four routes use identical uniform sampling from the fixed pool, rather than the original priority sampler.',
                    controller='For predicted d=grad(L)^T delta<0, s=min(1,0.1*L/(-d+1e-30)); otherwise s=0.1. Scale all group learning rates and restore them before the scheduler. Moments are not reset or rescaled.',
                    controller_limit='First-order prediction before constraints and geodesic tracing; no actual-loss acceptance check, field/render error bound or convergence guarantee.',
                    evaluation='Only the prescribed final update; decoded evaluation with five PBR views and 65536 surface queries. Primary metric is mean-view render PSNR.',
                    timing='All arms calculate the proposal diagnostic; recorded wall times include this instrumentation and do not establish controller overhead or speedup.',
                    scale_encoding='For each adaptive record, scale_below_one contains every [one-based update, scale] with scale<1. All other updates have scale=1. Original full trace hashes are retained.',
                    original_protocol_note='The frozen September protocol initially excluded the pilot from v1. The completed pilot was later admitted only as an explicitly exploratory appendix.',
                    reproducibility='This export and CPU command recompute endpoint and scale summaries; they do not expose a complete training replay or redistribute original models, queries or full traces. Initializer, hook, model and trace hashes identify the original experiment artifacts; hashes alone do not make those artifacts available.'))


def summarize(data):
    if data['state'] != 'terminal_exploratory':
        raise ValueError('A terminal exploratory export is required.')
    indexed = defaultdict(dict)
    for row in data['records']:
        if row['arm'] in indexed[row['uid']]:
            raise ValueError('Duplicate endpoint.')
        if row['state'] != 'complete' or row['finite_export'] is not True:
            raise ValueError('Unexpected incomplete or invalid endpoint.')
        if row['actual_updates'] != 5000 or not math.isfinite(row['metrics']['psnr']):
            raise ValueError('Invalid update count or metric.')
        indexed[row['uid']][row['arm']] = row
    if len(indexed) != 8 or len(data['records']) != 32:
        raise ValueError('Unexpected cohort size.')
    summaries, scale_rows = [], []
    cohorts = ('new_confirmation', 'gso_external')
    for cohort in cohorts:
        objects = [arms for arms in indexed.values() if arms['default']['cohort'] == cohort]
        if len(objects) != 4 or any(set(arms) != set(ARMS) for arms in objects):
            raise ValueError('Incomplete four-arm cohort.')
        for method, reference in COMPARISONS:
            values = [arms[method]['metrics']['psnr'] - arms[reference]['metrics']['psnr'] for arms in objects]
            summaries.append(dict(cohort=cohort, method=method, reference=reference,
                                  n=len(values), mean_db=statistics.mean(values),
                                  median_db=statistics.median(values),
                                  positive=sum(v > 0 for v in values)))
    for uid, arms in indexed.items():
        row = arms['adaptive']
        sparse = row['scale_below_one']
        updates = [r[0] for r in sparse]
        if len(updates) != len(set(updates)) or any(not 1 <= u <= 5000 for u in updates):
            raise ValueError('Invalid sparse scale updates.')
        if any(not math.isfinite(s) or not 0 <= s < 1 for _, s in sparse):
            raise ValueError('Invalid sparse scale values.')
        scales = [s for _, s in sparse] + [1.] * (5000 - len(sparse))
        scale_rows.append(dict(uid=uid, cohort=row['cohort'], updates=5000,
                               shrink_updates=len(sparse), shrink_fraction=len(sparse) / 5000,
                               median_scale=statistics.median(scales), minimum_scale=min(scales)))
    return dict(state=data['state'], objects=len(indexed), fits=len(data['records']),
                metric='mean_view_render_psnr', summaries=summaries,
                adaptive_scale_by_object=scale_rows,
                adaptive_updates=sum(r['updates'] for r in scale_rows),
                adaptive_shrink_updates=sum(r['shrink_updates'] for r in scale_rows),
                scope='Descriptive exploratory follow-up; no confirmation replacement, population confidence or speed claim.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'results/scale_followup.json')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--export-round31', type=Path,
                        help='Author-only export from original experiment directory; writes --input first.')
    args = parser.parse_args()
    if args.export_round31:
        args.input.parent.mkdir(parents=True, exist_ok=True)
        args.input.write_text(json.dumps(export(args.export_round31), indent=2, allow_nan=False) + '\n')
    result = summarize(read(args.input))
    rendered = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end='')


if __name__ == '__main__':
    main()
