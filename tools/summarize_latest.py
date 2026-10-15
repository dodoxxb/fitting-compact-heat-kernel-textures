"""Recompute final report aggregates from exported scalar records, on CPU.

This verifies arithmetic and denominators, not omitted training/image binaries.
Original export failures remain in the records: an explicitly qualified recovered
endpoint may be used without rewriting its original process state.
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def stats(values, lower_better=False):
    x = np.asarray(values, dtype=float)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError('Expected nonempty finite contrasts')
    boot = np.random.default_rng(500013).choice(x, (10000, len(x))).mean(axis=1)
    direction = -x if lower_better else x
    return dict(objects=len(x), mean=float(x.mean()), median=float(np.median(x)),
                improved=int((direction > 0).sum()), worsened=int((direction < 0).sum()),
                object_bootstrap_95=np.quantile(boot, [.025, .975]).tolist())


def close(actual, expected):
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-11):
        raise ValueError(f'Aggregate mismatch: recomputed {actual}, recorded {expected}')


def summarize(root):
    folder = root / 'results/latest'
    read = lambda name: json.loads((folder / name).read_text())
    heat = read('heat_features.json')
    metrics = ['pbr_psnr', 'native_psnr', 'ssim', 'lpips', 'pbr_pooled_psnr']
    objects, singles = [], []
    for obj in heat['objects']:
        seeds = []
        for seed in obj['seeds']:
            if not seed.get('qualified'):
                continue
            base = seed['variants']['fp64_diagonal']['endpoint']
            new = seed['variants']['heat_features']['endpoint']
            delta = {k: new[k] - base[k] for k in metrics}
            for k, value in delta.items():
                close(value, seed['delta'][k])
            seeds.append(dict(seed=seed['seed'], delta=delta))
        row = dict(uid=obj['uid'], name=obj['name'], dataset=obj['dataset'],
                   supported=obj['supported'], qualified=obj['qualified'], seeds=seeds)
        if obj['qualified']:
            if {s['seed'] for s in seeds} != {0, 1}:
                raise ValueError('A primary object must have both prescribed seeds')
            row['mean_seed_delta'] = {k: float(np.mean([s['delta'][k] for s in seeds])) for k in metrics}
            objects.append(row)
        else:
            singles.append(row)
    primary = {k: stats([o['mean_seed_delta'][k] for o in objects], k == 'lpips') for k in metrics}
    for k, result in primary.items():
        for key in ['mean', 'median']:
            close(result[key], heat['primary'][k][key])
        if result['objects'] != heat['primary'][k]['objects']:
            raise ValueError('Object denominator mismatch')
    datasets = {dataset: stats([o['mean_seed_delta']['pbr_psnr'] for o in objects if o['dataset'] == dataset])
                for dataset in sorted({o['dataset'] for o in objects})}
    heat_result = dict(declared=len(heat['objects']), supported=sum(o['supported'] for o in heat['objects']),
                       primary=primary, datasets=datasets, primary_objects=objects, excluded_objects=singles,
                       primary_seed_pairs=sum(len(o['seeds']) for o in objects),
                       primary_positive_seeds=sum(s['delta']['pbr_psnr'] > 0 for o in objects for s in o['seeds']),
                       timing_qualified=False)
    mechanism = read('mechanisms.json')
    interventions = {}
    for arm in mechanism['aggregates']:
        values = [o['arms'][arm]['recovered']['test']['psnr'] - o['arms']['baseline']['recovered']['test']['psnr']
                  for o in mechanism['objects'].values()]
        interventions[arm] = stats(values)
        close(interventions[arm]['mean'], mechanism['aggregates'][arm]['mean_recovered_test_delta'])
    readout = read('readout_controls.json')
    readout_result = {}
    for contrast, recorded in readout['contrasts'].items():
        values = [o['contrasts'][contrast]['psnr_delta'] for o in readout['objects'].values()
                  if finite(o['contrasts'][contrast]['psnr_delta'])]
        readout_result[contrast] = stats(values)
        readout_result[contrast]['excluded'] = len(readout['objects']) - len(values)
        close(readout_result[contrast]['mean'], recorded['mean'])
    routing = read('routing.json')
    rows = []
    for obj in routing['objects']:
        sets = [obj['render'][key] for key in ['original_views', 'shifted_views']]
        ratio = sum(s['Full_stream']['render_denoise_ms'] for s in sets) / sum(s['quad_adaptive']['render_denoise_ms'] for s in sets)
        close(ratio, routing['expansion']['speed_ratio_per_object'][obj['object']])
        rows.append(dict(object=obj['object'], mean_candidates=obj['families']['quad_adaptive']['mean_candidates_test'],
                         native_delta=obj['families']['quad_adaptive']['test_delta_db'],
                         pbr_deltas=[s['quad_adaptive']['delta_db'] for s in sets],
                         full_over_routed_render_speed=ratio, fusion_state=obj['fusion']['state'],
                         fusion_render_qualified=obj['fusion']['full_render_claim_qualified']))
    median = float(np.median([o['full_over_routed_render_speed'] for o in rows]))
    close(median, routing['expansion']['median_speed_ratio'])
    return dict(state='passed_scalar_recomputation', scope=__doc__.strip(), heat_features=heat_result,
                fixed_response_interventions=interventions, readout_controls=readout_result,
                routing=dict(objects=rows, median_speed_ratio=median,
                             speed_pass=routing['expansion']['speed_pass'], decision=routing['expansion']['decision']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, default=Path('outputs/latest_summary.json'))
    args = parser.parse_args()
    result = summarize(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(state=result['state'], heat=result['heat_features']['primary']['pbr_psnr'],
                          routing_median_speed=result['routing']['median_speed_ratio'], output=str(args.output))))


if __name__ == '__main__':
    main()
