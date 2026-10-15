"""Summarize an explicit failure-aware method-record table.

One record is one object/seed/method. Failed or missing models remain records,
but only finite, valid paired endpoints enter effects. Cohorts stay separate;
seed zero and the preplanned repeated-seed panel are never pooled as objects.
"""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def summarize(data, metric):
    import numpy as np
    if 'measurements' in data and 'coverage' in data:
        # Canonical audited export: measurements have already passed finite-model
        # checks; invalid fits remain explicit in invalid_completed_fits/trials.
        # A model hash is required so a bare, unaudited score table is not enough.
        metrics = ['mean_view_psnr', 'pooled_psnr', 'pooled_mse', 'lpips', 'ssim', 'surface_psnr']
        data = dict(state=data['state'], objects=data['coverage'],
                    records=[dict(uid=r['uid'], cohort=r['cohort'], seed=r['seed'], method=r['method'],
                                  sources=r.get('sources'), valid_model=isinstance(r.get('model_sha256'),str) and len(r['model_sha256'])==64,
                                  metrics={k:r.get(k) for k in metrics}) for r in data['measurements']],
                    comparisons=[dict(method=f'warm{g}', reference=ref, budget=g)
                                 for g in [4000,3000] for ref in ['HKTex','restart',f'capped{g}',f'capped{g}_restart',f'posthoc{g}',f'posthoc_rebuilt{g}']])
    indexed = {}
    for row in data['records']:
        key = (row['cohort'], row['uid'], row['seed'], row['method'])
        if key in indexed:
            raise ValueError('Duplicate object/seed/method record: ' + repr(key))
        indexed[key] = row
    comparisons = data.get('comparisons', [])
    groups = defaultdict(list)
    pairs = []
    missing = []
    for obj in data['objects']:
        for comparison in comparisons:
            budget = comparison.get('budget')
            key = (obj['cohort'], obj['uid'], 0)
            warm = indexed.get((*key, comparison['method']))
            ref = indexed.get((*key, comparison['reference']))
            reason = None
            parent = indexed.get((*key, data.get('parent_method', 'HKTex')))
            if budget is not None and parent and parent.get('sources') is not None and parent['sources'] <= budget:
                reason = 'budget_ineligible'
            elif not warm or not ref:
                reason = 'missing_endpoint'
            elif warm.get('valid_model') is not True or ref.get('valid_model') is not True:
                reason = 'invalid_or_unverified_model'
            elif not finite(warm.get('metrics', {}).get(metric)) or not finite(ref.get('metrics', {}).get(metric)):
                reason = 'nonfinite_or_missing_metric'
            common = dict(uid=obj['uid'], cohort=obj['cohort'], method=comparison['method'],
                          reference=comparison['reference'], budget=budget)
            if reason:
                missing.append(dict(**common, reason=reason))
                continue
            delta = warm['metrics'][metric] - ref['metrics'][metric]
            pair = dict(**common, delta=delta)
            pairs.append(pair)
            groups[(obj['cohort'], comparison['method'], comparison['reference'], budget)].append(delta)
    summary = []
    for (cohort, method, reference, budget), values in sorted(groups.items(), key=lambda item: repr(item[0])):
        x = np.asarray(values, dtype=float)
        bootstrap = np.random.default_rng(2800922).choice(x, size=(10000, len(x)), replace=True).mean(1)
        summary.append(dict(cohort=cohort, method=method, reference=reference, budget=budget,
                            objects=len(values), mean=float(x.mean()), median=float(np.median(x)),
                            positive=int((x > 0).sum()), negative=int((x < 0).sum()),
                            bootstrap95=np.quantile(bootstrap, [.025, .975]).tolist()))
    return dict(state=data.get('state', 'unspecified'), metric=metric,
                planned_objects=len(data['objects']), records=len(data['records']),
                seed0_summary=summary, paired=pairs, missing=missing,
                scope='Descriptive object bootstrap. Positive means numerically larger; LPIPS/MSE improve when negative. No pooling of cohorts or seeds.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--metric', default='mean_view_psnr')
    args = parser.parse_args()
    result = summarize(json.loads(args.input.read_text()), args.metric)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    for key in ['seed0_summary', 'paired', 'missing']:
        rows = result[key]
        with (args.output / (key + '.csv')).open('w', newline='') as handle:
            if rows:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    print(json.dumps(dict(state=result['state'], planned_objects=result['planned_objects'],
                         paired=len(result['paired']), excluded=len(result['missing']))))


if __name__ == '__main__':
    main()
