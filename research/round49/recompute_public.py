"""Recompute final cohort and cost summaries from the exported scalar records.

This does not recreate omitted images, meshes, training or GPU timing.
Only the standard library is needed. Object means, not seeds, are units.
"""
from pathlib import Path
import argparse,json,math,statistics

def summarize(root):
    records=root/'results/final_followup'
    read=lambda name:json.loads((records/name).read_text())
    c=read('confirmation.json');local=read('local.json');replay=read('REPLAY_RESULTS.json');ledger=read('COMPLETED_COHORTS.json')
    qualified=[o for o in c['objects'] if o.get('primary_qualified')]
    def seed_mean(obj):
        assert len(obj['seeds'])==3
        return statistics.mean(s['arms']['guarded_reset']['endpoint']['pbr_psnr']-s['arms']['reset_only']['endpoint']['pbr_psnr'] for s in obj['seeds'])
    gains=[seed_mean(o) for o in qualified]
    assert all(math.isclose(v,o['mean_seed_contrasts']['guarded_vs_reset']['pbr_psnr'],abs_tol=1e-12) for v,o in zip(gains,qualified))
    assert len(qualified)==19 and len(c['objects'])==24
    assert math.isclose(statistics.mean(gains),ledger['primary']['pbr_psnr']['mean'],abs_tol=1e-12)
    loc=[seed_mean(o) for o in local['objects']]
    assert all(math.isclose(v,o['seed_mean_deltas']['guarded_vs_reset']['pbr_psnr'],abs_tol=1e-12) for v,o in zip(loc,local['objects']))
    assert len(loc)==4
    pairs=[]
    for p in replay['pairs']:
        assert p['qualified'] and p['gpu_timing_qualified']
        a=p['variants']['heat_features']['endpoint'];b=p['variants']['fp64_diagonal']['endpoint']
        delta=a['pbr_psnr']-b['pbr_psnr']
        assert math.isclose(delta,p['delta']['pbr_psnr'],abs_tol=1e-12)
        budget=p['common_budget']['seconds'];chosen={}
        for key,v in p['variants'].items():
            chosen[key]=max((x for x in v['curve'] if x['accounted_seconds']<=budget),key=lambda x:x['accounted_seconds'])
        common=chosen['heat_features']['pbr_psnr']-chosen['fp64_diagonal']['pbr_psnr']
        assert math.isclose(common,p['common_budget']['pbr_delta'],abs_tol=1e-12)
        pairs.append(dict(uid=p['pair']['uid'],seed=p['pair']['seed'],pbr_delta=delta,native_delta=a['native_psnr']-b['native_psnr'],
                          accounted_total_percent_increase=100*(a['accounted_full_seconds']/b['accounted_full_seconds']-1),
                          discrete_common_budget_pbr_delta=common))
    assert len(pairs)==4 and len({p['uid'] for p in pairs})==2
    return dict(state='passed',scope=__doc__.strip(),confirmation=dict(declared=24,qualified=len(gains),completed_runs=ledger['completed_training_cells'],
        mean=statistics.mean(gains),median=statistics.median(gains),positive=sum(x>0 for x in gains),attempted_corrections=ledger['attempted_fits'],accepted_corrections=ledger['accepted_fits']),
        partial_local=dict(objects=4,mean=statistics.mean(loc),positive=sum(x>0 for x in loc)),
        heat_replay=dict(objects=2,pairs=pairs,mean_pbr_delta=statistics.mean(p['pbr_delta'] for p in pairs)))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]);p.add_argument('--output',type=Path)
    a=p.parse_args();text=json.dumps(summarize(a.root),indent=2)
    if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(text+'\n')
    print(text)

if __name__=='__main__':main()
