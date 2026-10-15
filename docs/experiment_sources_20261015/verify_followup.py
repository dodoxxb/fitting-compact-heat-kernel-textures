from pathlib import Path
import sys,json
import numpy as np
from scipy.sparse import load_npz
from scipy.linalg import eigh
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
from mechanism_math import importance_probabilities
r=json.loads((HERE/'records.json').read_text());rows=[]
for o in r['renders']:
 src=ROOT/'research/round47/diagnostics'/o['uid']/'0500';A=load_npz(src/'response.npz').astype(float);B=A[81920:147456];rng=np.random.default_rng(20261015);P=rng.choice([-1.,1.],size=(B.shape[1],128))/np.sqrt(128);Z=B@P;G=Z.T@Z/len(Z);lam,Q=eigh(G);ridge=1e-6*np.trace(G)/128;ZW=Z@(Q/np.sqrt(lam+ridge));ell=np.sum(ZW**2,axis=1);p=.5/len(Z)+.5*ell/ell.sum();p2=importance_probabilities(Z)
 for seed in [101,202,303]:
  x=np.random.default_rng(seed).choice(len(Z),16384,replace=True,p=p);y=np.random.default_rng(seed).choice(len(Z),16384,replace=True,p=p2);assert np.array_equal(x,y)
 rows.append(dict(name=o['name'],probability_max_abs=float(np.max(abs(p-p2))),all_sample_indices_identical=True))
# Read every recorded result and enforce denominator/count and expected scopes.
assert len(r['sampling'])==36 and len(r['weighted_sampling'])==24 and len(r['converged'])==36
assert all(x['qualified'] and x['half_mse_gap']<=1e-7 for x in r['converged'])
assert all(o['surface_selected_alpha']==.25 for o in r['renders'])
for o in r['renders']:
 assert len(o['rows'])==15
 for x in o['rows']:
  if x['alpha']==.25:assert x['delta_mean_view_psnr']>0
  if x['alpha']==1:assert x['delta_mean_view_psnr']<0
out=dict(state='passed',counts=dict(render_fields=60,rendered_field_views=300,reference_views=60,sampling_fits=36,weighted_fits=24,gap_stopped_fits=36,offgrid_operator_points=18),same_weighted_samples=rows,source_objects=4,operator_meshes=2,new_independent_objects=0)
(HERE/'followup_verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
