from pathlib import Path
import json,hashlib
import numpy as np
from mechanism_math import image_change
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def collect():
 renders=[]
 for path in sorted((HERE/'renders').glob('*/results.json')):
  r=read(path);assert r['state']=='complete';rows=[]
  for res in r['resolutions']:
   y=np.load(path.parent/f'{res}_target.npz');i=np.load(path.parent/f'{res}_alpha_0.npz');j=np.load(path.parent/f'{res}_alpha_1.npz');base=r['methods'][f'{res}_alpha_0']
   for alpha in r['alpha_grid']:
    k=f'{res}_alpha_{alpha:g}';rec=dict(r['methods'][k]);im=np.load(path.parent/f'{res}_alpha_{alpha:g}.npz');rec['delta_mean_view_psnr']=rec['srgb']['mean_view_psnr']-base['srgb']['mean_view_psnr'];rec['delta_pooled_psnr']=rec['srgb']['pooled_psnr']-base['srgb']['pooled_psnr'];rec['raw_linear_delta_pooled_psnr']=rec['linear']['pooled_psnr']-base['linear']['pooled_psnr'];rec['decomposition']={}
    for key,scale in [('srgb_uint8',255),('linear_raw',1)]:
     d=image_change(y[key]/scale,i[key]/scale,im[key]/scale);assert abs(d['identity_residual'])<1e-12
     chord=i[key].astype(float)+alpha*(j[key].astype(float)-i[key].astype(float));d['actual_image_vs_endpoint_chord_rms']=float(np.sqrt(np.mean(((im[key].astype(float)-chord)/scale)**2)));rec['decomposition'][key]=d
    rows.append(rec)
  renders.append(dict(name=r['name'],uid=r['uid'],surface_selected_alpha=r['surface_selected_alpha'],surface_validation_curve=r['surface_validation_curve'],gates=r['gates'],rows=rows,inclusive_seconds=r['inclusive_seconds'],torch_peak_allocated_bytes=r['torch_peak_allocated_bytes'],source_record_sha256=sha(path)))
 assert len(renders)==4
 def allrows(folder):
  files=sorted((HERE/folder).glob('*/results.json'));assert len(files)==4
  result=[]
  for p in files:
   r=read(p);assert r['state']=='complete';result+=r['rows']
  return result
 result=dict(date='2026-10-15',state='complete',scope='Exploratory mechanisms on reused development objects, not new independent-object evidence.',renders=renders,sampling=allrows('sampling'),weighted_sampling=allrows('weighted_sampling'),heat=read(HERE/'heat_results.json'),previous_diagnostics=read(ROOT/'research/theory_review_20261015/results.json'),identity_verification=read(HERE/'identity_verification.json'))
 files=list((HERE/'converged').glob('*/results.json'))
 if len(files)==4 and all(read(f)['state']=='complete' for f in files):result['converged']=[dict(name=read(f)['name'],uid=read(f)['uid'],**x) for f in sorted(files) for x in read(f)['rows']]
 (HERE/'records.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');return result
if __name__=='__main__':
 r=collect()
 for res in [128,256,512]:
  for alpha in [.25,1.]:
   v=[x['delta_mean_view_psnr'] for o in r['renders'] for x in o['rows'] if x['resolution']==res and x['alpha']==alpha];print(res,alpha,np.mean(v),sum(x>0 for x in v),v)
 print('counts',len(r['sampling']),len(r['weighted_sampling']),len(r.get('converged',[])))
