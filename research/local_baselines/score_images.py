"""Exact original functional metrics on saved sRGB UInt8 images; no rendering."""
from pathlib import Path
import sys,json,os
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
os.environ.setdefault('TORCH_HOME',str(ROOT/'research/local_baselines/torch_cache'))
from hktex.utils.metrics import compute_all_image_metrics
torch.set_num_threads(4)
def score(directory):
 out=Path(directory);report=json.loads((out/'render_results.json').read_text());gt=np.load(out/'UV_native_images.npz')['srgb_uint8'];t=torch.as_tensor(gt.astype(np.float32)/255,device='cuda');results={}
 for name in report['methods']:
  raw=np.load(out/f'{name}_images.npz')['srgb_uint8'];p=torch.as_tensor(raw.astype(np.float32)/255,device='cuda');rows=[]
  for i in range(len(p)):
   met=compute_all_image_metrics(p[i:i+1],t[i:i+1]);met['mse']=float((p[i]-t[i]).square().mean());rows.append(met)
  mse=float((p-t).square().mean());results[name]={'views':rows,'mean_per_view':{k:float(np.mean([r[k] for r in rows])) for k in rows[0]},'pooled_mse':mse,'pooled_psnr':float('inf') if mse==0 else -10*np.log10(mse),'image_uint8_minmax':[int(raw.min()),int(raw.max())]};print(out.name,name,results[name]['mean_per_view'],flush=True)
 report['metrics_state']='complete';report['image_metrics']=results;report['metric_source']='hktex/utils/metrics.py compute_all_image_metrics on sRGB uint8/255, original SqueezeNet LPIPS normalizeTrue';(out/'render_results.json').write_text(json.dumps(report,indent=2));return results
if __name__=='__main__':score(sys.argv[1])
