"""Bounded CPU diagnostics, never claims a full-space stability certificate."""
from pathlib import Path
import argparse,sys,time,json,hashlib
import numpy as np
from scipy.sparse import load_npz
from scipy.linalg import eigh
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'research/round51'),str(ROOT/'research/round46')]
from anchor_cpu import CASES
from solver import solve

def q(x):return dict(zip(['min','q10','median','q90','max'],map(float,np.quantile(x,[0,.1,.5,.9,1]))))
def mse(z,y):return float(np.mean((np.clip(z,0,1)-y)**2))
def main(name):
 uid=next(u for n,u in CASES if n==name);out=HERE/'sampling'/uid;out.mkdir(parents=True,exist_ok=False);rows=[];t0=time.perf_counter()
 for step in [250,500,750]:
  source=ROOT/'research/round47/diagnostics'/uid/f'{step:04d}';build=json.loads((source/'build.json').read_text());A=load_npz(source/'response.npz').astype(float);data=dict(np.load(source/'data.npz'));Y=data['target'].astype(float);C0=data['C0'].astype(float);E=A[155648:163840];ye=Y[155648:163840]
  rng=np.random.default_rng(20261015);P=rng.choice([-1.,1.],size=(A.shape[1],128))/np.sqrt(128)
  T=A[98304:114688]@P;G=T.T@T/len(T);lam,Q=eigh(G);ridge=1e-6*np.trace(G)/128;W=(Q/np.sqrt(lam+ridge))@Q.T
  ZE=E@P@W;GE=ZE.T@ZE/len(ZE);diagE=np.asarray(E.power(2).mean(axis=0)).ravel();lower=-np.ones_like(C0);lower[-1]=0
  for label,s,e in [('original16k',0,16384),('fresh16k',81920,98304),('larger65k',81920,147456)]:
   t=time.perf_counter();B=A[s:e];yf=Y[s:e];C,hist=solve(B,yf,C0,sweeps=100)
   R=B@C-yf;grad=B.T@R;gap=float(np.sum(grad*(C-np.where(grad>=0,lower,1.)))/yf.size)
   Z=B@P@W;GF=Z.T@Z/len(Z);d=np.asarray(B.power(2).mean(axis=0)).ravel();floor=1e-6*np.mean(d[:-1]);ratio=(diagE[:-1]+floor)/(d[:-1]+floor)
   D=C-C0;ef=float(np.mean((B@D)**2));ee=float(np.mean((E@D)**2));difference=eigh(GE-GF,eigvals_only=True)
   unexposed=(np.asarray(B.getnnz(axis=0)).ravel()[:-1]==0);test_exposure=np.asarray(E[:,:-1].power(2).sum(axis=0)).ravel()
   base=mse(E@C0,ye);new=mse(E@C,ye)
   row=dict(name=name,uid=uid,step=step,method=label,fit_rows=e-s,half_mse_gap=gap,sweeps=len(hist),active_box_fraction=float(np.mean((abs(C[:-1])>1-1e-6))),fit_delta_db=float(10*np.log10(mse(B@C0,yf)/mse(B@C,yf))),test_delta_db=float(10*np.log10(base/new)),direction_energy_fit=ef,direction_energy_test=ee,direction_energy_test_over_fit=ee/ef if ef else None,column_moment_ratio=q(ratio),unexposed_columns=int(unexposed.sum()),test_energy_fraction_unexposed=float(test_exposure[unexposed].sum()/test_exposure.sum()),projected_gram_difference_spectral_norm=float(max(abs(difference))),projected_test_trace_over_fit=float(np.trace(GE)/np.trace(GF)),projection_rank=128,anchor_ridge=ridge,anchor_overlaps_fit=label=='larger65k',seconds=time.perf_counter()-t)
   rows.append(row);print(name,step,label,row['test_delta_db'],row['direction_energy_test_over_fit'],flush=True)
  (out/'results.json').write_text(json.dumps(dict(state='running',rows=rows),indent=2,allow_nan=False))
 (out/'results.json').write_text(json.dumps(dict(state='complete',rows=rows,inclusive_seconds=time.perf_counter()-t0,scope='12 correlated events on four reused objects; 128-dimensional projected Gram diagnostic; no full-space guarantee.',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2,allow_nan=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--object',required=True);main(p.parse_args().object)
