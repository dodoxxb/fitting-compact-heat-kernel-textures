"""Fit uniform/projected-leverage controls from an explicit sparse response bundle."""
from pathlib import Path
import argparse,json,time
import numpy as np
from scipy.sparse import load_npz
from mechanism_math import importance_probabilities
from solver import solve

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--response',type=Path,required=True);p.add_argument('--data',type=Path,required=True,help='NPZ with target [N,3] and C0 [K+1,3]; final response column is background.')
 p.add_argument('--fit',default='81920:147456');p.add_argument('--validation',default='147456:155648');p.add_argument('--test',default='155648:163840');p.add_argument('--sample-count',type=int,default=16384);p.add_argument('--projection-rank',type=int,default=128);p.add_argument('--seeds',default='101,202,303');p.add_argument('--max-sweeps',type=int,default=4000);p.add_argument('--gap',type=float,default=1e-7);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():p.error('Use a new output path')
 A=load_npz(a.response).astype(float);z=np.load(a.data);Y=z['target'].astype(float);C0=z['C0'].astype(float)
 if A.shape[0]!=len(Y) or A.shape[1]!=len(C0):p.error('Input shape mismatch')
 slices=[]
 for arg in [a.fit,a.validation,a.test]:
  s,e=map(int,arg.split(':'))
  if not 0<=s<e<=len(Y):p.error('Invalid split')
  slices.append((s,e))
 for i,(s,e) in enumerate(slices):
  for t,u in slices[i+1:]:
   if max(s,t)<min(e,u):p.error('Fit/validation/test must be disjoint')
 (s,e),(v,w),(j,k)=slices;B=A[s:e];yf=Y[s:e];V=A[v:w];yv=Y[v:w];E=A[j:k];ye=Y[j:k];n=len(yf)
 rng=np.random.default_rng(20261015);t=time.perf_counter();P=rng.choice([-1.,1.],size=(A.shape[1],a.projection_rank))/np.sqrt(a.projection_rank);prob=importance_probabilities(B@P);prep=time.perf_counter()-t;rows=[]
 mse=lambda x,y:float(np.mean((np.clip(x,0,1)-y)**2));baseline=mse(E@C0,ye);lower=-np.ones_like(C0);lower[-1]=0
 for seed in map(int,a.seeds.split(',')):
  for name,weights in [('uniform',np.full(n,1/n)),('projected_leverage',prob)]:
   start=time.perf_counter();ids=np.random.default_rng(seed).choice(n,a.sample_count,replace=True,p=weights);scale=1/np.sqrt(n*weights[ids]);M=B[ids].multiply(scale[:,None]).tocsr();target=yf[ids]*scale[:,None];C=C0.copy();sweeps=0
   while sweeps<a.max_sweeps:
    C,h=solve(M,target,C,sweeps=min(200,a.max_sweeps-sweeps));sweeps+=len(h);g=M.T@(M@C-target);gap=float(np.sum(g*(C-np.where(g>=0,lower,1.)))/target.size)
    if gap<=a.gap:break
   alpha=min([0.,.125,.25,.5,1.],key=lambda x:mse(V@(C0+x*(C-C0)),yv))
   rows.append(dict(method=name,seed=seed,half_mse_gap=gap,qualified=gap<=a.gap,sweeps=sweeps,alpha=alpha,full_test_delta_db=float(10*np.log10(baseline/mse(E@C,ye))),selected_test_delta_db=float(10*np.log10(baseline/mse(E@(C0+alpha*(C-C0)),ye))),work_seconds=time.perf_counter()-start))
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(dict(state='complete',projection_seconds=prep,rows=rows,scope='Finite-pool diagnostic; requires user-supplied matrices. Empirical gap is not a rendering guarantee.'),indent=2,allow_nan=False));print(a.output)
if __name__=='__main__':main()
