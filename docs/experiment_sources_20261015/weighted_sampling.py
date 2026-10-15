"""Equal finite-pool target under inverse-probability weighted subsampling."""
from sampling_probe import *

def weighted(name):
 uid=next(u for n,u in CASES if n==name);out=HERE/'weighted_sampling'/uid;out.mkdir(parents=True,exist_ok=False)
 source=ROOT/'research/round47/diagnostics'/uid/'0500';A=load_npz(source/'response.npz').astype(float);data=dict(np.load(source/'data.npz'));Y=data['target'].astype(float);C0=data['C0'].astype(float)
 B=A[81920:147456];yf=Y[81920:147456];V=A[147456:155648];yv=Y[147456:155648];E=A[155648:163840];ye=Y[155648:163840];n=B.shape[0];lower=-np.ones_like(C0);lower[-1]=0
 t=time.perf_counter();rng=np.random.default_rng(20261015);P=rng.choice([-1.,1.],size=(B.shape[1],128))/np.sqrt(128);Z=B@P;G=Z.T@Z/n;lam,Q=eigh(G);ridge=1e-6*np.trace(G)/128;ZW=Z@(Q/np.sqrt(lam+ridge));ell=np.sum(ZW**2,axis=1);p=.5/n+.5*ell/ell.sum();pre=time.perf_counter()-t
 rows=[]
 for seed in [101,202,303]:
  for label,prob in [('uniform',np.full(n,1/n)),('projected_leverage',p)]:
   t=time.perf_counter();idx=np.random.default_rng(seed).choice(n,16384,replace=True,p=prob);w=1/np.sqrt(n*prob[idx]);M=B[idx].multiply(w[:,None]).tocsr();target=yf[idx]*w[:,None];sampletime=time.perf_counter()-t
   t=time.perf_counter();C,hist=solve(M,target,C0,sweeps=100);solvetime=time.perf_counter()-t;grad=M.T@(M@C-target);gap=float(np.sum(grad*(C-np.where(grad>=0,lower,1.)))/target.size)
   t=time.perf_counter();curve=[dict(alpha=a,mse=mse(V@(C0+a*(C-C0)),yv)) for a in [0.,.125,.25,.5,1.]];alpha=min(curve,key=lambda r:r['mse'])['alpha'];selecttime=time.perf_counter()-t
   baseline=mse(E@C0,ye);full=mse(E@C,ye);selected=mse(E@(C0+alpha*(C-C0)),ye)
   rows.append(dict(name=name,uid=uid,seed=seed,method=label,unique_samples=int(len(np.unique(idx))),half_mse_gap=gap,selected_alpha=alpha,validation_curve=curve,full_test_delta_db=float(10*np.log10(baseline/full)),selected_test_delta_db=float(10*np.log10(baseline/selected)),preprocess_seconds=pre if label=='projected_leverage' else 0.,sample_seconds=sampletime,solve_seconds=solvetime,selection_seconds=selecttime));print(name,seed,label,rows[-1]['full_test_delta_db'],rows[-1]['selected_test_delta_db'],flush=True)
 report=dict(state='complete',name=name,uid=uid,rows=rows,probabilities_min_max=[float(p.min()),float(p.max())],probability_sum=float(p.sum()),max_inverse_probability=float(1/(n*p.min())),scope='Historical follow-up, replacement sampling; same finite-pool quadratic target. Projected leverage not full-space leverage; surface targets not render objective.')
 (out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--object',required=True);weighted(p.parse_args().object)
