from sampling_probe import *
from mechanism_math import importance_probabilities

def converge(M,Y,C0):
 t=time.perf_counter();C=C0.copy();lower=-np.ones_like(C);lower[-1]=0;sweeps=0;history=[]
 while sweeps<4000:
  C,h=solve(M,Y,C,sweeps=min(200,4000-sweeps));sweeps+=len(h);g=M.T@(M@C-Y);gap=float(np.sum(g*(C-np.where(g>=0,lower,1.)))/Y.size);history.append([sweeps,gap])
  if gap<=1e-7:break
 return C,dict(sweeps=sweeps,half_mse_gap=gap,qualified=gap<=1e-7,seconds=time.perf_counter()-t,gap_history=history)
def main(name):
 uid=next(u for n,u in CASES if n==name);out=HERE/'converged'/uid;out.mkdir(parents=True,exist_ok=False);source=ROOT/'research/round47/diagnostics'/uid/'0500';A=load_npz(source/'response.npz').astype(float);data=dict(np.load(source/'data.npz'));Y=data['target'].astype(float);C0=data['C0'].astype(float);V=A[147456:155648];yv=Y[147456:155648];E=A[155648:163840];ye=Y[155648:163840];baseline=mse(E@C0,ye);rows=[]
 def run(label,seed,M,y):
  C,details=converge(M,y,C0);curve=[dict(alpha=a,mse=mse(V@(C0+a*(C-C0)),yv)) for a in [0.,.125,.25,.5,1.]];alpha=min(curve,key=lambda r:r['mse'])['alpha'];rows.append(dict(method=label,seed=seed,**details,validation_curve=curve,selected_alpha=alpha,full_test_delta_db=float(10*np.log10(baseline/mse(E@C,ye))),selected_test_delta_db=float(10*np.log10(baseline/mse(E@(C0+alpha*(C-C0)),ye)))))
  print(name,label,seed,rows[-1]['qualified'],rows[-1]['sweeps'],rows[-1]['full_test_delta_db'],flush=True);(out/'results.json').write_text(json.dumps(dict(state='running',name=name,uid=uid,rows=rows),indent=2))
 for label,s,e in [('original16k',0,16384),('fresh16k',81920,98304),('larger65k',81920,147456)]:run(label,None,A[s:e],Y[s:e])
 B=A[81920:147456];yf=Y[81920:147456];n=B.shape[0];rng=np.random.default_rng(20261015);P=rng.choice([-1.,1.],size=(B.shape[1],128))/np.sqrt(128);prob=importance_probabilities(B@P)
 for seed in [101,202,303]:
  for label,p in [('uniform',np.full(n,1/n)),('projected_leverage',prob)]:
   idx=np.random.default_rng(seed).choice(n,16384,replace=True,p=p);w=1/np.sqrt(n*p[idx]);run(label,seed,B[idx].multiply(w[:,None]).tocsr(),yf[idx]*w[:,None])
 (out/'results.json').write_text(json.dumps(dict(state='complete',name=name,uid=uid,rows=rows,scope='Exploratory solver-accuracy follow-up on same samples and four historical fields. Empirical half-MSE gap tolerance1e-7, cap4000sweeps.'),indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--object',required=True);main(p.parse_args().object)
