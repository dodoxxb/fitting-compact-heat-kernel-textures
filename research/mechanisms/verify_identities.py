"""Independent numerical checks including floor-crossing and degenerate cases."""
from mechanism_math import *
import json

def verify():
 rng=np.random.default_rng(20261015);checks={}
 for _ in range(100):
  y=rng.random((40,3));i=rng.random((40,3));j=rng.random((40,3));r=image_change(y,i,j);assert abs(r['identity_residual'])<1e-14
  assert (r['updated_mse']<r['initial_mse'])==(r['relative_change']<2*r['cosine'])
 checks['image_identity']=100
 regimes=set()
 for scale in [.01,.1,1.,10.]:
  w=rng.random((100,8))*scale;c=rng.uniform(-1,1,(8,3));sk=w[:,:3].sum(1);st=w[:,3:].sum(1);nk=w[:,:3]@c[:3];nt=w[:,3:]@c[3:]
  direct=nk/np.maximum(1,sk)[:,None]-(nk+nt)/np.maximum(1,sk+st)[:,None]
  assert np.max(abs(direct-deletion_change(nk,nt,sk,st)))<1e-14
  for a,b in zip(sk,sk+st):regimes.add('unnormalized' if b<=1 else 'crossing' if a<1 else 'normalized')
 assert regimes=={'unnormalized','crossing','normalized'};checks['deletion_floor_regimes']=sorted(regimes)
 for _ in range(50):
  f=rng.normal(size=(4,11,5));w=rng.dirichlet(np.ones(4));kf,kr,v=interpolation_variance(f,w);assert np.max(abs(kr-kf-v))<1e-13;assert np.linalg.eigvalsh(v).min()>-1e-12
 checks['variance_identity_and_psd']=50
 f=np.stack([np.eye(2),np.eye(2)]);k1,m1,_=interpolation_variance(f,[.5,.5]);f[1]*=-1;k2,m2,_=interpolation_variance(f,[.5,.5]);assert np.array_equal(k1,np.eye(2)) and np.array_equal(k2,np.zeros((2,2))) and np.array_equal(m1,m2);checks['same_nodes_different_interpolation']=True
 # Enumerate expectation rather than relying on Monte Carlo agreement.
 z=rng.normal(size=(40,5));p=importance_probabilities(z);loss=rng.random(40);assert abs(np.sum(p*loss/(40*p))-loss.mean())<1e-14;assert np.max(1/(40*p))<=2+1e-12;checks['weighted_risk_expectation']=True
 for _ in range(1000):
  pop=rng.random(10);emp=rng.random(10);choice=int(rng.integers(10));gamma=emp[choice]-emp.min();assert pop[choice]-pop.min()<=gamma+2*np.max(abs(pop-emp))+1e-14
 checks['finite_family_risk_bound']=1000
 pred=np.array([.75,.25]);assert np.mean((pred-[1,0])**2)==1/16;checks['bounded_non_nesting_example']=True
 return dict(state='passed',checks=checks,scope='Numerical checks of classical identities and counterexamples, not empirical generalization guarantees.')
if __name__=='__main__':print(json.dumps(verify(),indent=2))
