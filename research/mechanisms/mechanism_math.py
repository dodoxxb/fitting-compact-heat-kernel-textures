"""Small, explicit diagnostics. Classical identities, not a new optimizer."""
import numpy as np

def image_change(reference, initial, updated):
 y,i,j=(np.asarray(a,dtype=np.float64) for a in (reference,initial,updated))
 if y.shape!=i.shape or i.shape!=j.shape:raise ValueError('Shapes must match')
 e=i-y;d=j-i;l0=float(np.mean(e*e));energy=float(np.mean(d*d));cross=float(2*np.mean(e*d));l1=float(np.mean((j-y)**2))
 return dict(initial_mse=l0,updated_mse=l1,cross_term=cross,change_energy=energy,identity_residual=l1-l0-cross-energy,cosine=None if not l0*energy else -cross/(2*np.sqrt(l0*energy)),relative_change=None if not l0 else np.sqrt(energy/l0),image_line_oracle=0. if not energy else float(np.clip(-cross/(2*energy),0,1)))

def deletion_change(kept_numerator,tail_numerator,kept_mass,tail_mass):
 """Preclip exact difference; supports all denominator-floor regimes."""
 nk,nt=np.asarray(kept_numerator),np.asarray(tail_numerator);sk,st=np.asarray(kept_mass),np.asarray(tail_mass)
 a=np.maximum(1.,sk+st);b=np.maximum(1.,sk)
 return nk*(1/b-1/a)[...,None]-nt/a[...,None]

def interpolation_variance(factors,weights):
 f=np.asarray(factors,dtype=float);w=np.asarray(weights,dtype=float)
 if np.any(w<0) or not np.isclose(w.sum(),1):raise ValueError('Need convex interpolation weights')
 mean=np.einsum('i,ive->ve',w,f);factor=mean@mean.T;response=np.einsum('i,ive,iwe->vw',w,f,f);variance=np.einsum('i,ive,iwe->vw',w,f-mean,f-mean)
 return factor,response,variance

def importance_probabilities(projected_features,mixture=.5,ridge_relative=1e-6):
 """Target-preserving finite-pool probabilities; projection loses directions."""
 z=np.asarray(projected_features,dtype=float);n,d=z.shape
 if not (0<=mixture<1):raise ValueError('Uniform component must be positive')
 g=z.T@z/n;lam,q=np.linalg.eigh(g);ridge=max(ridge_relative*np.trace(g)/d,np.finfo(float).tiny)
 white=z@(q/np.sqrt(np.maximum(lam,0)+ridge));ell=np.sum(white*white,axis=1)
 return np.full(n,1/n) if ell.sum()==0 else (1-mixture)/n+mixture*ell/ell.sum()
