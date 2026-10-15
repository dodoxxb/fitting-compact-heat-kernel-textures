"""Portable fixed-time heat features; NumPy/SciPy only.

Input eigenpairs must be raw, before unrestricted eigenbasis alignment.
Multiplying an already mixed legacy basis by diagonal weights cannot recover
the raw heat kernel. This construction is not a weight-preserving conversion
of an existing trained texture; the reported experiments refit both variants.
"""
import numpy as np
from scipy.linalg import svd
from scipy.optimize import linear_sum_assignment


def diffuse(eigenvectors,eigenvalues,time):
    phi=np.asarray(eigenvectors,dtype=np.float64);lam=np.asarray(eigenvalues,dtype=np.float64)
    if phi.ndim!=2 or lam.shape!=(phi.shape[1],):raise ValueError('Incompatible eigenpair shapes')
    if not np.isfinite(phi).all() or not np.isfinite(lam).all() or (lam<0).any():raise ValueError('Expected finite basis and nonnegative eigenvalues')
    if not np.isfinite(time) or time<0:raise ValueError('Expected finite nonnegative diffusion time')
    return phi*np.exp(-time*lam/2)


def align(reference,features,mass,tolerance=1e-10):
    ref=np.asarray(reference,dtype=np.float64);z=np.asarray(features,dtype=np.float64);m=np.asarray(mass,dtype=np.float64)
    if ref.shape!=z.shape or z.ndim!=2 or m.shape!=(z.shape[0],):raise ValueError('Incompatible alignment shapes')
    if not all(np.isfinite(x).all() for x in (ref,z,m)) or (m<0).any() or not np.any(m>0):raise ValueError('Invalid features or mass')
    _,permutation=linear_sum_assignment(-np.abs((ref.T*m)@z));permuted=z[:,permutation]
    correlation=(permuted.T*m)@ref;identity=np.eye(z.shape[1]);attempts=[]
    for driver in ('gesdd','gesvd'):
        try:
            u,s,vt=svd(correlation,lapack_driver=driver);q=u@vt
            residual=float(np.linalg.norm((u*s)@vt-correlation)/max(np.linalg.norm(correlation),np.finfo(float).tiny))
            orthogonal_error=float(np.linalg.norm(q.T@q-identity));attempts.append(dict(driver=driver,residual=residual,orthogonal_error=orthogonal_error))
            if residual<=tolerance and orthogonal_error<=tolerance and np.isfinite(q).all():break
        except np.linalg.LinAlgError as exc:attempts.append(dict(driver=driver,error=str(exc)))
    else:raise ArithmeticError('No numerically qualified orthogonal alignment: '+str(attempts))
    result=permuted@q;sign=np.sign(np.sum(ref*(m[:,None]*result),axis=0));sign[sign==0]=1
    result*=sign
    return result,dict(permutation=permutation.tolist(),checks=attempts)
