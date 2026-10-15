"""Independent numerical checks of the stated conditional identities.

Projection and residualized-column identities use unconstrained least squares;
they are not asserted for an active colour bound or a clipped objective. The
box-constrained quadratic is checked separately through its convex gap bound.
"""
from pathlib import Path
import json
import numpy as np
from scipy.linalg import expm
from scipy.optimize import lsq_linear

def main():
    rng=np.random.default_rng(24091);checks={}
    A=rng.normal(size=(30,7));Y=rng.normal(size=(30,3));C=rng.uniform(-1,1,(7,3));r=A@C-Y
    g=A.T@r/Y.size;v=np.where(g>=0,-1.,1.);fw=float(np.sum(g*(C-v)));f=float(.5*np.mean(r*r))
    optimal=np.column_stack([lsq_linear(A,Y[:,c],bounds=(-1,1),tol=1e-12).x for c in range(3)])
    fstar=float(.5*np.mean((A@optimal-Y)**2));assert -1e-12<=f-fstar<=fw+1e-12
    checks['convex_box_gap']=dict(primal=f,reference_optimum=fstar,certified_lower=f-fw)
    quantized=C.astype(np.float16).astype(float);fq=float(.5*np.mean((A@quantized-Y)**2))
    assert fq-fstar<=fq-max(0,f-fw)+1e-12
    checks['reuse_lower_bound_after_quantization']=True
    a=rng.normal(size=(30,1));E=Y-A@np.linalg.lstsq(A,Y,rcond=None)[0]
    q=a-A@np.linalg.lstsq(A,a,rcond=None)[0];joint=np.column_stack([A,a])
    improved=Y-joint@np.linalg.lstsq(joint,Y,rcond=None)[0]
    assert abs(np.sum(E*E)-np.sum(improved*improved)-np.sum((q.T@E)**2)/np.sum(q*q))<1e-10
    checks['residualized_column_gain']=True
    e=rng.normal(size=(80,3));rr=rng.normal(size=(80,3));gain=2*np.sum(e*rr)-np.sum(rr*rr)
    actual=np.sum(e*e)-np.sum((e-rr)**2);assert abs(gain-actual)<1e-10
    alpha=np.clip(np.sum(e*rr)/np.sum(rr*rr),0,1);grid=np.linspace(0,1,1001)
    assert np.sum((e-alpha*rr)**2)<=min(np.sum((e-a*rr)**2) for a in grid)+1e-10
    a=rng.normal(size=(80,1));c=a.T@e/float((a*a).sum());actual=np.sum(e*e)-np.sum((e-a@c)**2)
    predicted=float(np.sum((a.T@e)**2)/(a*a).sum());assert abs(actual-predicted)<1e-10
    checks['residual_identity_and_scalar_gate']=True;checks['single_column_gain']=True
    B=rng.random((80,3));Y=rng.random((80,3));R=rng.normal(size=(80,3));E=Y-B
    raw_gain=2*np.sum(E*R)-np.sum(R*R)
    clipped_gain=np.sum(E*E)-np.sum((np.clip(B+R,0,1)-Y)**2)
    assert clipped_gain>=raw_gain-1e-10
    checks['clipped_residual_gain_bound']=True
    A=np.array([[.5],[1.]]);mean=.5;Y=np.ones((2,1));C=np.linalg.lstsq(A,Y-mean,rcond=None)[0]
    projection=np.sum((mean+A@C-Y)**2);clipped=np.sum((np.clip(mean+A@np.ones((1,1)),0,1)-Y)**2)
    assert abs(projection-1/20)<1e-12 and clipped==0 and np.all(abs(C)<=1)
    checks['clipping_counterexample']=dict(unclipped_projection_residual=projection,clipped_error=clipped,base_colour=mean,response_rows=A.ravel().tolist(),feasible_clipped_colour=1.)
    for _ in range(500):
        w=rng.random(50);wt=w+rng.uniform(-1e-5,1e-5,50);k=30;eps=np.max(abs(w-wt));ix=np.argsort(-w)[:k];iy=np.argsort(-wt)[:k];ww=np.sort(w)[::-1]
        if ww[k-1]-ww[k]>2*eps:assert set(ix)==set(iy)
        col=rng.uniform(-1,1,(50,3));u=np.zeros(50);v=np.zeros(50);u[ix]=w[ix];v[iy]=wt[iy]
        fu=u@col/max(1,u.sum());fv=v@col/max(1,v.sum());assert np.max(abs(fu-fv))<=2*np.sum(abs(u-v))+1e-12
    eps=1e-10;w=np.array([.5+eps,.5]);wt=np.array([.5,.5+eps]);col=np.array([1.,-1.])
    field=lambda z:np.clip(.5+z[np.argmax(z)]*col[np.argmax(z)],0,1)
    assert abs(field(w)-field(wt))>.99
    checks['support_margin_and_readout_bound']=True;checks['topk_discontinuity']=dict(weight_perturbation=eps,rgb_jump=float(abs(field(w)-field(wt))))
    L=np.diag([0.,1.,4.]);R=np.linalg.qr(rng.normal(size=(3,3)))[0];t=.3
    true=expm(-t*L);wrong=R@expm(-t*L)@R.T;right=R@expm(-t*(R.T@L@R))@R.T
    assert np.linalg.norm(true-right)<1e-12 and np.linalg.norm(true-wrong)>.01
    checks['basis_rotation']=dict(corrected_error=float(np.linalg.norm(true-right)),unrotated_generator_error=float(np.linalg.norm(true-wrong)))
    B=R@L@R.T;H=lambda t:.5*expm(-t*L)+.5*expm(-t*B)
    defect=np.linalg.norm(H(.3+.2)-H(.3)@H(.2));assert defect>1e-4
    bound=t*np.linalg.norm(L-B,2);err=np.linalg.norm(expm(-t*L)-expm(-t*B),2);assert err<=bound+1e-12
    checks['mixture_not_semigroup']=float(defect);checks['duhamel_contraction_bound']=dict(actual=err,bound=bound)
    out=Path(__file__).with_name('theory_contract_checks.json');out.write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
