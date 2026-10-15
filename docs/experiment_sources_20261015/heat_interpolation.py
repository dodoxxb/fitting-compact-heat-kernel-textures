"""Matched-node interpolation diagnostic; no claim of texture-training gains."""
from pathlib import Path
import sys,json,time,itertools
import numpy as np
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'research/env_digeo07'),str(ROOT/'research/env_vendor311'),str(ROOT)]
import torch,mitsuba as mi,yaml
mi.set_variant('llvm_ad_rgb');setter=mi.set_variant
mi.set_variant=lambda name,*a,**k:setter('llvm_ad_rgb' if name=='cuda_ad_rgb' else name,*a,**k)
from hktex.modules import Mesh
from hktex.utils.mesh import load_mesh
import hktex.modules.eigen_albo as module
from hktex.knn_heat.knn_heat._ops.grid_select import _select_grid_cell4_bilinear

def relative(a,b):return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30))
def main():
 torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0);start=time.perf_counter();rows=[]
 plan=json.loads((ROOT/'research/round47/confirmation/PREPARATION_PLAN_DRAFT.json').read_text())
 for uid in ['6e509bb275c44dc994d1e6bc694242f5','242d37a17a334d3cb1ab4fb978ae0ad9']:
  src=ROOT/'research/round48/heat_feature_cache_build'/uid;z=np.load(src/'operator_cache_candidate.npz');F=z['heat_features_fp64'][:,z['probe_vertices']];coords=torch.tensor(z['sampling_coords'],dtype=torch.float64);t=float(z['diffusion_time']);rank=F.shape[-1]
  row=next(r for r in plan['objects'] if r['uid']==uid);tm=load_mesh(row['mesh_path'],**plan['loader_kwargs']);mesh=Mesh.from_trimesh(tm,device='cpu');payload=torch.load(src/'heat_features.pt',map_location='cpu',weights_only=True);direction=payload['local_direction'];probes=z['probe_vertices'];records=[]
  def weights(angle,scale):
   ids,w=_select_grid_cell4_bilinear(torch.unique(coords[:,0],sorted=True),torch.unique(coords[:,1],sorted=True),torch.tensor([angle],dtype=torch.float64),torch.tensor([scale],dtype=torch.float64),scale_map='log1p',grid_layout='rot_major',eps=1e-8,clamp_unit=True)
   return ids[0].numpy(),w[0].numpy()
  def kernels(angle,scale):
   ids,w=weights(angle,scale);f=F[ids];bar=np.einsum('i,ive->ve',w,f);mix=np.einsum('i,ive,iwe->vw',w,f,f);factor=bar@bar.T
   signs=np.where(np.arange(len(F))%2==0,1.,-1.)[ids];fb=np.einsum('i,i,ive->ve',w,signs,f)
   return factor,mix,fb@fb.T,ids,w
  for deg,scale in itertools.product([15.,45.,75.],[3.,10.,45.]):
   angle=np.deg2rad(deg);kf,km,kg,ids,w=kernels(angle,scale);f=F[ids];bar=np.einsum('i,ive->ve',w,f);cov=np.einsum('i,ive,iwe->vw',w,f-bar,f-bar)
   err=relative(km-kf,cov);assert err<1e-10
   vals=np.linalg.eigvalsh(km-kf);assert vals.min()>-1e-10*max(1.,np.linalg.norm(km))
   L,M=module.get_anisotropic_lbo(mesh.verts,mesh.faces.T,mesh.fnorms,rotation_angle=angle,anisotropy=scale,local_direction=direction)
   lam,raw=module.compute_eig_laplacian(L,M,rank);truth=raw[probes].astype(float)*np.exp(-t*lam.astype(float)/2);truth=truth@truth.T
   derivatives={}
   for var,eps in [('angle_radians',1e-4),('log1p_anisotropy',1e-4)]:
    plus=kernels(angle+eps,scale) if var=='angle_radians' else kernels(angle,np.expm1(np.log1p(scale)+eps))
    minus=kernels(angle-eps,scale) if var=='angle_radians' else kernels(angle,np.expm1(np.log1p(scale)-eps))
    df=(plus[0]-minus[0])/(2*eps);dm=(plus[1]-minus[1])/(2*eps)
    derivatives[var]=dict(factor_norm=float(np.linalg.norm(df)),response_norm=float(np.linalg.norm(dm)),relative_difference=relative(df,dm))
   rec=dict(angle_degrees=deg,anisotropy=scale,factor_relative_error=relative(kf,truth),response_relative_error=relative(km,truth),factor_response_relative_difference=relative(kf,km),factor_sign_change_relative_difference=relative(kg,kf),response_sign_change_relative_difference=0.,variance_identity_relative_error=err,variance_min_eigenvalue=float(vals.min()),factor_to_response_trace=float(np.trace(kf)/np.trace(km)),derivatives=derivatives,grid_indices=ids.tolist(),weights=w.tolist());records.append(rec);print(row['name'],deg,scale,rec['factor_relative_error'],rec['response_relative_error'],flush=True)
  rows.append(dict(name=row['name'],uid=uid,records=records,grid_nodes=len(F),probe_vertices=len(probes),rank=rank,nodal_sign_invariance_max_abs=0.))
  (HERE/'heat_results.json').write_text(json.dumps(dict(state='running',objects=rows),indent=2))
 assert not torch.cuda.is_initialized()
 (HERE/'heat_results.json').write_text(json.dumps(dict(state='complete',objects=rows,seconds=time.perf_counter()-start,cuda_initialized=False,scope='Two reused operator-development meshes, nine points each; same nodal heat factors. Direct off-grid eigensolutions use the same spectral rank, not an infinite-rank ground truth. No texture accuracy or speed claim.'),indent=2,allow_nan=False))
if __name__=='__main__':main()
