"""Build an opt-in fixed-time cache from raw mesh eigenpairs, on CPU.

Run in an HKTex environment. Keep the resulting cache separate from old
trained fields; it changes the spectral interpolant and those fields need
refitting. No CUDA context or source-texture evaluation is required here.
"""
from pathlib import Path
import argparse,hashlib,json,sys,time
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hktex-root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--mesh',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--variant',choices=('heat_features','fp64_diagonal'),default='heat_features')
    a=p.parse_args();assert not a.output.exists(),'Use a separate new cache; overwriting is not supported'
    root=a.hktex_root.resolve();sys.path[:0]=[str(root/'research/round49'),str(root)]
    import torch,yaml,mitsuba as mi
    from types import SimpleNamespace
    from fixed_time_features import diffuse,align
    # HKTex imports a CUDA renderer globally; this CPU precomputation never
    # calls the renderer. Redirect only that unused import-time variant.
    mi.set_variant('llvm_ad_rgb');setter=mi.set_variant
    mi.set_variant=lambda name,*args,**kw:setter('llvm_ad_rgb' if name=='cuda_ad_rgb' else name,*args,**kw)
    from hktex.modules import Mesh
    from hktex.utils.mesh import load_mesh
    import hktex.modules.eigen_albo as module
    torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0);assert not torch.cuda.is_initialized()
    start=time.perf_counter();trainer=yaml.safe_load(a.config.read_text())['trainer'];cfg=dict(trainer['eigen_albo']);t=float(trainer['model']['diff_time'])
    assert not cfg.get('normalise_evals',False),'This recipe assumes unscaled eigenvalues'
    mesh=Mesh.from_trimesh(load_mesh(str(a.mesh),merge_tex=True,normalise_size=True),device='cpu')
    k=min(int(cfg['k_eig']),mesh.N_verts-1);assert k>int(cfg.get('knn_embedding_dim',64))
    cfg['k_eig']=k;obj=module.EigenAlboInterpolation.__new__(module.EigenAlboInterpolation)
    obj.cfg=SimpleNamespace(**cfg);obj._mesh=mesh;obj.device='cpu';iso,basis=obj._precompute_iso_eigen();direction=obj._compute_local_directions()
    ref=diffuse(basis,iso[:k].numpy(),t) if a.variant=='heat_features' else basis.astype(float)
    flat=[];coordinates=[];checks=[]
    for degrees in range(0,181,cfg['precompute_angles_every_deg']):
        angle=np.deg2rad(degrees)
        for index,scale in enumerate(cfg['precompute_anisotropies']):
            lapl,mass=module.get_anisotropic_lbo(mesh.verts,mesh.faces.T,mesh.fnorms,rotation_angle=angle,anisotropy=float(scale),local_direction=direction)
            values,vectors=module.compute_eig_laplacian(lapl,mass,k)
            features=diffuse(vectors,values,t) if a.variant=='heat_features' else vectors
            aligned,audit=align(ref,features,mass)
            weights=np.zeros(k) if a.variant=='heat_features' else values[np.asarray(audit['permutation'])]
            flat.append(np.concatenate([weights,aligned.reshape(-1)]).astype(np.float32));coordinates.append([angle,scale]);checks.append(audit)
            ref=aligned
            if index==0:row_ref=aligned
        ref=row_ref
    payload=dict(iso_eigen=iso.float(),all_eigen=torch.from_numpy(np.asarray(flat)),sampling_coords=torch.tensor(coordinates,dtype=torch.float32),
        mass=torch.tensor(mass).unsqueeze(0).float(),local_direction=direction.float())
    assert all(torch.isfinite(v).all() for v in payload.values())
    a.output.parent.mkdir(parents=True,exist_ok=True);torch.save(payload,a.output)
    meta=dict(state='complete',variant=a.variant,diffusion_time=t,k_eig=k,mesh_sha256=hashlib.sha256(a.mesh.read_bytes()).hexdigest(),
        cache_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),seconds=time.perf_counter()-start,alignment_checks=checks,
        contract='Raw eigenpairs rebuilt. Fixed diffusion time; isotropic KNN state unchanged. Requires refitting, not a guaranteed conversion of existing trained fields.')
    a.output.with_suffix('.json').write_text(json.dumps(meta,indent=2));print(json.dumps({k:v for k,v in meta.items() if k!='alignment_checks'},indent=2))


if __name__=='__main__':main()
