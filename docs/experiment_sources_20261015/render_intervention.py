"""Actual colour interpolation and resolution intervention, no image blending."""
from pathlib import Path
import argparse,copy,json,os,sys,time,hashlib
ROOT=Path(__file__).resolve().parents[2]; HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'research/round51'))
from gpu_probe import context,load_field,set_state,FieldNet,matrices
from anchor_cpu import CASES
import numpy as np
import torch

def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2,allow_nan=False));q.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(x,y):
 e=(x.astype(float)-y.astype(float))**2
 mv=e.mean(axis=(1,2,3));pooled=float(e.mean())
 return dict(mse=pooled,pooled_psnr=float(-10*np.log10(max(pooled,1e-30))),mean_view_psnr=float(np.mean(-10*np.log10(np.maximum(mv,1e-30)))))
def main(name):
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest');torch.cuda.set_per_process_memory_fraction(.65)
 from memory_adapter import install
 install()
 uid=next(u for n,u in CASES if n==name);out=HERE/'renders'/uid;out.mkdir(parents=True,exist_ok=False)
 start=time.perf_counter();source=ROOT/'research/round47/diagnostics'/uid/'0500';old=ROOT/'research/round51/gpu'/uid
 build=json.loads((source/'build.json').read_text());path=Path(build['inputs']['before']['path'])
 arr=dict(np.load(source/'data.npz'));base=dict(np.load(path));C0=arr['C0'];C1=np.load(old/'normalized_colours.npz')['full']
 alphas=[0.,.125,.25,.5,1.]; report=dict(state='running',name=name,uid=uid,methods={},protocol_sha256=sha(HERE/'PROTOCOL.md'),script_sha256=sha(__file__),source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (path,old/'normalized_colours.npz',source/'data.npz')},alpha_grid=alphas,resolutions=[128,256,512],scope='Four reused development objects; all image alpha comparisons descriptive, no inference-speed claim.')
 # Selection uses only pre-existing native surface validation, not any images.
 from scipy.sparse import load_npz
 A=load_npz(old/'normalized.npz');yv=arr['target'][147456:155648]
 curve=[dict(alpha=a,mse=float(np.mean((np.clip(A[16384:24576]@(C0+a*(C1-C0)),0,1)-yv)**2))) for a in alphas]
 report['surface_validation_curve']=curve;report['surface_selected_alpha']=min(curve,key=lambda v:v['mse'])['alpha'];save(out/'results.json',report)
 uid,teacher,ck,tm,raw,mesh,eigen=context(name);m=load_field(path,ck,mesh,eigen)
 basis=eigen._eigen_vec_knn;report['basis_bytes']=basis.numel()*basis.element_size()
 if report['basis_bytes']<6*1024**3:eigen._eigen_vec_knn=basis.to('cuda')
 set_state(m,mesh,eigen,base,C=C0)
 f=torch.as_tensor(arr['faces'][155648:159744],device='cuda');b=torch.as_tensor(arr['bary'][155648:159744],device='cuda')
 pts=mesh.barycentric_to_cartesian(b,mesh.get_face_vertices(f));br=mesh.cartesian_to_barycentric(pts,mesh.get_face_vertices(f))
 with torch.no_grad():AA,_,_,_,_=matrices(m,mesh,eigen,f,br)
 net=FieldNet(m,mesh,eigen,C0,False)
 with torch.no_grad():
  small=np.concatenate([net(pts[i:i+256],f[i:i+256]).cpu().numpy() for i in range(0,len(f),256)])
  large=np.concatenate([net(pts[i:i+2048],f[i:i+2048]).cpu().numpy() for i in range(0,len(f),2048)])
 report['gates']=dict(same_query_max_abs=float(np.max(abs(large-np.clip(AA@C0,0,1)))),batch_max_abs=float(np.max(abs(small-large))))
 assert max(report['gates'].values())<2e-5,report['gates'];save(out/'results.json',report)
 import shared_evaluator as se
 import mitsuba as mi,drjit as dr
 from hktex.rendering.uv_texture_renderer import UVTextureRenderer
 from hktex.utils.mitsuba import dr_no_jit
 from native_hybrid import native_bitmap,make_hybrid_shape
 se.install_compact_bridge();torch.set_num_threads(2)
 sync=lambda:(torch.cuda.synchronize(),dr.sync_thread())
 for res in report['resolutions']:
  cfg=copy.deepcopy(se.CONFIG);cfg['camera_config']['img_width']=res;cfg['camera_config']['img_height']=res;cfg['point_batching']=2048
  ren=UVTextureRenderer(cfg);denoiser=mi.OptixDenoiser(input_size=(res,res),temporal=False)
  shape,refs=make_hybrid_shape(raw,native_bitmap(np.zeros((2,2,3),np.float32)),net,batching=2048)
  gt=UVTextureRenderer(copy.deepcopy(cfg)).mesh_to_mitsuba(raw)
  images={a:[] for a in ['target']+alphas};linear={a:[] for a in images};seconds=0.
  for view in range(5):
   ren.change_camera_param(azimuth_deg=180+view*72,elevation_deg=0)
   scenes={'target':ren.make_scene(gt,with_params=False),'field':ren.make_scene(shape,with_params=False)}
   for a in ['target']+alphas:
    if a!='target':net.C.copy_(torch.as_tensor(C0+a*(C1-C0),device='cuda',dtype=torch.float32))
    sync();t=time.perf_counter()
    with torch.no_grad(),dr_no_jit(True):
     im=mi.render(scenes['target' if a=='target' else 'field'],seed=119,spp=4);dr.eval(im);sync();den=denoiser(im);dr.eval(den);sync()
    seconds+=time.perf_counter()-t
    x=np.array(im);u=np.array(mi.Bitmap(den).convert(mi.Bitmap.PixelFormat.RGB,mi.Struct.Type.UInt8,srgb_gamma=True));assert np.isfinite(x).all()
    linear[a].append(x);images[a].append(u)
   print(name,res,'view',view,flush=True)
  for a in images:
   label='target' if a=='target' else f'alpha_{a:g}'
   im=np.stack(images[a]);lin=np.stack(linear[a]);np.savez_compressed(out/f'{res}_{label}.npz',srgb_uint8=im,linear_raw=lin)
   if a!='target':report['methods'][f'{res}_{label}']=dict(resolution=res,alpha=a,srgb=stats(im/255,np.stack(images['target'])/255),linear=stats(lin,np.stack(linear['target'])))
  report.setdefault('resolution_render_denoise_seconds',{})[str(res)]=seconds;save(out/'results.json',report)
 report.update(state='complete',inclusive_seconds=time.perf_counter()-start,torch_peak_allocated_bytes=torch.cuda.max_memory_allocated());save(out/'results.json',report)
 print(name,'COMPLETE',report['inclusive_seconds'],flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--object',required=True);main(p.parse_args().object)
