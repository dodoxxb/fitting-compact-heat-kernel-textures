"""Render an explicit saved field along a colour-only coefficient path.

Requires the documented HKTex GPU/renderer environment and user-supplied mesh,
teacher metadata, field export and candidate colour arrays. No hidden datasets.
"""
from pathlib import Path
import argparse,copy,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(Path(__file__).resolve().parent),str(ROOT/'research/round23'),str(ROOT/'research/local_baselines'),str(ROOT)]

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mesh',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--field',type=Path,required=True);p.add_argument('--colours',type=Path,required=True,help='NPZ with full [K+1,3] candidate, background last.');p.add_argument('--output',type=Path,required=True);p.add_argument('--resolutions',default='128,256,512');p.add_argument('--fractions',default='0,.125,.25,.5,1');a=p.parse_args()
 if a.output.exists():p.error('Use a new output directory')
 import torch,mitsuba as mi,drjit as dr
 mi.set_variant('cuda_ad_rgb');torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest');torch.cuda.set_per_process_memory_fraction(.65)
 from memory_adapter import install
 install()
 from hktex.modules import Mesh,EigenAlboInterpolationKNN
 from hktex.utils.mesh import load_mesh
 from common import load_field
 from hktex.rendering.uv_texture_renderer import UVTextureRenderer
 from hktex.utils.mitsuba import dr_no_jit
 from native_hybrid import native_bitmap,make_hybrid_shape
 import shared_evaluator as se
 ck=torch.load(a.checkpoint,map_location='cuda',weights_only=True);tm=load_mesh(str(a.mesh),**ck['loader_kwargs']);raw=load_mesh(str(a.mesh),merge_tex=False,normalise_size=True)
 assert np.max(abs(tm.vertices[tm.faces]-raw.vertices[raw.faces]))<1e-6
 mesh=Mesh.from_trimesh(tm,device='cuda');eigen=EigenAlboInterpolationKNN(ck['eigen_config'],mesh);m=load_field(a.field,ck,mesh,eigen)
 if eigen._eigen_vec_knn.numel()*eigen._eigen_vec_knn.element_size()<6*1024**3:eigen._eigen_vec_knn=eigen._eigen_vec_knn.to('cuda')
 eigen.reset();m.prepare_kernels(mesh,eigen)
 base=np.load(a.field);C0=np.vstack([base['_kernel_colours'],base['_mean_colour'].reshape(1,3)]);C1=np.load(a.colours)['full'];assert C1.shape==C0.shape and np.isfinite(C1).all()
 class Field(torch.nn.Module):
  def __init__(self):super().__init__();self.register_buffer('C',torch.as_tensor(C0,device='cuda',dtype=torch.float32))
  def requires_face_ids(self):return True
  def forward(self,pts,face_ids):
   f=face_ids.long();b=mesh.cartesian_to_barycentric(pts,mesh.get_face_vertices(f));_,_,ii,ww=m.diffuse_heat_kernels(eigen,m.prepare_points(mesh,eigen,f,barys=b));ii=ii[...,0].T;ww=ww[...,0].T;ww=ww/ww.sum(1,keepdim=True).clamp_min(1);return ((ww[...,None]*self.C[ii]).sum(1)+self.C[-1]).clamp(0,1)
 net=Field();se.install_compact_bridge();torch.set_num_threads(2);a.output.mkdir(parents=True);report=dict(state='running',rows=[],scope='Descriptive colour-path rendering, not image-selected optimization or a speed benchmark.');fractions=list(map(float,a.fractions.split(',')))
 if any(x<0 or x>1 for x in fractions):p.error('Fractions must be in [0,1]')
 sync=lambda:(torch.cuda.synchronize(),dr.sync_thread())
 for res in map(int,a.resolutions.split(',')):
  cfg=copy.deepcopy(se.CONFIG);cfg['camera_config'].update(img_width=res,img_height=res);ren=UVTextureRenderer(cfg);denoiser=mi.OptixDenoiser(input_size=(res,res),temporal=False)
  shape,refs=make_hybrid_shape(raw,native_bitmap(np.zeros((2,2,3),np.float32)),net,batching=2048);gt=UVTextureRenderer(copy.deepcopy(cfg)).mesh_to_mitsuba(raw)
  rawimages={x:[] for x in ['target']+fractions};images={x:[] for x in rawimages}
  for view in range(5):
   ren.change_camera_param(azimuth_deg=180+72*view,elevation_deg=0);scenes={'target':ren.make_scene(gt,with_params=False),'field':ren.make_scene(shape,with_params=False)}
   for alpha in images:
    if alpha!='target':net.C.copy_(torch.as_tensor(C0+alpha*(C1-C0),device='cuda',dtype=torch.float32))
    with torch.no_grad(),dr_no_jit(True):
     im=mi.render(scenes['target' if alpha=='target' else 'field'],seed=119,spp=4);dr.eval(im);sync();den=denoiser(im);dr.eval(den);sync()
    rawimages[alpha].append(np.array(im));images[alpha].append(np.array(mi.Bitmap(den).convert(mi.Bitmap.PixelFormat.RGB,mi.Struct.Type.UInt8,srgb_gamma=True)))
  y=np.stack(images['target']).astype(float)/255
  for alpha in images:
   label='target' if alpha=='target' else f'alpha_{alpha:g}';u=np.stack(images[alpha]);lin=np.stack(rawimages[alpha]);assert np.isfinite(lin).all();np.savez_compressed(a.output/f'{res}_{label}.npz',srgb_uint8=u,linear_raw=lin)
   if alpha!='target':
    e=((u/255-y)**2).mean(axis=(1,2,3));report['rows'].append(dict(resolution=res,alpha=alpha,mean_view_psnr=float(np.mean(-10*np.log10(np.maximum(e,1e-30))))))
  (a.output/'results.json').write_text(json.dumps(report,indent=2))
 report['state']='complete';(a.output/'results.json').write_text(json.dumps(report,indent=2));print(a.output)
if __name__=='__main__':main()
