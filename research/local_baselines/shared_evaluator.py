"""Shared original-config five-view renderer. No training or implicit downloads.
API: evaluate_shapes(asset, shapes{name: mi.Mesh}, output, gt_name='UV_native').
Persist sRGB uint8 metric images and linear raw renders. Metrics may be computed later.
"""
import time,json,statistics,copy,os
from pathlib import Path
import numpy as np,torch,yaml,trimesh
import mitsuba as mi,drjit as dr
mi.set_variant('cuda_ad_rgb')
from hktex.rendering.uv_texture_renderer import UVTextureRenderer
from hktex.rendering.torch_texture_renderer import TorchTextureNetwork
from hktex.utils.mitsuba import dr_no_jit
ROOT=Path(__file__).resolve().parents[2]
CONFIG=yaml.safe_load((ROOT/'configs/render.yaml').read_text())['renderer']
CONFIG['point_batching']=int(os.environ.get('POINT_BATCHING','512'))
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
_sync=lambda:(torch.cuda.synchronize(),dr.sync_thread())

def install_compact_bridge():
 from hktex.rendering.util import TorchTexture
 if getattr(TorchTexture,'_local_compact_installed',False):return
 original=TorchTexture.eval
 def compact(self,si,active=True,dirs=None,norms=None,albedo=None):
  idx=dr.compress(active & si.is_valid());n=dr.width(si.p)
  if dr.width(idx)==n:return original(self,si,active,dirs,norms,albedo)
  result=dr.zeros(mi.Color3f,n)
  if dr.width(idx):
   val=original(self,dr.gather(mi.SurfaceInteraction3f,si,idx),True,dirs,norms,albedo)
   for k in range(3):dr.scatter(result[k],val[k],idx)
  return result
 TorchTexture.eval=compact;TorchTexture._local_compact_installed=True

def make_network_shape(network,raw_mesh,batching=512):
 """network(pts,face_ids) returns RGB; use GT's original geometry/normals."""
 install_compact_bridge();wrapper=TorchTextureNetwork(network,point_batching=batching);texture=mi.load_dict({'type':'torch_texture'});texture.network=wrapper
 props=mi.Properties();props['mesh_bsdf']=mi.load_dict({'type':'principled','base_color':texture})
 shape=mi.Mesh('network',len(raw_mesh.vertices),len(raw_mesh.faces),has_vertex_normals=True,props=props);p=mi.traverse(shape)
 p['vertex_positions']=np.asarray(raw_mesh.vertices,np.float32).ravel();p['faces']=np.asarray(raw_mesh.faces,np.uint32).ravel();p['vertex_normals']=np.asarray(raw_mesh.vertex_normals,np.float32).ravel();p.update();return shape,wrapper

def make_neural_shape(model,mesh,raw_mesh):
 class Net(torch.nn.Module):
  def requires_face_ids(self):return True
  def forward(self,pts,face_ids):
   f=face_ids.long();b=mesh.cartesian_to_barycentric(pts,mesh.get_face_vertices(f));return self.fwd(pts,f,b)
 net=Net();net.fwd=torch.compile(lambda pts,f,b:model(pts,face_ids=f,barys=b),dynamic=True)
 return make_network_shape(net,raw_mesh,CONFIG['point_batching'])

def evaluate_shapes(asset,shapes,output,gt_name='UV_native',warm=5):
 out=Path(output);out.mkdir(parents=True,exist_ok=True);ren=UVTextureRenderer(copy.deepcopy(CONFIG));names=list(shapes);assert gt_name in names
 _sync();t=time.perf_counter();denoiser=mi.OptixDenoiser(input_size=(256,256),temporal=False);_sync();denoiser_build_ms=(time.perf_counter()-t)*1000
 report={'asset':asset,'protocol':'Original configs/render.yaml5views256²spp4/path default max depth/ground/light/cameras; seed119; independent per-frame OptiX denoiser; sRGB UInt8 metrics. Same compact bridge and resolved point_batching for all Torch methods (512 original or explicit matched8192 extension).','renderer_config':CONFIG,'point_batching':CONFIG['point_batching'],'compilation':'torch.compile dynamic attempted; teacher/fast spectral shader compiled; INF static Direct/Baked wrappers; MLP may graph-break on configuration; TCNN native CUDA called through partialcompiled Python wrapper with logged fallback, not universally fullycompiled','tf32':False,'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'denoiser_build_ms':denoiser_build_ms,'methods':names,'views':[],'metrics_state':'images_saved_for_separate_metrics','notes':['Same-process first invocation, not independent process cold.','Mesh/material build, PNG/NPZ decoding and model load are separate setup costs.','Timed synchronized render plus denoise, with denoise reported separately. No image conversion/save/metrics in timing.','OptiX temporal=False per-frame; no previous-denoised camera image input.']}
 rng=np.random.default_rng(60171);images={n:[] for n in names};linear={n:[] for n in names}
 for view in range(5):
  az=180+view*72;ren.change_camera_param(azimuth_deg=az,elevation_deg=0);scenes={};build={}
  for n in names:
   _sync();t=time.perf_counter();scenes[n]=ren.make_scene(shapes[n],with_params=False);_sync();build[n]=(time.perf_counter()-t)*1000
  rows=[];orders=[];last={};frame_gates=[]
  with torch.no_grad(),dr_no_jit(True):
   for repeat in range(1+warm):
    order=list(rng.permutation(names));orders.append(order)
    for n in order:
     _sync();t=time.perf_counter();im=mi.render(scenes[n],seed=119,spp=4);dr.eval(im);_sync();raw_ms=(time.perf_counter()-t)*1000
     td=time.perf_counter();den=denoiser(im);dr.eval(den);_sync();den_ms=(time.perf_counter()-td)*1000
     last[n]=(im,den);rows.append({'method':str(n),'repeat':repeat,'render_ms':raw_ms,'denoise_ms':den_ms,'total_ms':raw_ms+den_ms})
    # Conversion and validation are outside all timed method intervals.
    for teacher,fast in [('HKTex','Fast_HKTex'),('INF','Fast_INF')]:
     if teacher in last and fast in last:
      delta=np.array(last[teacher][0])-np.array(last[fast][0]);mx=float(np.abs(delta).max())
      frame_gates.append({'repeat':repeat,'teacher':teacher,'fast':fast,'linear_max_abs':mx,'linear_rmse':float(np.sqrt(np.mean(delta.astype(np.float64)**2))),'finite':bool(np.isfinite(delta).all()),'pass_1e_5':bool(np.isfinite(delta).all() and mx<=1e-5)})
  for n,(im,den) in last.items():
   a=np.array(im);assert np.isfinite(a).all();linear[n].append(a);bmp=mi.Bitmap(den).convert(mi.Bitmap.PixelFormat.RGB,mi.Struct.Type.UInt8,srgb_gamma=True);u=np.array(bmp);images[n].append(u);bmp.write(str(out/f'{n}_view{view}.png'))
  r={'view':view,'azimuth':az,'elevation':0,'scene_build_ms':build,'frame_gates':frame_gates,'orders':orders,'rows':rows,'warm_median_ms':{n:statistics.median(x['total_ms'] for x in rows if x['method']==n and x['repeat']>0) for n in names},'warm_render_median_ms':{n:statistics.median(x['render_ms'] for x in rows if x['method']==n and x['repeat']>0) for n in names}};report['views'].append(r);(out/'render_results.json').write_text(json.dumps(report,indent=2));print(asset,'view',view,r['warm_median_ms'],flush=True)
 for n in names:np.savez_compressed(out/f'{n}_images.npz',srgb_uint8=np.stack(images[n]),linear_raw=np.stack(linear[n]))
 report['torch_peak_allocated_all_models_bytes']=torch.cuda.max_memory_allocated();report['memory_scope']='All methods resident simultaneously, PyTorch allocator only; not per-method complete VRAM'
 report['object_mean_view_median_ms']={n:float(np.mean([v['warm_median_ms'][n] for v in report['views']])) for n in names};(out/'render_results.json').write_text(json.dumps(report,indent=2));return report
