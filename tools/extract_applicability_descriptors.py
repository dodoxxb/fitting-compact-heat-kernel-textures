"""CPU-only exploratory descriptors; outcome files are never read here."""
from pathlib import Path
import ast,json,hashlib,types,sys
import numpy as np
import scipy.spatial
import PIL
import argparse
import trimesh
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data-root',type=Path,required=True,help='Mesh assets and original prepared training-pool layout; no GPU required.')
parser.add_argument('--code-root',type=Path,default=Path(__file__).resolve().parents[1])
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();ROOT=args.data_root.resolve();CODE=args.code_root.resolve();args.output.parent.mkdir(parents=True,exist_ok=True)
# Execute the existing loader function without importing the GPU package tree.
source=(CODE/'hktex/utils/mesh.py').read_text();tree=ast.parse(source)
node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='load_mesh')
ns={'trimesh':trimesh,'hktex':types.SimpleNamespace(info=lambda *a:None)}
exec(compile(ast.Module(body=[node],type_ignores=[]),'<existing mesh loader>','exec'),ns)
load_mesh=ns['load_mesh'];rows=[]
meta={x['uid']:x for x in json.loads((CODE/'data/confirmation_objects.json').read_text())['objects']}
meta.update({x['uid']:x for x in json.loads((CODE/'data/heat_feature_objects.json').read_text())['objects']})
for uid,row in sorted(meta.items()):
 result=dict(uid=uid,name=row.get('feature_name',row['name']),dataset=row.get('dataset',row.get('source')),status='pending')
 try:
  mesh=load_mesh(str(ROOT/row['path']),merge_tex=True,normalise_size=True)
  v=np.asarray(mesh.vertices);faces=np.asarray(mesh.faces);tri=v[faces];e01=tri[:,1]-tri[:,0];e02=tri[:,2]-tri[:,0];area=np.linalg.norm(np.cross(e01,e02),axis=1)/2
  if row.get('screening',{}).get('V'):assert len(v)==row['screening']['V'],(len(v),row['screening']['V'])
  # Exact first and second moments of the uniform surface distribution.
  sums=tri.sum(axis=1);first=np.einsum('f,fi->i',area,sums/3)/area.sum()
  second=np.einsum('f,fij->ij',area,(np.einsum('fki,fkj->fij',tri,tri)+np.einsum('fi,fj->fij',sums,sums))/12)/area.sum()
  eig=np.linalg.eigvalsh(second-np.outer(first,first));eig=np.maximum(eig,0)
  den=(e01**2).sum(1)+(e02**2).sum(1)+((tri[:,2]-tri[:,1])**2).sum(1)
  quality=4*np.sqrt(3)*area/np.maximum(den,1e-30)
  feats=dict(log10_vertices=float(np.log10(len(v))),log10_area_cv_plus_one=float(np.log10(1+area.std()/area.mean())),pca_flatness=float(np.sqrt(eig[0]/eig[2])),pca_elongation=float(np.sqrt(eig[1]/eig[2])),mean_triangle_quality=float(np.average(quality,weights=area)))
  train=ROOT/'research/round47/confirmation/prepared'/uid/'seed0/train_pool.npz'
  if not train.exists():train=ROOT/'research/round50/prepared'/uid/'data/seed0/train_pool.npz'
  if train.exists():
   with np.load(train) as data:
    count=min(16384,len(data['colour']));c=data['colour'][:count].astype(float);pos=data['pos'][:count].astype(float)
   feats['training_colour_variance']=float(np.mean(np.var(c,axis=0)))
   dist,ids=scipy.spatial.cKDTree(pos).query(pos,k=2,workers=1)
   feats['training_local_colour_difference']=float(np.mean((c-c[ids[:,1]])**2))
   result['training_descriptor_points']=count
  result.update(status='complete',vertices=len(v),faces=len(faces),features=feats,source_sha256=hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest())
 except Exception as exc:result.update(status='unavailable',error=type(exc).__name__+': '+str(exc))
 rows.append(result);print(uid[:8],result['name'],result['status'],flush=True)
args.output.write_text(json.dumps(dict(scope=__doc__,geometry_loader_sha256=hashlib.sha256(source.encode()).hexdigest(),objects=rows),indent=2,ensure_ascii=False)+'\n')
