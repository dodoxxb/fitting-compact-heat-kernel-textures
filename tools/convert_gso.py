"""OBJ->GLB representation adapter, with per-corner geometry/UV/texture checks."""
from pathlib import Path
import argparse,json
import numpy as np,trimesh
p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args()
src=trimesh.load(a.input,force='mesh',process=False)
assert isinstance(src,trimesh.Trimesh) and src.visual.kind=='texture'
def image(m):
    mat=m.visual.material;im=getattr(mat,'baseColorTexture',None)
    if im is None:im=getattr(mat,'image',None)
    assert im is not None,'No decoded colour texture';return np.asarray(im.convert('RGB'))
before=image(src);out=Path(a.output);src.export(out,file_type='glb')
dst=trimesh.load(out,force='mesh',process=False)
assert src.faces.shape==dst.faces.shape
x=src.vertices[src.faces];y=dst.vertices[dst.faces];scale=max(np.ptp(x,axis=(0,1)).max(),1e-9)
geom=float(np.max(np.abs(x-y))/scale)
uv=float(np.max(np.abs(src.visual.uv[src.faces]-dst.visual.uv[dst.faces])))
same=bool(np.array_equal(before,image(dst)))
report=dict(input=a.input,output=a.output,source_vertices=len(src.vertices),converted_vertices=len(dst.vertices),faces=len(src.faces),max_relative_corner_position_error=geom,max_corner_uv_error=uv,exact_rgb_texture=same,remeshing=False,resizing=False)
out.with_suffix('.conversion.json').write_text(json.dumps(report,indent=2))
assert geom<=2e-6 and uv<=2e-6 and same,report
