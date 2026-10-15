"""Resource adapter: bounded FAISS scratch and checkpointed spectral gathers.

No optimizer, samples, update count, readout or heat-feature recipe changes.
Large constant spectral arrays stay on CPU; small arrays reside on GPU.
Chunking can change floating-point gradient accumulation order, so it needs
value/gradient qualification and is applied identically to both variants.
"""
import inspect
import torch
from torch.utils.checkpoint import checkpoint

CHUNK=2048
GPU_BASIS_LIMIT=400*1024**2

def _evec_chunk(m_idx,m_w,v,b,basis):
    n=len(m_idx);dim=basis.shape[-1]
    out=torch.zeros((n,4,dim),dtype=basis.dtype,device=m_w.device)
    indices=m_idx.to(basis.device)
    verts=v.to(basis.device)
    for i in range(3):
        vi=verts[:,i].unsqueeze(1).expand(-1,m_idx.shape[1])
        values=basis[indices,vi].to(m_w.device)
        out=out+values*b[:,i].view(n,1,1)
    return (out*m_w.unsqueeze(-1)).sum(dim=1)

def evecs(m_idx,m_w,v,b,basis):
    def fn(mi,mw,vi,ba):return _evec_chunk(mi,mw,vi,ba,basis)
    parts=[]
    for start in range(0,len(m_idx),CHUNK):
        args=[x[start:start+CHUNK] for x in (m_idx,m_w,v,b)]
        parts.append(checkpoint(fn,*args,use_reentrant=False) if torch.is_grad_enabled() and (m_w.requires_grad or b.requires_grad) else fn(*args))
    return torch.cat(parts)

def evals(m_idx,m_w,values):
    def fn(mi,mw):return (values[mi]*mw.unsqueeze(-1)).sum(dim=1)
    parts=[]
    for start in range(0,len(m_idx),CHUNK):
        args=(m_idx[start:start+CHUNK],m_w[start:start+CHUNK])
        parts.append(checkpoint(fn,*args,use_reentrant=False) if torch.is_grad_enabled() and m_w.requires_grad else fn(*args))
    return torch.cat(parts)

def install():
    import faiss
    original_resource=faiss.StandardGpuResources
    def resources(*a,**kw):
        obj=original_resource(*a,**kw);obj.setTempMemory(32*1024**2);return obj
    faiss.StandardGpuResources=resources
    import hktex.modules.eigen_albo as em
    import hktex.knn_heat.knn_heat.knn_gather as gm
    original=em.EigenAlboInterpolation.precompute_all_eigen
    source=inspect.getsource(original)
    import textwrap
    source=textwrap.dedent(source)
    needle='all_eigen.to(torch.float32).to(self.device)';assert source.count(needle)==1
    source=source.replace(needle,'all_eigen.to(torch.float32).cpu()')
    scope=dict(vars(em));exec(compile(source,inspect.getfile(original),'exec'),scope)
    em.EigenAlboInterpolation.precompute_all_eigen=scope['precompute_all_eigen']
    base=em.EigenAlboInterpolation.configure
    def configure(self,mesh):
        base(self,mesh)
        self._eigen_val=self._eigen_val.to(self.device)
        if self._eigen_vec.numel()*self._eigen_vec.element_size()<=GPU_BASIS_LIMIT:
            self._eigen_vec=self._eigen_vec.to(self.device)
    em.EigenAlboInterpolation.configure=configure
    gm._ops._gather_evals_flat=evals
    gm.SpectralKnnGather._make_evecs_fn=lambda self:evecs
    # The same ordered UV candidates and first matching faces are retained.
    # Only the point dimension of the validity check is chunked more finely.
    import hktex.trainers.uv_texture as uv
    source=textwrap.dedent(inspect.getsource(uv.UvTextureTrainer.data_dependent_initialisation))
    needle='chunk_size = 1024';assert source.count(needle)==1
    source=source.replace(needle,'chunk_size = 128')
    scope=dict(vars(uv));exec(compile(source,inspect.getfile(uv.UvTextureTrainer),'exec'),scope)
    uv.UvTextureTrainer.data_dependent_initialisation=scope['data_dependent_initialisation']
