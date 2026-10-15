# GPU reproduction and portability

The single-object launcher is a newly packaged entry around the same numerical
routines used in the expansion. It preserves the configuration, seed adapter,
calibration deletion/refit and continuation hooks; it does not reuse fitted
scientific checkpoints. Training enables TF32 matrix multiplication and matmul precision `high`;
decoded evaluation disables TF32 matrix multiplication and uses `highest`.
This does not assert that cuDNN convolution paths, including perceptual metrics,
are explicitly configured to disable TF32. Eager execution and strict
deterministic algorithms do not mean TF32-disabled training.

## Environment

The experiment host used Linux, Python 3.11, PyTorch 2.8.0+cu129,
torchvision 0.23.0+cu129, GPU Faiss 1.13.1, DiGeo 0.0.7, Mitsuba 3.7.1 and
Dr.Jit 1.2.0 on RTX 3090 GPUs. These are observed versions, not a claim that a
clean environment has been rebuilt and tested on every platform.

A starting installation recipe is:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.8.0+cu129 torchvision==0.23.0+cu129 --index-url https://download.pytorch.org/whl/cu129
python -m pip install -r requirements-gpu.txt
```

GPU Faiss 1.13.1 must be installed using a CUDA-compatible distribution. The
upstream HKTex environment uses the `faiss-gpu-cuvs` conda package. CUDA toolkit,
compiler, NVIDIA driver and OptiX support are system prerequisites, not ordinary
Python dependencies. Build DiGeo against the installed PyTorch ABI; an unrelated
prebuilt extension may fail with missing symbols. Then compile the included
boundary correction with `python tools/build_native.py`. `TORCH_CUDA_ARCH_LIST`
may be set explicitly when building without a visible GPU.

Image metric evaluation may fetch the standard torchvision/SqueezeNet weights
through TorchMetrics on its first use. Those weights are not bundled. Set
`TORCH_HOME` to a writable cache directory if desired.

## Commands and constraints

`tools/single_object.py --help` exposes mesh, GPU, seed, kernel budget, output,
optional spectral cache and stopping stage. The input must be a textured GLB;
`tools/convert_gso.py` supplies the checked OBJ-to-GLB adapter used for GSO.
The original source-loader semantics, spectral cache format and mesh topology
checks still apply. The launcher never repairs rejected geometry.

The parent uses 1,000 initial sources and the original adaptive density control
up to 5,000. Budget-limited scratch retains the original adaptive control under
a smaller cap. Warm and both continuations reset optimiser/scheduler state and
freeze the source count. Thus equal update counts do not imply equal wall time.
If the parent has no more kernels than the requested budget, compression is
marked ineligible rather than counted as a win.

The compressed initializer is produced on the fresh 4,096-point calibration
set. The final evaluator uses distinct 65,536-point queries and the original
five PBR views at 256 square pixels and four samples per pixel. The stronger
rebuilt-support colour fit is also present as an individual source entry;
it does not silently replace the frozen warm initializer.

The default wrapper is designed for the paper's 4,000/3,000 budgets. The legacy
compression script also computes a disjoint-merge ablation and therefore may
reject a budget requiring more disjoint merges than available. This is not a
guarantee of arbitrary-ratio compression. Run commands in a new output directory;
all generated hooks and local extension metadata are ignored by version control.

## What is still separate

The public single-object entry does not replay the complete historic adaptive
search, the 71-object scheduling order, LocalHK training,
or all numerical-failure diagnostics automatically. The frozen protocols and
per-object results describe these scopes. The endpoint should be described as
"single-object core recipe plus audited evidence", not an automatic complete
paper reproduction. Clean-environment installation remains a separate release
validation requirement; the accompanying audit states exactly what was tested.

`tools/representation_controls.py` separately performs the native UV ladder and
the frozen scalar-quantisation control on an existing parent checkpoint and
query set. It rejects nonfinite model arrays before encoding, records actual
bitstream sizes, decodes before rendering, and retains exclusions. The scalar
codec does not eliminate the common spectral operator. Its UV helper functions
were extracted verbatim from the study's original implementation; their source
hashes and the import-only change are recorded in `provenance.json`.
