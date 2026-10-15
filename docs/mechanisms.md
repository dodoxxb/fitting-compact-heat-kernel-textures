# Reproducing the mechanism extension

The Appendix H records reuse four development texture objects and two prior
operator meshes. They add no independent benchmark objects. All scalar outcomes,
including failed hypotheses, are in results/mechanisms/records.json.

## CPU summaries and identities

Install requirements-analysis.txt and requirements-figures.txt. From the code root:

```bash
python research/mechanisms/verify_identities.py
python research/mechanisms/summarize_mechanisms.py --input results/mechanisms/records.json --output outputs/mechanisms --assets
```

The summary must match results/mechanisms/summary.json. The plot and four tables
are regenerated from per-condition records. The checks include normalizer-floor
crossings, the bounded-readout counterexample, interpolation covariance/PSD,
the risk inequality, and finite-pool importance-weight expectation. They are
numerical checks of classical mathematics, not proofs of generalization.

## Run a new sampling control

Install requirements-mechanisms.txt. Supply a SciPy sparse response.npz with a
final constant background column, and arrays.npz containing target (N by3) and
C0 (K+1 by3). Kernel coefficients use[-1,1], background[0,1]. Declare disjoint
fitting-pool, validation and test slices explicitly. The example indices match
the historical exported arrays but can be changed for your own bundle.

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python research/mechanisms/probe_sampling_portable.py \
  --response inputs/response.npz --data inputs/arrays.npz \
  --fit 81920:147456 --validation 147456:155648 --test 155648:163840 \
  --sample-count 16384 --projection-rank 128 --seeds 101,202,303 \
  --max-sweeps 4000 --gap 1e-7 --output outputs/sampling.json
```

Uniform and a half-uniform/projected-leverage mixture both use replacement
sampling and inverse-probability weighting. The default is the follow-up gap
criterion; use `--max-sweeps 100` for the
initial100-sweep cap. A failed gap criterion is reported, not silently accepted.
Projection cost, fitting cost and selected fraction remain visible. The package
does not include historical response matrices or targets.

## Render an actual coefficient path

Install the GPU environment/native components in reproduction.md. Supply a mesh,
the teacher checkpoint metadata from the fitting pipeline, a compatible exported
field, and a colour NPZ with a full candidate matrix (background last). The
checkpoint config and mesh must match the field. From the code root:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
python research/mechanisms/render_colour_path.py \
  --mesh inputs/object.glb --checkpoint inputs/teacher.pt \
  --field inputs/field.npz --colours inputs/colours.npz \
  --resolutions 128,256,512 --fractions 0,.125,.25,.5,1 \
  --output outputs/colour_path
```

This re-renders five fixed cameras at each actual resolution and saves raw
linear and independently denoised sRGB images. It does not choose a fraction
from the images, run new geometry training, or claim a throughput benchmark.
Use the same native reference, camera/material settings, and a separate
validation rule if selecting an intervention. Four instances may be assigned
to separate GPUs with separate output directories. Do not share output paths.

The original historical run additionally checked 256/2048-query batching and
wrapper agreement before producing all60 fields. The exported entry point was
replayed on one real field (15images) with identical PSNR on the existing3090
host. This does not qualify every new asset or guarantee a clean GPU installation.

## Scope of operator and historical scripts

mechanism_math.py exposes the exact interpolation-variance and deletion-change
identities for arbitrary user-supplied matrices. The off-grid physical-reference
experiment additionally needs saved nodal heat factors, a matching mesh and
anisotropic eigensolver settings. Its exact execution source and the other
historical runners are under docs/experiment_sources_20261015 for inspection;
they expect a completed historical workspace and are not self-contained launchers.
Do not infer a texture-training benefit from these operator-only comparisons.
