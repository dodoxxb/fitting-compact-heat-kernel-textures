# Completed follow-up, 12 October 2026

The archive retains the earlier fitting-route, coding and deployment evidence.
This addition closes the existing conditional-fitting and fixed-time-feature studies.
It does not add a general LocalHK reproduction or a universal fitting policy.

## Recompute the final numbers

```sh
python research/round49/recompute_public.py --root . --output outputs/final_summary.json
python -m unittest discover -s tests -p 'test_fixed_time_features.py'
python tools/download_manifest.py --manifest data/confirmation_objects.json --shell outputs/download_confirmation.sh
```

Summary reproduction reads exported scalar records. It does not re-audit omitted
model arrays, rendered images or GPU timings. File integrity and experiment
qualification are separate. The frozen old confirmation analyzer expected
22 eligible objects; its exported top-level state remains `partial`. The
coverage-aware ledger accounts for all 24 reserved objects: 171 completed
cells, 27 cache-blocked cells and 18 fixed-rank unsupported cells. No replacement
or omitted failure is treated as a completed measurement.

## Conditional fitting protocol and interpretation

Nineteen qualified objects (12 Objaverse, 7 GSO), three training seeds and three
arms use 3,000 kernels and 5,000 continuous updates. Shared native SGD pools
contain one million points; batches contain 4,096 points. The common native
test contains 65,536 points and the render protocol five views. The correction
uses 65,536 fitting points, 8,192 guard points disjoint from that fit but drawn
from the SGD pool, 100 coordinate sweeps and step candidates 0, 1/8, 1/4, 1/2,
1. Interventions occur at updates 4,000, 4,250, 4,500 and 4,750. Matched reset
controls perform the same scheduled colour-optimizer resets, including after
rejection. Independent test values never select the step. The full recipe,
dependency/source hashes and pair plan are in the exported records and paper.

Mean correction-minus-reset rendering gain is 0.01930 dB (15/19 positive;
object-bootstrap 95% interval [0.00617, 0.03710]). Only 123/228 corrections
are accepted. Nine runs reject all candidates and have bitwise-equal final
parameter arrays to reset controls. Shared-resource work times do not qualify
as isolated speed measurements. The four-object partial local evaluator gives
0.00564 dB with two gains and two losses. Its uniform surface-validity rerun
has 36 qualified runs; the 21 original failed attempts are not pooled with it.
It omits author-system adaptive density control and proxy approximation.

## Rebuild fixed-time heat features on CPU

Install the HKTex GPU/environment dependencies described elsewhere, even for
this cache-only command (HKTex's import graph includes renderer modules).
Use a fresh output filename, an acquired source mesh and the supplied config:

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python tools/build_heat_feature_cache.py --hktex-root . --mesh datasets/example.glb --config configs/texture_hktex_knn.yaml --output outputs/features.pt
```

The command never calls a renderer and redirects its unused import-time
CUDA variant to LLVM. The standard configuration uses 256 spectral terms,
49 anisotropy/angle grid nodes and diffusion time 0.06245046192097831. It
recomputes RAW eigensolutions before alignment. Diagonal rescaling of an
already mixed legacy cache cannot restore the original kernel. For the
verified baseline, add `--variant fp64_diagonal`. Both variants use checked
FP64 orthogonal factors and identical sequential reference choices; stored
arrays are FP32. The isotropic nearest-neighbour embedding is unchanged.
Keep caches separate and refit each texture with the appropriate cache;
do not attach a changed cache to old weights and call it a quality-preserving
conversion. A different diffusion time needs a new cache.

The portable builder was run on Barricade and all five stored tensors match
the experiment cache bitwise. Four CPU tests cover kernel preservation,
noncommuting diagonal weights, rank deficiency and invalid inputs. This is
a local integration check, not a clean-machine GPU reproduction. The original
training recipes include site-specific admission and audit infrastructure;
this release does not claim a portable one-command replay of all new GPU runs.

## What the complete cost comparison establishes

Two diagnostic meshes times two seeds give four same-card pairs/eight fits.
Initial learned arrays, native targets and batches match within each pair;
the initial function need not match because the interpolant changes. Seed-one
execution reverses the variant order. There is no conditional correction.
All four improve rendering, native PSNR, SSIM and LPIPS. Rendering gains are
0.37716, 0.67194, 1.00926 and 0.82000 dB (two-object mean 0.71959).
Accounted total cost increases by 0.15--1.90%; this is not a speedup.

Each total sums measured fresh cache work, shared native-data preparation
charged equally, training, export, preflight and nine-checkpoint evaluation.
It is not an uninterrupted cold-start measurement. One-second monitoring
qualifies GPU occupancy on all pairs, not host or file-cache isolation.
Legacy endpoint records retain their original timing flag; pair-level
qualification is added separately, never retroactively substituted.
Curves exclude offline evaluation/export and the startup outside their clock;
the total-cost table retains those costs. Discrete common-budget comparisons
use only actual checkpoints, without interpolation. The object count is two;
seeds and repeated seed-zero pilots are not extra independent objects.

## Attribution

HKTex supplies the intrinsic representation, heat-kernel formulation and
original implementation. Five inspected files match public author commit
9e942972c5411a364f12563eb6864520f9effe70 byte for byte. This verifies those
files, not the provenance of every file in the historical local checkout.
The fixed-time feature identity is classical linear algebra. The analysis
concerns a particular basis/interpolation convention, not an error in the
heat equation or the original representation. Local FP32 SVD failures have
not been reproduced in the authors' environment. Full provenance is in
results/final_followup/UPSTREAM_AUDIT.json.
