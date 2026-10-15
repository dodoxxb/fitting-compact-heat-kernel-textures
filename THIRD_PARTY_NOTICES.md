# Source attribution

## HKTex

The `hktex/` package, `optimisation.py`, and the two files under `configs/` derive
from the original HKTex distribution by Simone Foti, Caner Korkmaz, Stefanos
Zafeiriou and Tolga Birdal. Its MIT licence and copyright notice are retained
in `LICENSE`. The local source snapshot had no Git metadata; its exact files
are identified by SHA-256 in `provenance.json`. A historical commit is not
invented. An obsolete example path in a source comment was made generic.

Original work: *Heat Kernel Textures: the Geodesic Gaussians That Do Not Splat*,
ECCV 2026, DOI [10.1007/978-3-032-37595-7_17](https://doi.org/10.1007/978-3-032-37595-7_17),
[arXiv:2609.07557](https://arxiv.org/abs/2609.07557).
[Original project page](https://circle-group.github.io/research/HeatKernelTextures/).

The surrounding experiment, compression and audit code was developed for this
study. Copied experiment routines keep their historical filenames so their
hash provenance can be traced. They do not rename or claim authorship of the
underlying heat-kernel representation, geodesic optimiser or renderers.

The repository MIT terms cover software outside the separately identified
BSD-licensed native source. They do not automatically license the new manuscript,
rendered third-party assets, or original datasets. The manuscript's distribution
and arXiv licence are selected separately by its authors.

## DiGeo

`native/geodesic_boundary_diag.cu` derives from DiGeo under BSD-3-Clause. Its full
licence is retained in `native/LICENSE`.

Verified upstream source:
[commit 5557253bd33121ea40060f601ca0a27a29b47683](https://github.com/circle-group/DiGeo/commit/5557253bd33121ea40060f601ca0a27a29b47683).
The original CUDA file SHA-256 is
`cc181af043a7c56f9224a40a06d242d081401942cd45c05134bba58ec4c0b572`.
The audited patch changes the initial local edge index from `edge_idx - 1` to
`edge_idx` and places registration in an isolated namespace. This fixes a
boundary-start implementation issue; it is not a new geodesic method.

The extension must be compiled locally. No precompiled DiGeo, Torch, CUDA,
Faiss, Mitsuba, OptiX, LPIPS or image-model binaries/weights are distributed.
Those dependencies retain their own licences.

## Data

Public metadata identifies external objects; source assets remain with their
original providers. Objaverse objects have individual author/licence metadata.
The selected Kubric GSO entries explicitly state CC BY-SA 4.0, which is preserved
instead of assuming a collection-wide CC BY licence. Downloading an asset does
not relicense it under this repository's software licence.

## Follow-up source attribution

Five numerically relevant files were verified byte-identical against public
HKTex commit 9e942972c5411a364f12563eb6864520f9effe70. See
`results/final_followup/UPSTREAM_AUDIT.json`. This does not establish the
origin of the entire historical checkout or reproduce the author environment.

## Publication template and figure provenance

The report was typeset with the official
[CVPR author kit](https://github.com/cvpr-org/author-kit), commit
`291758547e923160eb4d37079b7b9f0dfce82355`; its upstream notices are retained.
Only the compiled `paper/main.pdf` is distributed — LaTeX sources, style files
and intermediate figure assets are not part of this repository. The template
does not imply conference acceptance.

The report PDF (`paper/main.pdf`) is distributed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), DOI
[10.13140/RG.2.2.25593.43366](https://doi.org/10.13140/RG.2.2.25593.43366).
This differs by design from the code licence (MIT). The method overview figure
is a conceptual illustration; every other figure derives from recorded data or
from reconstruction views rendered over separately acquired assets, whose
upstream licences are listed in `docs/data.md` and are **not** altered by the
report's CC BY 4.0 declaration. Reference HKTex PDF pages are not distributed.
