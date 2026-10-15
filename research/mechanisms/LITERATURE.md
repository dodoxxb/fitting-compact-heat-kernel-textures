# Public literature and actionable questions — 15 October 2026

This is a targeted search, not an exhaustive review or a claim that every
community post was accessible. Searches covered arXiv, OpenReview, official
GitHub repositories, Hugging Face paper pages, ModelScope, and X/Twitter.
No social-media assertion is used as scientific evidence. Publication dates
and the dates on which we encountered a work are different.

| Verified primary source | Useful question | Action in this release |
|---|---|---|
| [Cyr et al., MSML 2020](https://proceedings.mlr.press/v107/cyr20a.html) | What does solving linear output variables actually establish? | Cite the direct adaptive-basis/LSGD precedent; retain the bounded, clipped, sampled and rendered distinctions. No claim to invent alternating least squares. |
| [Cohen–Migliorati, 2017](https://doi.org/10.5802/smai-jcm.24) | Is sampling the fitted response space stably? | 12 historical events with projected Gram and update-energy diagnostics; 24 weighted/uniform finite-pool fits with inverse-probability correction. |
| [Frozen-PINN, ICLR 2026](https://arxiv.org/abs/2405.20836) | Can each source of numerical error be controlled separately? | Distinguish solve gaps, sampling transfer and deployment; no import of its PDE/ODE accuracy or speed claims. The arXiv v3 was revised 15 April 2026; this is not a new October submission. OpenReview forum/PDF access hit a browser challenge; the primary arXiv paper provides the verified reference. |
| [HKTex official code](https://github.com/circle-group/hktex) | What readout and bounds does the released representation implement? | Retain attribution and exact normalization/background conventions; pinned source audit in prior review. |
| [LocalHK / ThermalRF](https://arxiv.org/abs/2609.14105) | Does operator consistency survive interpolation? | Same nodal heat factors, 18 off-grid tests, factor/response interpolation and finite-difference derivatives. ThermalRF already motivates response mixing; this is not our independent invention. No complete author-code reproduction claim. |
| [Neural Harmonic Textures](https://arxiv.org/abs/2604.01204), [official code](https://github.com/nv-tlabs/neural-harmonic-textures) | Does a more expressive local appearance model improve the actual sampled image objective? | Discuss the distinct roles of spatial support, appearance capacity and supervision. Do not add an unmatched new texture benchmark. |
| [Mip-Splatting](https://arxiv.org/abs/2311.16493), [official code](https://github.com/autonomousvision/mip-splatting) | Does a fitted correction transfer across pixel sampling rates? | Actual128/256/512 rendering at fixed cameras, 60 fields and five views each, raw and denoised outputs. Resolution robustness is not proof of aliasing removal. |
| [Shared Gaussians for mipmapped SVBRDFs, July2026](https://arxiv.org/abs/2607.27943) | Could support be shared across appearance channels/scales? | Explain as a future cross-channel question requiring full material data, per-level targets and byte accounting. Current RGB/base-colour assets cannot establish a material-compression claim. No verified author implementation was found in the inspected paper. |
| [TexF](https://arxiv.org/abs/2609.23606), [RTXNTC](https://github.com/NVIDIA-RTX/RTXNTC) | Which resource does compression save? | Preserve separate file, resident-state, decode-stage and render-cost reporting. |

Hugging Face: the [Mip-Splatting paper page](https://huggingface.co/papers/2311.16493)
was accessible and links back to arXiv. Its generated summary is not treated as
independent evidence. The direct NHT paper page did not load in this session.
ModelScope and X/Twitter targeted searches did not yield a verifiable, directly
relevant new technical discussion for this question. Unrelated results were
excluded; this does not establish that no such discussion exists.

The promising synthesis is a diagnostic one: a small optimization gap is a
statement about a sampled objective, whereas usefulness is a statement about
an independent field/image objective at a specified resolution and cost.
Classic identities make those conditions explicit. They do not imply a new
universal optimizer, and a negative sampling intervention must remain visible.
