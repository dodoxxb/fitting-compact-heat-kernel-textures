<div align="center">

# Fitting Compact Heat Kernel Textures

### Does a compact heat-kernel texture render better because it is *compact*, or because it receives *more fitting*?

**Zhicheng Du**<sup>1</sup>&nbsp; · &nbsp;**Lan Ma**<sup>1,†</sup>

<sup>1</sup>Tsinghua Shenzhen International Graduate School, Tsinghua University &nbsp;&nbsp; <sup>†</sup>Corresponding author

<p>
  <a href="https://doi.org/10.5281/zenodo.23401836">
    <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.23401836.svg" alt="Zenodo DOI">
  </a>
  <a href="paper/main.pdf">
    <img src="https://img.shields.io/badge/Technical%20Report-49%20pages%20PDF-brightgreen.svg" alt="Technical report PDF">
  </a>
  <a href="https://doi.org/10.13140/RG.2.2.25593.43366">
    <img src="https://img.shields.io/badge/Report%20DOI-10.13140%2FRG.2.2.25593.43366-blue.svg" alt="Report DOI">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
  </a>
  <a href="docs/reproduction.md">
    <img src="https://img.shields.io/badge/python-3.11-blue.svg" alt="Python">
  </a>
</p>

<p>
  <i>We study this question inside the HKTex representation, then separate three
  things that are usually reported together: conditional colour fitting,
  heat-feature consistency, and deployment cost.</i>
</p>

</div>

**TL;DR:** Compactness and fitting effort are confounded in most reports. We isolate them and find that compress–refine is worth **−0.060 to +0.074 dB** against an equally extended direct fit, conditional colour correction is worth **+0.586 dB** warm but only **+0.019 dB** after continuous fitting, and consistent heat features are worth **+0.277 dB** — while streaming evaluation gives a median **1.26×** render speedup.

---

📢 A Note on Scope
---

This repository publishes the code, the scalar evidence records and the compiled report. It is a **controlled study with scoped improvements** — **not** a new texture representation, and **not** a universally better optimizer.

- **Released**: 15 October 2026. Versioned software archives are preserved on Zenodo — this release is [DOI 10.5281/zenodo.23401836](https://doi.org/10.5281/zenodo.23401836) (`v2026.10.16`), and the concept DOI [10.5281/zenodo.23401835](https://doi.org/10.5281/zenodo.23401835) always resolves to the latest version.
- **The report** is a 49-page PDF: an eight-page main paper, two reference pages (32 entries) and a 39-page supplement (Appendices A–H). It carries its own DOI [10.13140/RG.2.2.25593.43366](https://doi.org/10.13140/RG.2.2.25593.43366) and is distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **No arXiv preprint and no conference acceptance is claimed.** The full HKTex multiview benchmark and the complete LocalHK method are **not reproduced** here.

---

✨ Highlights
---

| | |
| :--- | :--- |
| 🎯 **Confounds separated** | Compactness, fitting budget and colour correction are measured against matched baselines instead of pooled |
| 📊 **Per-contrast denominators** | Every finding states its own cohort size; follow-up cohorts are never added as new independent samples |
| ✅ **Failures retained** | Gap-stopped fits, unsupported meshes, incomplete seeds and routing failures are recorded, not dropped |
| 🖥️ **CPU-verifiable** | The full evidence chain checks on CPU in minutes — no GPU required to audit the numbers |
| 🔁 **Public recompute** | The conditional/cost result is recomputable from published data via `research/round49/` |
| 📦 **Integrity manifests** | `release_manifest.json` + `provenance.json` pin every distributed file and the vendored upstream snapshot |

---

## What the experiments establish

| Question | Finding | Denominator and scope |
| :--- | :--- | :--- |
| Does compact initialization beat more fitting? | Compress–refine minus equally extended direct: **−0.060 / +0.074 dB** at 4,000 / 3,000 kernels | 23 / 26 finite new-Objaverse pairs; editing costs extra |
| Is colour correction always useful? | **+0.586 dB** on warm-start development fields, **+0.019 dB** after continuous direct fitting | Separate 4-object and 19/24-object studies; no pooled effect |
| Can consistent heat features improve quality? | **+0.277 dB**, 15/19 positive object means | 22 declared, 19 complete with two seeds; timing not qualified |
| Can evaluation be faster? | Median **1.26×** render speedup with streaming | 15 mutually accuracy-qualified fields, RTX 3090; global spectral state retained |
| Does routing fewer terms help? | Median full-stream/routed frame-time ratio **0.415** | Four reused objects; the tested routing is *slower*; three fusion replay failures retained |

The fitting-route study admits 71 objects; 59 have valid seed-zero references. **Every contrast has its own denominator** — follow-up cohorts overlap earlier pools and are not added as new independent samples.

---

## Getting Started

### Installation

Python 3.11. Dependencies are split by task so a CPU check stays small:

| Task | Requirements file | Installs |
| :--- | :--- | :--- |
| Verify evidence, run tests | `requirements-analysis.txt` | numpy, scipy |
| Regenerate report figures | `requirements-figures.txt` | + matplotlib |
| Applicability analysis | `requirements-applicability.txt` | + scikit-learn, trimesh |
| Full GPU fitting | `requirements-gpu.txt` | + PyTorch stack, Mitsuba, DiGeo |
| Mechanism checks | `requirements-mechanisms.txt` | + numba |

```bash
python -m pip install -r requirements-analysis.txt
```

GPU installation requires matching PyTorch/CUDA and GPU Faiss builds. Follow the [GPU environment guide](docs/reproduction.md) before assuming the stack works — this package is **source, not a prebuilt container**.

### Verify the package on CPU

These commands inspect the included scalar evidence. They do **not** rerun omitted training or re-audit raw image and model binaries.

```bash
python tools/verify_release.py                 # integrity: 221 allowlisted files
python -m unittest discover -s tests -p 'test_*.py'
python tests/theory_contracts.py
python tools/summarize_records.py --input results/evidence.json --output outputs/routes
python research/round49/recompute_public.py --output outputs/conditional_and_cost.json
python tools/summarize_latest.py --output outputs/latest.json
```

`tools/verify_release.py` re-checks SHA-256 hashes against `release_manifest.json`. It verifies integrity only — it makes no claim of experimental completeness or clean-environment reproducibility.

Regenerate the principal route plot and the feature-confirmation table:

```bash
python -m pip install -r requirements-figures.txt
python tools/build_report_assets.py --output outputs/report_assets
```

### Mechanism checks

```bash
python -m pip install -r requirements-mechanisms.txt
python research/mechanisms/verify_identities.py
python research/mechanisms/summarize_mechanisms.py --input results/mechanisms/records.json --output outputs/mechanisms --assets
```

This adds 60 rendered fields at three real resolutions, 96 conditional fits including 36 gap-stopped checks, and 18 off-grid operator comparisons. It **reuses four historical texture objects** and two operator meshes, and therefore adds **no independent-object generalization count**. See [`docs/mechanisms.md`](docs/mechanisms.md) for explicit-input CPU/GPU tools and [`results/mechanisms/summary.json`](results/mechanisms/summary.json) for the summarized outcome.

### Fit a new mesh

Follow the [GPU environment guide](docs/reproduction.md) first, including native components and compatible Faiss/renderer dependencies.

```bash
python tools/build_native.py
python tools/single_object.py --mesh datasets/object.glb --gpu 0 --seed 0 --budget 3000 --output outputs/object_seed0 --plan
python tools/single_object.py --mesh datasets/object.glb --gpu 0 --seed 0 --budget 3000 --output outputs/object_seed0
```

The launcher runs the core parent, direct, edited/refined and continuation routes with decoded evaluation. Every fit uses the declared **5,000-update recipe**; parent training and editing are part of the warm route's cost. Use a fresh output directory — failed jobs remain inspectable. `--through parent` gives a smaller installation check.

Fixed-time cache construction is available via `tools/build_heat_feature_cache.py`. Streaming, conditional solving, learned readouts and experimental routing are included as documented research components; see [`docs/claim_map.md`](docs/claim_map.md) for what each one does and does not claim. Site-specific queues and historical checkpoints are **not** included.

---

## Repository layout

```text
├── hktex/                  Core HKTex package (mesh sampling, trainers, video utils)
├── optimisation.py         HKTex fitting entry point
├── tools/                  Release, verification, summarization and fitting tools
├── research/
│   ├── mechanisms/         Mechanism identity checks and summaries (CPU)
│   ├── round49/            Public recompute of the conditional/cost results
│   └── local_baselines/    Local comparison baselines
├── configs/                Render and texture fitting configurations
├── results/                Scalar evidence records (large raw dumps see below)
├── docs/                   Protocols, claim map, reproduction and data guides
├── data/                   Dataset metadata and object manifests (no assets)
├── paper/main.pdf          The 49-page technical report
└── tests/                  Unit tests and numerical theory contracts
```

Some large raw result dumps are excluded from Git history and distributed separately: `results/final_followup/training_records.json`, `results/deployment_followup.json`, `results/codec_controls.json`, `results/final_policies.json`. The smaller files needed by the CPU commands above remain in the repository.

---

## Data, provenance and limitations

[`docs/data.md`](docs/data.md) lists dataset identifiers, acquisition URLs, hashes and source licences. **No raw meshes, textures, pretrained image networks or experiment checkpoints are bundled.**

```bash
python tools/download_manifest.py --cohort new_confirmation --shell download.sh
# Inspect the acquisition plan and asset licences before executing it.
bash download.sh
```

* `data/heat_feature_objects.json` lists all 22 feature-confirmation objects, including unsupported meshes and the incomplete-seed object. Records preserve failure states rather than silently dropping them.
* Original export failures followed by qualified export-only recovery are distinguished from training completion; seeds are averaged within objects before aggregation.
* `provenance.json` records SHA-256 identities for the vendored HKTex source snapshot; `release_files.txt` / `release_manifest.json` are the reviewed distribution allowlist and its hashes.
* Hash and CPU checks do **not** claim fresh-machine GPU reproducibility. See [`docs/final_release.md`](docs/final_release.md) for the distinctions that matter when reading the numbers.
* Historical analysis scripts are intentionally not distributed. Observations that depend on them are quoted in the documentation rather than linked.

### Applicability

The expanded supplement restores proofs, initialization/sampling controls, complete-cost curves, local model tests and deployment trade-offs. [`docs/content_retention.md`](docs/content_retention.md) explains the change from the earlier single-column long report. [`docs/applicability.md`](docs/applicability.md) and [`docs/applicability_reproduction.md`](docs/applicability_reproduction.md) separate a useful fit-state interpretation from an unvalidated shape classifier.

---

## Citation

<details open>
<summary><b>Archived software release</b> — this is what the Zenodo DOI identifies</summary>

```bibtex
@software{du2026fitting,
  author    = {Du, Zhicheng and Ma, Lan},
  title     = {Fitting Compact Heat Kernel Textures},
  month     = {10},
  year      = {2026},
  publisher = {Zenodo},
  version   = {v2026.10.16},
  doi       = {10.5281/zenodo.23401836},
  url       = {https://github.com/dodoxxb/fitting-compact-heat-kernel-textures},
  note      = {Accompanying 49-page technical report: \url{https://www.researchgate.net/publication/415911132_Fitting_Compact_Heat_Kernel_Textures}}
}
```
</details>

<details>
<summary><b>Technical report</b> — has its own DOI</summary>

```bibtex
@techreport{du2026fittingreport,
  author      = {Du, Zhicheng and Ma, Lan},
  title       = {Fitting Compact Heat Kernel Textures},
  month       = {10},
  year        = {2026},
  institution = {Tsinghua Shenzhen International Graduate School, Tsinghua University},
  note        = {Technical report, 49 pages. CC BY 4.0},
  doi         = {10.13140/RG.2.2.25593.43366},
  url         = {https://www.researchgate.net/publication/415911132_Fitting_Compact_Heat_Kernel_Textures}
}
```
</details>

<details>
<summary><b>Original HKTex work</b> — please also cite</summary>

```bibtex
@inproceedings{foti2026hktex,
  title     = {Heat Kernel Textures: the Geodesic Gaussians That Do Not Splat},
  author    = {Foti, Simone and Korkmaz, Caner and Zafeiriou, Stefanos and Birdal, Tolga},
  booktitle = {ECCV 2026},
  pages     = {306--323},
  year      = {2026},
  doi       = {10.1007/978-3-032-37595-7_17}
}
```
</details>

**Which DOI to use:** `10.5281/zenodo.23401836` pins this software version and `10.5281/zenodo.23401835` (concept DOI) always points at the latest one; `10.13140/RG.2.2.25593.43366` identifies the report document itself.

---

## Acknowledgements

This work studies the [HKTex](https://arxiv.org/abs/2609.07557) representation introduced by Simone Foti, Caner Korkmaz, Stefanos Zafeiriou and Tolga Birdal; the [`hktex/`](hktex/) package, `optimisation.py` and two files in [`configs/`](configs/) derive from their MIT-licensed distribution. The native boundary-diagnostic component derives from [DiGeo](native/LICENSE) under BSD-3-Clause. We thank all upstream authors for releasing their code and data.

Issues about the fitting protocol, the mechanism checks or the reported figures are welcome; please include the configuration and the command you ran so the result can be reproduced.

---

## Licences and attribution

* **Software**: [MIT](LICENSE). The repository reuses the HKTex distribution, whose MIT notice is retained. [`hktex/`](hktex/), `optimisation.py` and the two files in [`configs/`](configs/) derive from it.
* **Native component**: `native/geodesic_boundary_diag.cu` derives from DiGeo under [BSD-3-Clause](native/LICENSE). DiGeo must be built against your local Torch/CUDA ABI, not copied as a binary.
* **Report PDF** (`paper/main.pdf`): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), DOI [10.13140/RG.2.2.25593.43366](https://doi.org/10.13140/RG.2.2.25593.43366). The report licence is deliberately stated separately from the code licence. Note that recoloured reconstruction examples shown in the report render assets obtained under their own third-party terms; those upstream terms are not altered by this declaration — see [`docs/data.md`](docs/data.md) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
* **Datasets and templates**: each retains its own terms. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
