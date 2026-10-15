# Coded fitting-route controls

`results/codec_controls.json` contains the terminal, independently audited
codec completion on the original seed-zero fields. It preserves the complete
71-object admission ledger and original training failures, the 59 valid
references, all prescribed endpoint statuses, object-level measurements,
actual bitstream sizes, input/artifact hashes, and repeated-reference
calibrations. It neither repairs failed training outcomes nor replaces the
original confirmation evidence.

The available source fields are 59 references, 103 refinements and 118
extended-direct controls. Each is evaluated with the existing C8B12S8,
C8B14S10 and C8B16S12 settings: 840 prescribed coded endpoints. The standard
C8B12S8 setting is primary; finer geometry/response precision is a sensitivity
check. Colour coefficients use eight bits throughout. The paired refined and
extended-direct routes each receive two stages of 5,000 updates. This matches
update counts, not end-to-end computation: refinement also incurs editing and
its first stage may use more kernels.

Default-setting reference/refined PBR results are reused only after checking
the original model and fixed-query hashes, byte-identical newly encoded
payloads, decoded-array identity, evaluator/renderer source and protocol,
and image metrics. Direct-codec controls and finer-setting controls are
new evaluations. Evaluation uses the actual decoded fields in the original
five-view PBR renderer. The independent surface-query score is a separate
metric. Each new evaluator process also renders the unchanged reference;
the export reports calibration drift without subtracting it from scores.
Standard-setting comparisons combine verified historical and new processes.
Concurrent render timings are not controlled speed measurements.
The evaluator disables TF32 matrix multiplication and uses `highest` float32
matmul precision; it does not explicitly disable cuDNN TF32 convolution paths.

## Recompute the reported statistics on CPU

From the release root, install `requirements-analysis.txt` and run:

```bash
python research/round35/analyze.py \
  --recompute results/codec_controls.json \
  --output outputs/codec_recomputed
```

This recomputes object-paired effects, medians, signs, fixed-seed bootstrap
intervals, byte-constrained candidate summaries and repeated-reference
calibration summaries. It uses the public audited records and NumPy; no
GPU, model files, query files or image files are required. It is arithmetic
regeneration, not a new audit of absent binary artifacts. The default
filesystem-audit mode of `analyze.py` expects the original experiment tree
and is not a one-command public rerun of the 71-object study.

Generate the supplementary LaTeX tables and macros from terminal evidence:

```bash
python research/round35/write_paper.py \
  --input results/codec_controls.json \
  --output outputs/codec_tables
```

The writer rejects incomplete evidence. Finite mean-view PSNR counts are
distinguished from all valid paired fields. Exact-match views remain in the
coverage and pooled-MSE records; undefined infinite-PSNR differences are not
treated as ordinary finite gains. Original failure records and artifact hashes
remain in the export, even when their original binaries are not redistributed.

## Evaluate your trained fields with all three settings

After the documented single-object training pipeline has produced compatible
parent, warm and budget-capped continuation models, run:

```bash
python tools/representation_controls.py \
  --teacher outputs/object_seed0/parent \
  --queries outputs/object_seed0/queries.npz \
  --models parent:outputs/object_seed0/parent/model.npz,warm:outputs/object_seed0/warm/model.npz,capped_restart:outputs/object_seed0/capped_restart/model.npz \
  --codec-modes C8B12S8,C8B14S10,C8B16S12 \
  --skip-uv --gpu 0 --output outputs/object_codec_controls
```

This command requires the GPU dependencies and a fresh output directory.
Omit `--skip-uv` to also evaluate the native UV ladder. The supplied query file,
teacher checkpoint, mesh and spectral cache must belong to the same object.
The `capped_restart` field is the 10,000-update direct control, not its
5,000-update predecessor. The one-object command accepts a trained budget;
the released full study spans two source caps and does not claim to expose
its entire original multi-GPU scheduler as a portable entry point.

Use the `.jhk` payload sizes recorded in `controls.json` under
`codec[model_and_mode].bytes`. The `bytes` fields in the raw evaluator refer
to decoded NPZ working files and must not be used as compressed storage.
Payload accounting includes all texture arrays, indices and headers. Common
geometry and spectral caches remain required and are outside these bytes.
Equal source caps and precision settings do not imply equal actual file sizes
or equal runtime memory.

## Scope of the measured candidate envelope

For a refined bitstream's actual byte budget, the analysis considers six
measured extended-direct candidates: two source caps times three fixed codec
settings. Only candidates no larger than the refined payload are feasible;
the lowest observed pooled image MSE defines the reported envelope. All six
candidates must be present. An incomplete candidate set and no feasible
candidate are separate outcomes, and no feasible candidate is not a quality
win. Zero MSE is retained.

The envelope chooses among measured test scores after evaluation. It is a
descriptive comparison of finite candidates, not a deployable configuration
selector, an interpolated equal-byte method or proof of a global Pareto
optimum. The three fixed settings are all reported rather than selecting a
favourable precision separately for each fitting route.

The frozen protocol is quoted verbatim in the report appendix; historical round
scripts are not distributed with this repository.
`provenance.json` records the original hashes of the exported evidence and
the current analysis/writer/protocol snapshots. Paths in the evidence identify
original experiment artifacts; they do not claim those binaries are bundled.
