# Final audited evidence

`evidence.json` is the terminal Round28/30 export: 71 planned objects, 91
object--seed slots, 728 training-arm outcomes, 1,011 decoded-model measurements
and 413 UV records. These are not 1,011 independent objects. There are 581 valid
evaluated training fits; blocked, invalid, environment-failed and ineligible
outcomes remain explicit. Exact-view cases retain pooled metrics.

`numerical_diagnostic.json` contains the separate six-run selected-case follow-up.
It does not replace failed confirmation outcomes.

`scale_followup.json` contains the completed exploratory update-scale follow-up:
eight previously observed objects (four Objaverse and four GSO), one seed and
four continuation routes, with all 32 final endpoints retained. Each route uses
5,000 additional updates, a 3,000-kernel budget and uniform fixed-pool sampling.
This differs from the original priority-sampled confirmation. The original
September protocol excluded this pilot from v1; the completed pilot is now
included only as a separate exploratory appendix. It changes no confirmation
score, failure record or denominator in `evidence.json`.
The original frozen protocol is quoted verbatim in the report's exploratory
appendix, including its original exclusion statement. Historical round scripts
are not distributed with this repository; the 5,000-update recipe below is the
authoritative description of the actual update rule.

The export records per-object metrics, initializer/model/query/source SHA-256
hashes, the selection and evaluation protocol, and every adaptive update with
scale below one. Omitted adaptive updates have scale exactly one. This lossless
encoding of the scalar sequence permits CPU recomputation of 810 reduced steps
out of 40,000; all eight objects have median scale one. Full training traces,
models and query arrays are not bundled. Artifact hashes identify the original
inputs and outputs; they do not substitute for distributing those artifacts.

Run `python tools/summarize_scale_followup.py` from the release root to recompute
paired mean-view render-PSNR effects, medians, positive counts and scale
statistics. The command does not rerun training or establish a speedup. The
adaptive-minus-default means are +0.0217 dB for the selected Objaverse objects
and +0.0052 dB for GSO. A fixed one-tenth scale loses on all eight objects.
The small, selected, single-seed comparison does not establish an optimal scale
rule or independent population-level improvement.

`warmup_followup.json` closes the simple-control gap with eight additional fits
on those same objects. A single prescribed multiplier increases from 0.1 to 1
over the first 100 updates; no duration is selected using its endpoint quality.
Initial parameters, RNG state and the first loss match the reused controls,
and each object retains its original GPU. All eight fits completed. An initial
hook-location setup error occurred before any optimization and is recorded
separately; it is not counted as a numerical quality failure.

```bash
python research/round35_warmup/recompute_public.py --input results/warmup_followup.json
```

Warmup minus default averages +0.01177 dB across eight objects. Adaptive minus
warmup averages only +0.00166 dB, with mixed object-level signs. These descriptive
results establish neither statistical equivalence nor an adaptive-controller
advantage. All adaptive scale reductions occur before the first scheduled
learning-rate decay, which prevents attributing their end solely to improved
initialization. The scale branch is closed without further parameter tuning.

`codec_controls.json` completes the scalar-coding comparison with the equally
extended direct controls. All 177 object--precision tasks and all 840 prescribed
endpoints passed the terminal audit, with no new coding/evaluation failures.
The endpoints comprise 162 hash-verified historical measurements and 678 new
evaluations of actual decoded fields. They use the original seed-zero models:
59 references, 103 refinements and 118 extended-direct fields, each at three
fixed codec precisions. These repeated configurations do not add independent
objects. The complete original 71-object admission ledger and training failures
remain in the export; failed or unavailable fits were not repaired or replaced.

```bash
python research/round35/analyze.py --recompute results/codec_controls.json --output outputs/codec_recomputed
python research/round35/write_paper.py --input results/codec_controls.json --output outputs/codec_tables
```

For the primary codec setting on new Objaverse objects, refined minus extended
direct is -0.06190 dB at 4,000 kernels (23 pairs, 7 positive) and +0.06521 dB at
3,000 kernels (26 pairs, 15 positive). The corresponding mean relative changes
in actual payload bytes are +0.629% and +0.454%. The 3,000-kernel effect has a
descriptive object-bootstrap interval spanning zero; it is not evidence of a
uniform improvement. Both fitting routes use two stages of 5,000 updates, with
editing and different first-stage kernel counts remaining additional resource
differences. All precision settings, paired effects, medians, intervals and
actual byte ratios are retained, rather than selecting a favourable setting.

The 177 repeated-reference calibrations accompany the endpoint records. They
measure cross-process rendering drift and are not subtracted from coded
scores. Standard-setting reference/refined scores reuse verified historical
PBR measurements; direct scores and finer-setting scores are newly evaluated.
Texture-payload bytes refer to complete `.jhk` files, excluding shared geometry
and the still-required spectral cache; decoded working NPZ bytes are not
compressed storage. Concurrent elapsed times do not establish a speedup.

The byte-constrained comparison is a retrospective envelope of six measured
direct candidates per object. It selects the lowest measured pooled MSE among
candidates fitting the refined bitstream's actual size, with no interpolation.
No feasible candidate is not a win, and zero-MSE cases remain valid. This
envelope is neither a deployable selection rule nor a global Pareto claim.
See [the codec-control guide](../docs/codec_controls.md); the frozen protocol
is quoted verbatim in the report appendix and historical round scripts are not
distributed here. CPU commands reproduce the
recorded arithmetic; model, query, payload and image hashes identify omitted
original artifacts rather than distributing them.

## Deployment pilots integrated on 3 October 2026

`deployment_followup.json` contains raw paired outcomes for the separate
conditional-solver, scalar-rounding, streaming, calibration-gate and support
experiments. Eight reused objects have two routes; four historical detail
objects form a separate replay. These records neither alter primary evidence
nor increase independent confirmation coverage. Use
`python tools/summarize_deployment_followup.py` for CPU recomputation and see
`docs/deployment_followup.md` for failure and compiler contracts.

`final_policies.json` records all 16 final policy fields, four historical
base/hard/smooth comparisons, four post-outcome equal-sweep controls and the
compacted-reference failure replay. Run `python tools/summarize_final_policies.py`
to verify frozen choices and recompute outcomes. Source-assisted calibration,
existing PBR views and post-outcome control status are retained. All five detail
candidates are included; selection never removes failed alternatives from the
record. Original primary evidence remains unchanged.
