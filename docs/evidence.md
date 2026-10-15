# Evidence format and aggregation

The frozen expansion is complete. It admits 71 objects, of which 59 have a
valid seed-zero reference; eligibility and endpoint validity further determine
each paired denominator. Separate numerical, scale and warmup diagnostics do
not increase this independent-object count or replace confirmation outcomes.
The current inventory and the commands for those separate exports are in
[`results/README.md`](../results/README.md).

Frozen protocols retain their historical wording. The closing paragraph of
[`precision_contract.md`](precision_contract.md) describes the earlier,
pre-expansion manuscript snapshot; its "pending expansion" statement is not
the current release state. Likewise, the original round-31 scale protocol
recorded the decision to exclude that pilot from v1; the completed pilot and
the later fixed-warmup control now appear as exploratory appendices, with the
main confirmation evidence unchanged. Historical round scripts are not
distributed with this repository, so pre-execution declarations are quoted
here rather than linked.

The final `results/evidence.json` is a data-only, audited export containing:

- `state`, which must identify incomplete previews explicitly;
- `coverage`, all admitted objects and expected seeds;
- `trials`, including failed/incomplete training outcomes;
- `training_outcomes`, every prescribed arm, distinguishing environment failure,
  numerical/initialisation failure, invalid export, eligibility and blocked dependencies;
- `measurements`, valid decoded models, hashes, source counts, representation
  bytes and image/surface metrics;
- `uv`, native-UV ladder scores and complete representation bytes;
- `invalid_completed_fits`, processes that completed their step schedule but
  produced invalid models;
- per-object `pairs`, descriptive `summary`, and source hashes.

The public summary tool independently reconstructs seed-zero pairs from
`measurements`. It requires a model hash and relies on the finite-model gate
performed by the evidence exporter. It does not pretend to recheck absent
checkpoint arrays from scalar scores. Invalid fits remain in the evidence even
when absent from measurement pairs. Each cohort and comparison retains its own
denominator; no seed is selected for quality.

A CUDA initialisation failure before training is an environment outcome, not
evidence of unstable optimisation. A model with nonfinite parameters after a
finite loss trace is a different failure. Blocked descendant arms are retained
but are not counted as independent failure cases.

The tool also accepts a simple interchange form with `objects`, `records`, and
`comparisons`. Each record has `uid`, `cohort`, `seed`, `method`, `sources`,
`valid_model` and a `metrics` mapping. Each comparison has `method`, `reference`
and optional `budget`. This form supports independently created replication
records. Numeric infinities or NaNs are excluded from finite-difference means,
with an explicit reason rather than a fabricated finite value.

The final manuscript may report additional repeated-seed, frontier and failure
analyses; this small CLI deliberately states its seed-zero scope. A larger
number of saved fits is not a larger independent dataset.

`tools/write_paper_results.py` is the portable version of the manuscript's
expanded-results writer. It reads this canonical export and the independent
six-run numerical-diagnostic export, writes only beneath `--output`, and
regenerates coverage, matched-quality, post-hoc, coding, UV-dominance,
multi-metric, interval and repeated-seed tables, numeric macros, and the paired
effect plot. Study-specific explanatory text accompanies these tables; this is
a reproduction writer for this study, not a general report generator for other
datasets. Its input must be `terminal_audited` unless `--allow-preview` is given.

Precision wording in the frozen protocols refers to the configured matrix
products: the evaluator sets CUDA matmul TF32 off and float32 matmul precision
to `highest`. It does not explicitly disable cuDNN TF32 convolution paths.
The current manuscript and reproduction guide state this narrower contract.
