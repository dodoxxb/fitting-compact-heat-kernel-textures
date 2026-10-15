# Final bounded deployment studies

Run `python tools/summarize_final_policies.py` from the release root. This CPU
command recomputes all frozen choices and summary quantities from
`results/final_policies.json`. It requires only Python's standard library.
`research/round44/selection.py` is a pure-Python extraction of the measured
choice rules, checked against every recorded field. It accepts calibration
MSE and complete payload bytes, not test outcomes.

Eight existing objects supply two correlated fields each. New area-uniform
calibration/test queries are disjoint draws (32,768 and 65,536), but the five
PBR views remain the earlier diagnostic views. No new independent objects,
new viewpoint confirmation or joint precision/readout policy is claimed.

- Readout selection retains inner selection for both routes of the lowpoly
  failure and selects fitted all-response readout for 14 other fields. Mean
  paired-object render speed ratio is 1.0363, with mean PBR change -0.00374 dB.
  All 16 source-surface and diagnostic PBR changes satisfy the 0.05 dB loss
  allowance in this pilot. This is source-assisted selection, not a uniform
  quality certificate.
- Precision selection uses six bits on nine fields and eight on seven;
  none selects four. Complete field bytes fall by 2.8824% on average, with
  mean PBR change -0.02128 dB. Surface checks pass 16/16 but image checks
  only 14/16. Runtime coefficients remain FP32. The original eight-bit
  comparator is nearest rounding, not an optimally fitted eight-bit codec.
- Query-wise reference fallback preserves the original outer candidate IDs,
  improves fresh query passes from 14/16 to 15/16, but passes all three query
  sets on only 14/16 fields. Mean speed relative to unguarded streaming is
  0.583x. Compacted reference arithmetic still differs from the original
  batch on Pink Bow. This heuristic is experimental and is not a reliable
  default. The scope differs from the earlier compiler-matched benchmark.

Four historical objects separately compare smooth and hard local detail.
New training/validation/test surface draws have sizes 32,768/8,192/32,768.
All detail variants use 300 base sweeps plus 300 further sweeps. A post-outcome
control continues the base itself to 600 total sweeps; it changes no frozen
selection. Smooth 64-unit detail beats that base on all four objects, with
mean PBR gain 0.11853 dB, 2.6469% additional field bytes and 1.6224% additional
render time. Timing has only two measured repeats. Final base gaps are not
zero, so this is an equal-sweep comparison, not a converged capacity bound.
The same-count hard comparison averages +0.02269 dB; byte sizes can differ.
The validation-selected byte-feasible base/hard alternative averages
+0.05958 dB. That restricted candidate set is not a global Pareto frontier.
The selected alternative is chosen on source-surface validation, not PBR.

The serialized mode is charged as a five-byte DPL1 or LDT1 envelope around
the original codec. The local audit verifies all 88 complete envelopes and
checks decoded geometry against the executed geometry. These containers
still require external mesh and spectral state. `unpack_mode` only validates
and extracts the mode and underlying codec bytes; it is not a complete
standalone renderer or spectral-state decoder.

Frozen experimental scripts and protocols are included as inspected snapshots.
GPU replay needs the original prepared fields, spectral caches, native source
assets and historical manifests not redistributed here. The repository's
primary training/codec entry points remain the supported end-to-end paths;
the historical deployment-study launcher is not presented as one-command
clean-environment reproduction. Four initial equal-sweep jobs failed at import
before output creation; the import path alone was repaired, and original logs
remain in the local study ledger. This is distinct from a numerical failure.

The final paper retains failures alongside the positive choices. It supports
measured, optional deployment allocations rather than an automatically optimal
combined system or a universally superior representation.
