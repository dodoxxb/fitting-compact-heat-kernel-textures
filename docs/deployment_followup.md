# Completed deployment follow-up

This study is included in the integrated manuscript as a separate pilot. It
uses eight existing objects and two 3,000-kernel routes, with a separate replay
of four available historical detail models. It does not increase the primary
confirmation denominator or replace its training initializer.

Run `python tools/summarize_deployment_followup.py` to recompute the main
deployment quantities from `results/deployment_followup.json`. The JSON joins
raw per-field records; machine-local path references are removed or made
relative while numeric outcomes are retained. Paired object means and
field-level timing medians have different sampling units. View and timing
repetitions must not be treated as independent objects.

The conditional solver reaches the common calibration gap on 16/16 fields.
Its 6.264x median solve-time comparison includes only six fields on which the
global accelerated control also converges. The control uses a conservative
global curvature bound; this is not a best-known-solver claim. Matrix
construction and device preparation are separate from solve time.

The streaming operator retains the spectral basis and original routing.
On fitted four-bit fields, its qualified full-render speed ratio is 1.256x over an additionally compiled
query control, on 15 mutually passing fields. Historical raw label `Eager`
retains component compilation. The later six-bit composition disables
compilation; its speed ratios are not interchangeable with this comparison.
Incremental PyTorch temporary allocation is not total GPU memory.

Fitted six-bit colours reduce field bytes by 5.47% versus nearest eight-bit
coding at a mean 0.082 dB loss; decoded runtime coefficients remain FP32.
The calibration-only backend gate fails one held-out field. Using all 50
already-computed responses saves selection work but loses up to 0.482 dB
on a lowpoly model. These failed controls are part of the evidence.

Inspected source modules are supplied for the sparse colour solver, selected
readout, streaming contraction, solver accuracy benchmark and numerical
theory checks. GPU replay still requires the prepared original fields,
response matrices, spectral state and rendering environment. Those inputs and
the complete experimental launcher overlay are not included in this source-only
release. These modules have been run on the configured RTX3090 host; no new
clean-container or cross-architecture validation is claimed.

AudioCache is cited as indirect motivation for deciding what to reuse and how
to validate reuse, not as this operator's algorithm. TexF and RTXNTC motivate
separate bitstream, decoded-state and execution accounting. None of these
external systems is reproduced by this pilot.

The subsequent bounded follow-up is documented in [final policies](final_policies.md).
It tests explicit selection and extra-fitting controls without replacing the
above unconditional outcomes. Source-assisted readout selection avoids the
observed lowpoly loss in this pilot; precision selection and query-wise
fallback still fail some checks. The final detail effect uses the separately
identified, post-outcome equal-sweep control.
