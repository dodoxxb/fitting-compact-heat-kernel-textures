# Mechanism completion, 2026-10-15

Locked before the new outcomes. Historical development objects are Orange,
PoisonDartFrog, SmallLimo and PinkOmbreHair. They are not a new independent cohort.
Do not alter historical records or substitute failed objects.

1. Parameter-space render intervention: at update 500, interpolate the saved
initial colours and the saved full normalized conditional solution using
alpha = 0, 0.125, 0.25, 0.5, 1. Geometry and all response parameters stay fixed.
Render each field afresh at 128, 256, 512 pixels, five original cameras, seed119,
4 spp, original path integrator/material/lighting. Denoise independently at the
actual resolution. No resizing. Report both pooled and mean-view sRGB PSNR and
raw linear errors; no inference speed claim from this diagnostic. Same-query
wrapper and 256/2048 batch gates must pass. All alpha values are descriptive;
image-best alpha is an oracle, never a held-out selection rule. Compare the
surface-validation-selected alpha separately. Four GPU processes, one per card.

2. Sampling stability: all four historical objects at updates250/500/750.
Compare original16k, fresh16k, nested fresh65k response matrices, native targets,
box-constrained 100-sweep colour solves, exact solver gap, active bounds,
train/independent-test directional energy and per-column exposure. Inspect
Gram matrices on a fixed 128-dimensional Rademacher projection, whitened by a
separate target-free 16k anchor within the existing fresh pool. This is a
projected diagnostic, not a full-spectrum stability certificate. Larger65k
contains the anchor samples; explicitly report this dependence. No learned
predictor, statistical independence assumption, or cross-object claim.
CPU two concurrent workers, each limited to two BLAS threads.

3. Heat-factor interpolation: reuse raw eigensolutions from the two prior
operator-development meshes; compare factor interpolation and response mixing
with exactly identical nodal heat kernels. Fixed nine off-grid points and
128 vertices, plus deterministic factor sign changes preserving every node.
Measure kernel discrepancies, normalized operator positivity and finite-
difference parameter derivatives. This isolates interpolation semantics;
it cannot attribute the 19-object training gain to interpolation or predict
texture accuracy. No new training claim. If existing records cannot support
matched nodes, report the gap rather than impute results.

Literature: search public arXiv, OpenReview, GitHub, Hugging Face, ModelScope and
X/Twitter through 2026-10-15. Community findings generate questions, not evidence
of algorithmic effectiveness. Record unavailable or unrelated results explicitly.

## Follow-up locked after the initial sampling diagnostic, before its outcomes

The observed fresh16k versus nested65k gap motivates a finite-pool intervention,
not a new independent confirmation. At update500 on the same four objects,
compare 16,384 replacement samples from the same65,536-row pool, for seeds
101/202/303: uniform sampling versus a 50:50 uniform/projected-ridge-leverage
mixture. Both objectives use inverse-probability row scaling 1/sqrt(N*p_i), so
both are unbiased estimates of the same finite-pool quadratic risk. Leverage
is computed in the fixed128-dimensional projected response space, with ridge
1e-6 times mean Gram eigenvalue; it is not exact full-space leverage. Colours
are box-constrained, 100sweeps. Select alpha among0/.125/.25/.5/1 on native
surface validation, score unchanged independent test. Report full steps too,
all seeds, solve gaps and preprocessing/solve/selection cost separately.
Do not interpret replacement-sampling uniform as the historical distinct16k.

## Solver-accuracy follow-up

The100-sweep weighted controls have non-negligible empirical box gaps. To avoid
attributing solver error to sampling, resolve all three sample pools at update500
and all24 weighted controls using the exact same sampled matrices/targets, with
stopping half-MSE box gap <=1e-7 or a hard4000-sweep cap. Report achieved gaps and
all caps. This follow-up was selected after inspecting the100-sweep outcomes;
it tests robustness to solve accuracy and remains exploratory. No renderer or
sample selection is changed. Run two CPU workers, two BLAS threads each.
