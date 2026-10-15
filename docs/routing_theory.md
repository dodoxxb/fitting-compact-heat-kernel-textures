# What the routing experiment tests mathematically

These are classical inequalities and cost identities used to interpret the experiment, not claims of new mathematical theory. They apply to a frozen field; they do not make geometry fitting convex or certify an unseen query distribution.

## 1. Conditional fitting isolates one source of error

For fixed kernel geometry, fixed query support and an unclipped readout, write the sampled field as `AC+b`. Solving a constrained least-squares colour problem tests what this particular response matrix can explain. The empirical optimization gap is the difference between the current coefficients and the optimum over the stated feasible colour set. Estimating that gap requires a sufficiently accurate solve or an optimization certificate; a finite iterative run is not automatically the optimum.

An empirical residual can still contain sampling error and representation error. A small fitting residual cannot identify either without independent samples. Updating positions or shape changes `A`; a conditional solution need not remain optimal and does not guarantee a better subsequent geometry trajectory. The earlier continuation controls are therefore necessary evidence, not optional embellishments.

## 2. Dropping responses changes numerator and denominator

Let the full and deployed unnormalized readout weights, padded to a common candidate set, be `u_i >= 0` and `v_i >= 0`. These include their respective top-k selections; a response outside either selected set is assigned zero. Define

`s = sum_i u_i`, `s' = sum_i v_i`, `d = max(1,s)`, `d' = max(1,s')`,

`N = sum_i u_i c_i`, `N' = sum_i v_i c_i`, `C = max_i ||c_i||_infinity`,

`E = sum_i |u_i - v_i|`.

Both fields use the same background and componentwise clipping. Clipping is nonexpansive. Adding and subtracting `N'/d` gives

`||f-f'||_infinity <= ||N-N'||_infinity/d + ||N'||_infinity |d-d'|/(d d')`.

Since `||N-N'|| <= C E`, `|d-d'| <= |s-s'| <= E`, and `||N'|| <= C s'`,

`||f-f'||_infinity <= C E/d + C s' |d-d'|/(d d') <= 2 C E/d`.

This bound covers missed support, changed retained weights and displacement of the original top-k set. It is conservative; it does not assume positive RGB experts. In the special case of deleting a subset of the original readout without replacing weights, `E` is exactly the omitted mass. If both sums are at most one, both denominators equal one and the sharper bound is `C E`. If the original denominator is explicitly retained, only the numerator term remains, which is the different contract used in the preceding cached-response probe.

The quadratic router predicts response scores, but it has no uniform bound on `E`. Calibration on a finite surface sample therefore yields an empirical operating point, not a certified deployment guarantee. The fact that the selected native operating point passed the rendered tolerance on these four fields does not turn its predicted tail into a proof.

## 3. A small field perturbation is not a fixed quality loss

On one fixed query distribution `q`, let `L = E_q ||f-Y||_2^2/3`, `D = E_q ||f'-f||_2^2/3`, and `delta=f'-f`. Expansion and Cauchy--Schwarz give

`L' - L = 2 E_q[(f-Y) dot delta]/3 + D <= 2 sqrt(L D) + D`.

The cross term can have either sign. Deleting responses can improve the measured surface loss while worsening another target. For `L>0`, keeping PSNR loss below `eta` requires `L'/L <= 10^(eta/10)` on that same distribution. An absolute RGB tolerance alone is not this relative-error contract.

Rendered PSNR is computed after a different sampling process, illumination, visibility, sRGB conversion and denoising. No global stability bound for that whole pipeline is provided here. A local differentiable approximation `delta_image ≈ J delta_field` suggests measuring error with render sensitivity, but discontinuous support, visibility and nonlinear postprocessing prevent treating this as an exact global identity. SmallLimo is a concrete counterexample to transferring the tested surface-calibration rule unchanged to image quality; it does not identify which renderer component causes the mismatch.

## 4. Less arithmetic must pay for its selection

For the same source field and evaluation backend, write

`T_full = T_shared + T_eval(50) + T_readout`,

`T_route = T_shared + T_score + T_select + T_dispatch + T_eval(K) + T_readout'`.

These are a bookkeeping identity with measured components, not a linear model of GPU time. Reducing `K` is useful only if the avoided evaluation/readout time exceeds the added scoring, selection and dispatch time. Allocator, batching and occupancy effects can change either evaluation term. In particular, `50/E[K]` is not a measured speedup or a rigorous hardware upper bound.

If the extra preparation cost is `P >= 0` and deployment renders `n` frames under the measured workload, total savings require

`n (T_full - T_route) > P`.

When steady-state routing is slower, increasing reuse cannot amortize it into an advantage under that workload. All three non-fallback routed fields have that measured outcome, including after fusion at the query level. A different workload or implementation would require a new measurement.

## 5. Implication for method design

There are three separate requirements: retain enough information, identify it before expensive evaluation, and execute it efficiently. A gate that reads both expert outputs has already incurred both costs. A gate that predicts identities beforehand can save evaluations but can still lose on latency and add nearly a second compact field's worth of encoded coefficients, as here.

For a future experiment, a candidate table shared by a surface patch could amortize identification across coherent queries. That changes storage, approximation and boundary behavior and has not been implemented or validated in this round. It should not be described as an achieved method or as novel without related-work and implementation comparisons. The current pilot's predetermined stop rule has been respected.
