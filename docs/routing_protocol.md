# Early routing pilot, frozen before its outcomes

Four historical development fields: Orange, SmallLimo, Hummingbird, Piggybank.
No new independent confirmation objects or training seeds are claimed.
Frozen HK3000 fields and Round53 native splits: 8192 fit, 4096 calibration,
16384 test. No colour/geometry refit. Source supervision is available for
offline calibration and its costs are charged.

Full outer routing remains 50 biharmonic neighbours; baseline inner readout
is 30. Both distance and a fitted per-source quadratic geometric score rank
candidates BEFORE expensive spectral gathering/contraction. Distance weighting
is computed over all original outer candidates, not recomputed on a subset.
Retained filtered responses use the actual partial denominator max(1,sum w)
and the original at-most-30 readout. Unchanged full50 is always a candidate.

Quadratic score: ten polynomial terms in normalized relative XYZ, ridge 0.1,
fit to log(max(filtered response, 1e-6)), coefficients/scales stored FP16,
all calibration and deployment use the decoded coefficients.
Fixed candidate budgets: 8,16,24,32,40,50. Adaptive prediction uses a tail
estimate from the predicted top30 weights, with tolerance grid
0.000125,0.00025,0.0005,0.001,0.002,0.004,0.008,0.016 and reserve 0,8,16.
Its estimate is empirical, not a certified bound on unseen queries.

Oracle controls evaluate all responses before choosing. They are headroom
diagnostics, not accelerated methods: fixed inner8/16/24/30 and adaptive
inner counts at RGB tolerances from the same eight-value grid, using exact
full-field distortion after partial renormalization.

For each family, select the smallest average candidate count among candidates
whose native calibration PSNR is no more than 0.05 dB below full streaming.
Tie-break on calibration MSE. Full baseline is an admissible fallback.
Selection does not use test or rendered outcomes. Retain all candidates and
all failures. Counts are not timing measurements or a Pareto proof.

Evaluate original, full streaming, distance fixed, quadratic fixed,
quadratic adaptive, oracle fixed, oracle adaptive. All have the same source
field, geometry, precision, renderer, batch size and compiler settings.
Test PBR on the original five views (seed119) and five shifted views
(azimuth offset36°, elevation15°, seed120). View results do not select policies.
Measure shuffled synchronized query timings, PBR render+denoise, preparation,
encoded gate bytes, gate tensors, and global basis bytes. Oracle timing never
counts as early-routing acceleration. GPU co-tenancy and host contention are
recorded; single GPU generation does not imply a second-hardware validation.

The 2x2 query timing uses full/adaptive with reference/streaming contractions.
Matched backend accuracy is checked. Any additional unmeasured transfers,
allocator scopes or renderer costs are stated explicitly.

Boundary stress: nearby points on a shared face, with small barycentric
perturbations, reporting incremental error relative to the original field;
this is not a complete seam/temporal guarantee.

Expansion rule: proceed to new-object confirmation only if early routing
passes both view sets within 0.05 dB on every development object, achieves
median >=1.20x full-render speed versus full streaming, and introduces no
nonfinite values. A fallback is reported as a fallback, never a new success.
If this rule fails, close this pilot and explain the failed mechanism. It is
an engineering go/no-go rule, not a conference acceptance criterion.
