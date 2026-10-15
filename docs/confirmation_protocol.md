# Round 28: dataset expansion and common-asset confirmation

Declared before Round 28 asset download, geometry screening or fitting. Previous
Round 17–27 results are known. Public objects were inspected in Round 19 and are
replications, not untouched holdouts. New objects exclude every historical asset,
candidate-manifest UID and UID-named experiment directory.

## Recruitment (independent of fitting results)

- Public replication: all 10 LocalHK main-table assets, all 8 low-poly assets,
  and all 5 official HKTex tuning examples from the pinned Round 19 UID map.
  Official tuning examples are descriptive replications, never a test set.
- New confirmation: a fixed hash-ordered metadata panel of at most 320 assets
  (240 with metadata V<=12000, 80 above), intersected with Objaverse LVIS.
  Eligibility: static, textured, downloadable, likes/views>=1, CC BY/BY-SA/CC0,
  metadata V=4..60000, GLB<=100MB; at most one new object per author, excluding
  previously encountered authors where locally available metadata identifies them.
  Metadata sampling frame is the union of already downloaded metadata shards,
  pinned to Objaverse revision 21e4e142159e2153706c23a3a02e55cec5591cea.
- Select the first 40 geometry-success objects in locked order, without viewing
  texture-fit results. Fewer than 40 is reported if the panel does not supply 40.
  Screen every candidate and retain every rejection. Geometry screen is the
  original connected/manifold/textured/ray<=20 check, finite positive-area faces,
  loaded V=4..60000, and finite isotropic eigenbasis. Original small-mesh rank
  reduction is retained. Full spectral failures do NOT trigger replacement.
- Download timeout 180s, screen timeout 180s, full spectral timeout 1800s.
  Existing public mesh/cache bytes can be reused only with hash verification;
  failed public caches receive one fresh declared attempt, without mesh repair.

## Frozen primary fitting recipe

All fresh parent fits use the same corrected-native, strict/eager HKTex adapter
as the current paper: source-colour initialization, original nearest-sampled UV
training target, priority sampler, loss, optimizer, ADC and 5000 updates. One
physical GPU per object/seed, identical cached spectral operator for all arms.
No newly proposed sampler, colour solver or learning-rate change is bundled in.

Primary arms: parent5000; capped-from-scratch5000 at K=4000/3000; corrected
deletion/refit post-hoc initializers; warm5000 at both budgets; full-capacity
parent restart5000 with reset optimizers and fixed parent support count.
Compare warm vs post-hoc, capped, parent, and full restart separately. Charge
parent training and compression to warm's end-to-end cost. Kernel count is not
assumed identical to file size. If parent N<=K, record budget ineligibility,
do not count that case as compression success; still retain parent/restart and
scratch controls. No best-seed selection or replacement after training failure.

Seed0 for every admitted object. Seeds1/2 for the first two rows of each public
LocalHK cohort and the first four admitted new objects, in frozen order; no
replacement if one fails. Official tuning examples use seed0 only.

## Evaluation and accounting

Same fixed 65536 area-uniform surface queries, seed100913; same original five
256-square, 4-spp PBR views, full-frame PSNR/SSIM/LPIPS, render seed119. Exported
FP16/NPZ is actually decoded. Fresh native-bilinear targets are separately
labelled if added; they do not silently replace the historical surface target.
The common-object comparison is a local matched HKTex/compression study, not a
reproduction of LocalHK-PF/PQ without its actual implementation. Published
LocalHK albedo metrics/A100 timings are not mixed with local PBR/3090 numbers.
Timing under concurrent workloads is descriptive; no isolated speed claim.
Standalone NPZ bytes, spectral cache bytes, failures and missing pairs reported.

Report cohorts separately and per-object paired effects, median/mean, win counts,
and object-level bootstrap 95% intervals. Within-object seeds averaged first.
Retain all outcome rows and reasons for missingness. These are still Objaverse
cohorts, not multiple independent source datasets or the original 313/162 set.
Current release remains frozen until the expanded results pass audit.
