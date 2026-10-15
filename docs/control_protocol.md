# Matched continuation and representation controls

Declared while the first four new parent fits are still running, before any
Round 28 parent endpoint or quality metric is available. These controls apply to
every completed object/seed, not only favorable cases.

1. Restart each completed capped-from-scratch 3000/4000 model for 5000 updates,
   resetting optimizers/schedules and freezing the source count, exactly as warm
   and full-parent restart do. This compares two 5000+5000 update curricula with
   the same continuation contract. Source kernel counts can differ when ADC did
   not saturate; record them. Total time, not just steps, remains separately
   charged. This avoids relying only on a 5000-step scratch comparator.
2. Seed0 on every admitted object: native UV PNG ladder at long edges
   32,64,128,256,512,1024, and source resolution. Charge complete PNG plus UV
   coordinates and texture-face indices; share identical underlying geometry.
   Keep both original-target and native-bilinear surface diagnostics labelled.
   The source-resolution rendering is an identity contract, not a new method.
3. Seed0: apply the previously frozen C8B12S8 codec to parent/warm4000/warm3000,
   decode and render them; charge actual bitstream bytes rather than the decoded
   NPZ. This keeps precision reduction as a comparator to changing kernel count.

The four secondary workers wait for their corresponding primary GPU lock, so
they never overlap a primary job on the same GPU. Cross-GPU CPU contention can
still occur; these are not isolated throughput measurements.
