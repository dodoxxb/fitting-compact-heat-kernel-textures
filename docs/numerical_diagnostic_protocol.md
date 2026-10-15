# Round29: bounded numerical-failure diagnostic

Declared after inspecting Round28 results, before these runs. Exploratory,
selected failure case; never a replacement for confirmation failures.

Object: GSO ProSport_Harness_to_Booster_Seat, local UID
9f5837c34fcc43bf222f72ed96c0f15e, parent seed0 and existing corrected deletion
initializers. The original warm4k has finite recorded losses but nonfinite
FP32 positions/angles/anisotropies. Thus export-only corruption is ruled out.

Matrix: K=4000/3000, learning-rate scale=1/0.1/0.01, 5000 updates, seed0.
Replay scale1 to check reproducibility with observation enabled. All optimizer
group learning rates scale together; same targets, sampler, optimizers,
schedulers, fixed kernel counts, corrected native code and decoded evaluation.
Use the original physical GPU2, currently released by Round28 GSO jobs.
Capture the first nonfinite gradient and post-update parameter occurrence.
Observation does not clip, replace, resample, repair or stop an update.
Retain failed outcomes. No choice of the best scale is a held-out result.

Six fits only. No additional architectures, datasets or unconstrained search.
Compare source hashes, finite state/export status and original five-view PBR.
These experiments are independent of the locked Round28 primary/control queues.
