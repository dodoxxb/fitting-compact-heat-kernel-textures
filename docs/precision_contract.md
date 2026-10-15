# Implementation details, fixed before the first new fit

The protocol's “strict/eager” means strict deterministic algorithms and disabled
Torch compilation. Inspection confirms that the inherited author entry enables
TF32 and float32 matmul precision `high` for training. Round 22 parent status
files also record `tf32=true`, `matmul_precision=high`. Round 28 preserves this
setting in **every** arm; it does not claim TF32-disabled training. Final decoded
evaluation disables TF32 and uses `highest`, as in the current paper.

Fresh 4096-point calibration draws use seed280017. Test65536 draws retain
seed100913, and are not used for editing/refit/selection. This new calibration
seed is shared across arms of an object and across fit seeds; the exact old
per-object calibration draw is not needed to compare freshly trained arms.
The corrected cached deletion/refit initializes warm training; a rebuilt-graph
colour refit is additionally reported as a stronger post-hoc control. It is not
used to select the warm initializer after examining any test metric.

The source and package preserved in `paper_before_expansion/` are historical
pre-expansion snapshots. The expansion and subsequent controls are now complete
in the current manuscript and terminal evidence; those archived PDFs are not
the current submission candidate. The separate deployment pilot's compiler
settings are documented in `deployment_followup.md` and must not be inferred
from this training-precision contract.
