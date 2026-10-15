# Reproduce the exploratory applicability analysis

This is post-hoc analysis of existing results, not new training or a validated selection policy. The two cohorts share 18 objects and must not be pooled. Geometry clustering does not use effects, but the analyst had already seen aggregate effects. No object is removed from the primary estimates.

```bash
pip install -r requirements-applicability.txt
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python tools/analyze_applicability.py --output outputs/applicability.json
python tools/build_applicability_assets.py --output outputs/applicability_figures
```

The first command consumes included scalar records and geometry/input descriptors. It uses fixed k=2, deterministic initialization and permutations, a family of 18 associations, fixed-penalty ridge leave-one-object-out prediction, and a training-fold mean baseline. Texture and baseline-quality predictors are separate from geometry-only clustering. Permutation groups are iterated in sorted order to make results independent of the Python hash seed. The second command reads the released analysis and frozen historical cost points to regenerate two measured-data figures; it does not rerun timings.

Optional descriptor re-extraction requires separately acquired input meshes and the original prepared seed-zero training pools (not bundled):

```bash
python tools/extract_applicability_descriptors.py --data-root ./experiment_data --output outputs/descriptors.json
```

Mesh paths are those in `data/confirmation_objects.json` and `data/heat_feature_objects.json`. Training pools are sought in `research/round47/confirmation/prepared/<uid>/seed0/train_pool.npz`, then the `data` subdirectory of `research/round50/prepared/<uid>`, followed by `seed0/train_pool.npz` below the data root. The extractor runs the released mesh-loader function without importing GPU dependencies, uses exact triangle surface moments and a single-thread nearest-neighbor query, and records unavailable inputs rather than fabricating descriptors. No quality outcome files are read by the extractor. Fresh extraction requires the same converted assets and loader semantics; record-only reproducibility is a distinct claim.

Heat-feature effects average two complete seeds. Correction effects average three complete seeds against reset-only continuation. Four heat-feature objects have negative effects. Neither the positive sign count nor the post-hoc grouping supplies a causal mechanism or a deployable router. See `docs/applicability.md` and Appendix G for interpretation.
