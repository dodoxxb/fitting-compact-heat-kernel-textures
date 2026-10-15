# Claims, evidence and executable components

The report's main question is whether compact-field quality gains survive fair
fitting controls. Conditional fitting and deployment then ask where computation
is useful. The release does not combine different protocols into one synthetic
quality/speed score.

| Claim | Records | Entry / component | Boundary |
|---|---|---|---|
| Matched updates change the compact-initialization advantage | `results/evidence.json`, `results/codec_controls.json` | `tools/summarize_records.py`, `tools/single_object.py`, `research/round35/analyze.py` | 71 admitted, 59 valid references; finite pairs differ by capacity |
| Correction depends on fitting state | `results/final_followup/confirmation.json`, `local.json`, `COMPLETED_COHORTS.json` | `research/round49/recompute_public.py`; conditional solver in `research/round36` and `research/round38` | 19/24 reserved objects; partial independent local evaluator, not full LocalHK |
| Fixed-time feature consistency improves held-out quality | `results/latest/heat_features.json`, `heat_features_audit.json` | `tools/build_heat_feature_cache.py`, `research/round49/fixed_time_features.py`, `tools/summarize_latest.py` | 22 declared, 19 two-seed complete; equal arrays, potentially different initial functions; no speed claim |
| Separate diagnostic pairs improve at similar accounted cost | `results/final_followup/REPLAY_RESULTS.json` | `research/round49/recompute_public.py`, `plot_cost.py` | Two diagnostic objects, two seeds; not the broader confirmation's timing |
| Streaming reduces measured latency and temporary allocation | `results/deployment_followup.json` | `research/round37/streaming.py`, `tools/summarize_deployment_followup.py` | RTX 3090; retained basis and extra decay cache; query qualification required |
| Conditional response structure matters after recolouring | `results/latest/mechanisms.json`, `background_gauge.json` | `tools/summarize_latest.py` | Four frozen fields; not full retraining ablations |
| An extra branch need not beat a stronger single branch | `results/latest/readout_controls.json` | `research/round51/reduced_math.py`, `research/round52`, `research/round54/models.py` | Nine finite historical objects; failed/exact cases retained |
| Fewer candidates do not imply faster rendering | `results/latest/routing.json`, `routing_replay.json`, `routing_audit.json` | `research/round57/routing.py`, `fused_routing.py`, `tools/summarize_latest.py` | Four historical fields; failed speed criterion and three failed fusion replays; no expansion |

Historical directory numbers are provenance identifiers, not names of proposed
models. Modules under `research/` are inspection/reuse components, unless a
standalone command is explicitly documented. The publication does not present
these components as an end-to-end validated automatic budget allocator.

The projection identity, box first-order gap, diagonal majorizer and fixed-time
Gram invariance use classical mathematics. Our contribution is the empirical
question, controlled evidence and scoped implementations, not discovery of these
identities. Rendering can disagree with a surface-error proxy. Numerical
agreement can fail at positive top-k ties. A local-code audit must not be
attributed to an error in the original HKTex theory.

## Exploratory applicability

Appendix G uses the two separate 19-object cohorts; 18 objects overlap. `tools/analyze_applicability.py` consumes `results/applicability/descriptors.json` and the existing endpoint records. No validated geometric applicability class or selection policy is claimed. `tools/build_applicability_assets.py` regenerates its scatter plot and the restored historical continuation-cost figure. See `docs/applicability_reproduction.md`.
