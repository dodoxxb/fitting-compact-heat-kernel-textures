# Independent source validation: Google Scanned Objects, Kubric distribution

Declared before downloading any candidate mesh or fitting any GSO object.
Round28 Objaverse runs are underway. No GSO result has informed the method.

Use the Google/Kubric public GSO v1.0 manifest (1033 entries in the retrieved
snapshot, not assumed identical to the 1030-object original article count).
The manifest declares individual licenses; preserve these strings, including
CC BY-SA 4.0 where given, and source attribution. Data source:
https://storage.googleapis.com/kubric-public/assets/GSO/GSO.json
Documented by Google/Kubric's movi_c_worker.py and asset_source.py. Original GSO:
https://research.google/blog/scanned-objects-by-google-research-a-dataset-of-3d-scanned-common-household-items/

Hash-order all static FileBasedObject entries with metadata 4..60000 vertices;
retain at most two candidates per named category and lock the first 24. Empty
or `None` category is not selected. Use seed
`hktex-round28-gso-external-20260922-v1`. Download/screen all 24 (or the available
number if fewer), and take the first eight geometry-success objects with at
most one per category. Do not replace spectral or training failures.

Use the original visual OBJ with its PNG/MTL, not collision geometry. Convert
to GLB once for the inherited runner's filename contract, without remeshing,
repair, decimation or texture resizing. Verify per-face vertex coordinates
(2e-6 relative scale), UV corners (2e-6), face count and exact decoded RGB
texture bytes against the OBJ before accepting conversion. Keep archive and
converted hashes. Screen the converted file with the same original geometric
predicates, small-mesh spectral-rank reduction and 4..60000 vertex range.
Archives<=150MB, download<=180s, screening<=180s, spectral<=1800s.

All eight admitted objects run seed0; first two admitted run seeds1,2 also.
Exactly the parent/capped/posthoc/warm/full-restart recipe and same evaluation
as Round28, plus matched capped restart, native UV ladder and precision codec.
Training uses original TF32/high, deterministic algorithms, eager execution;
evaluation uses TF32 off/highest. Source-colour initializer and corrected native
geodesic adapter are unchanged. Absolute K=3000/4000 ineligibility is retained.
No hyperparameter tuning on this panel. Source-specific aggregates are kept
separate: this checks applicability across independently sourced assets, not
transfer of a learned network or multiview reconstruction.

GPU workers wait for the corresponding Round28 primary and secondary workers
to terminate, then acquire the same per-GPU lock. No overlap on a physical GPU.
UIDs in this panel are local SHA256-derived identifiers from `GSO:<asset_id>`;
they are explicitly NOT Objaverse UIDs. Original asset_id/category are published.
