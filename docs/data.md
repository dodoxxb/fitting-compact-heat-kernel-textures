# Data and admission

`data/objects.json` contains 71 admitted identifiers: ten LocalHK main-table
objects, eight LocalHK low-poly objects, five official HKTex tuning examples,
forty newly recruited Objaverse objects, and eight GSO objects from Kubric.
Admission precedes fitting. Spectral failure or numerical failure does not remove
an object from the manifest. Some parents have fewer than a requested kernel
budget and are therefore ineligible for that compression comparison.
There are 59 valid seed-zero references; the other 12 admitted objects remain
in the coverage ledger. Neither 71 nor 59 is the number of valid pairs at every
budget. Additional seeds and the scale/warmup follow-ups reuse objects and
must not be counted as new independent data.

The Objaverse revision is pinned in every corresponding row. Public objects are
replications, and tuning examples are descriptive; they are not untouched test
data. Newly recruited objects use the fixed metadata sampling frame and geometry
screen described in `confirmation_protocol.md`. HKTex reports 313 UV-fitting,
162 multiview-fitting and 92 ablation objects; these sets need not be disjoint
and their complete membership has not been verified here. This study does not
reproduce that complete benchmark or the original multiview task.

`data/recruitment.json` separately retains the 320 Objaverse and 24 GSO
candidates, frozen ranks, admission flags, geometry-screen outcomes, metadata
shard hashes and historical UID exclusions. Failures are kept in this file.
To generate a candidate-acquisition plan instead of the admitted-object plan,
pass `--manifest data/recruitment.json` to the downloader. The assets themselves
are not bundled. Metadata hashes identify the sampling frame; candidate rows do
not turn the opportunistically available metadata shards into a random sample
of all Objaverse.

The GSO distribution was obtained from the official Kubric bucket. Its manifest
has 1,033 entries, whereas the GSO paper describes 1,030 objects; these refer to
the selected distribution and publication respectively. The selected entries'
CC BY-SA 4.0 metadata are retained. An asset's local 32-character GSO identifier
is a namespace hash, not an Objaverse UID. GSO archive hashes verify the original
payload; a separate mesh hash identifies the checked visual OBJ-to-GLB output.

The script emits `curl --proxy ''` commands and SHA-256 checks. Archives are not
automatically extracted or converted. Conversion uses the visual geometry,
preserves triangle-corner positions and UVs within 2e-6, and requires exact RGB
texture equality. It does not remesh, resize or repair the model. Collection
metadata alone do not establish identical rendering geometry after conversion.

No raw external model, texture, environment, checkpoint or third-party paper PDF is
included in this repository. Selected attributed reconstruction views appear in
our report. Licence and attribution requirements accompany the
original assets when users retrieve them themselves.
