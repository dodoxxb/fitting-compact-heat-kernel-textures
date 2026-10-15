# Mechanism-completion release, 15 October 2026

The 49-page report retains eight main pages and the previous scientific
appendices, then adds Appendix H. There are32 references:30 paper references
have matched DOI metadata (some use arXiv DOIs), one paper uses its official
PMLR proceedings entry, and RTXNTC uses the official software repository.
No generated disclosure or archive manifest is inserted into the manuscript.

This extension includes96 conditional solves,60 actual rendered fields at
three resolutions (300 field views and60 reference views),18 matched-node
off-grid operator comparisons, numerical identity checks, protocols, raw scalar
records, and portable sampling/rendering tools. Old experiments remain intact.
These new diagnostics reuse four texture objects and two operator meshes;
they are not new independent-object benchmark coverage.

Reproduction levels remain distinct: archive integrity; CPU scalar arithmetic;
explicit-input mathematical/fitting tools; and GPU replay requiring separately
acquired assets and the documented runtime. The exported CPU sampler reproduced
an actual two-arm case. The exported renderer reproduced one actual object,
three fractions and five128-square views with identical PSNR on this host.
This is not a clean-machine GPU installation or a rerun of all original training.

The generic renderer exposed an unnecessary eager import of the historical
object registry in the shared helper. Its import is now lazy, so an explicit
mesh/checkpoint render does not require that private historical registry.

Historical execution sources are retained for inspection in
experiment_sources_20261015. The original dated release documentation is
preserved separately; current claims are described by README.md and the report.
