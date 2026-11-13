Model update and confidence estimates
============================================

Version 1.1 provides four revised evidence-mode ranking models, explicit
elemental/adduct features, a learned candidate selector, and probability
calibration with an absent-answer state. There is no separate hard
biological-formula classifier. Halogen formulas have explicit F, Cl, Br and I
features.

Candidate budgets
-----------------

The recommended limits for MS1+MS2, MS1-only, MS2-only and mass-only inputs
are 50, 50, 25, 75, respectively. These defaults were selected using a separate
validation partition. Override them with
``MsbuddyConfig(max_candidates=350)`` or ``--max-candidates 350``.
Legacy model bundles retain a default limit of 350. Calibration was validated
for budgets up to 350; larger custom budgets need separate evaluation.

Enable halogen formulas explicitly with ``MsbuddyConfig(halogen=True)`` or
``--halogen``. The historical default remains ``False``. Formula generation
still depends on the configured element bounds and fragment/neutral-loss
database queries; a ranker cannot recover an excluded formula.

Probability and FDR interpretation
----------------------------------------

Present formulas share probability with an explicit absent-answer state.
The probabilities therefore need not sum to one across the reported formulas.
A lone weak candidate no longer becomes certain through normalization.

``estimated_fdr`` at rank N is the expected false fraction among that query's
top N formulas: one minus the sum of their calibrated probabilities divided
by N. At rank one it is a posterior error probability. Across queries, sort
top-1 calls by this error probability and average accepted values to estimate
the pooled false fraction. These estimates depend on calibration and domain
match; they are not distribution-free FDR guarantees.

Calibration is assessed for pooled top-1 reporting. Adduct and acquisition
subgroups can behave differently. Diagnostic multimers remain overconfident;
validate multimer-rich analyses separately and report that subgroup separately.

The ``normalized_estimated_prob`` output column is retained for compatibility.
It now preserves probability mass for an absent answer, rather than forcing
the reported candidates to sum to one. Independent legacy probabilities are
scaled down if their sum exceeds one, but weak probabilities are not inflated.

Compatibility
-------------

The new models are validated for singly charged ppm inputs, including
the monomer and multimer adducts recorded in the bundle. Absolute-tolerance,
multiply charged and other adduct inputs use the bundled legacy models.
Chemically equivalent adduct spellings share the same model routing. Custom legacy model files
remain supported through ``data_files`` or ``--ml-model``.

Standard model downloads are atomic and verified against the release's
SHA-256 digest. Existing custom model files are loaded without applying the
standard asset digest. Partial ``update_config`` calls preserve unspecified
settings and retain loaded models unless their data paths change.

The complete validation report and aggregate metrics accompany the
`1.1.0 release <https://github.com/Philipbear/msbuddy/releases/tag/v1.1.0>`_.
The original method is described in
`BUDDY <https://doi.org/10.1038/s41592-023-01850-x>`_, including the published
Supplementary Note 10 isotope simulation.
