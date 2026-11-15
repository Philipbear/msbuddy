Model update and confidence estimates
============================================

Version 1.4 uses one 749,644-parameter spectral encoder and one shared
candidate scorer for MS2 and joint MS1+MS2 annotation. The scorer combines
the spectrum representation, soft atom-count likelihoods, accurate fragment
mass patterns and physical formula evidence. It no longer needs numerical
ranker scores as inputs. Inference uses CPU ONNX Runtime and LightGBM;
PyTorch and CUDA are not required.

There is no hard biological-formula classifier or element-presence gate.
F, Cl, Br and I remain available for every query by default. Atom-count
predictions contribute ranking evidence without excluding candidate formulas.

The default model download is 8,751,792 bytes. The encoder itself is about
3.05 MB; the model download also contains the shared scorer, candidate
selector and probability calibration. Optional compatibility weights are
169,787,342 bytes and are fetched only when needed. The redesigned formula
and common databases total 18,408,367 bytes. Together, standard model and
database downloads total 27,160,159 bytes, compared with 32,365,088 bytes in
version 1.3 and about 633.7 MB in version 1.2.

See :doc:`model_report_1_4` for the current evaluation and :doc:`database_format`
for the lossless numerical format.

Optional acquisition metadata
-----------------------------

``MsbuddyConfig(ms_instr=...)`` accepts ``orbitrap``, ``qtof``, ``timstof`` and
``fticr``. It selects the documented mass tolerances. For mixed or manually
configured inputs, ``MetaFeature(instrument=..., collision_energy=...,
collision_energy_unit=...)`` can supply a per-query instrument and energy.
Per-query metadata do not override the engine's mass tolerances.

Energy units must be explicit: ``"30 nce"``, ``"20 eV"``, or a number with
``collision_energy_unit="nce"`` or ``"ev"``. Bare numeric units are not guessed.
Ranged or stepped energies are masked. Supported MGF keys include
``INSTRUMENT_TYPE``, ``COLLISION_ENERGY`` and ``COLLISION_ENERGY_EV``.
Missing metadata remain valid inputs; instrument type is not inferred from
mass tolerance.

Candidate budgets
-----------------

The recommended limits for MS1+MS2, MS1-only, MS2-only and mass-only inputs
are 50, 50, 25, 75, respectively. These defaults were selected using a separate
validation partition. Override them with
``MsbuddyConfig(max_candidates=350)`` or ``--max-candidates 350``.
Legacy model bundles retain a default limit of 350. The compact spectral
models use the recommended 25-candidate MS2 and 50-candidate joint operating
points. Other custom budgets use the established numerical rankers;
confidence at a different budget needs separate evaluation.

Halogen formulas are enabled by default with ``halogen=True``. Both
halogen-free and halogen-containing candidates are searched for every query.
Only when prior knowledge rules out F, Cl, Br and I should you use
``MsbuddyConfig(halogen=False)`` or ``--no-halogen``. The reported model
validation and candidate-budget selection used halogens enabled for all queries.
The default fluorine range is ``(0, 80)``. Explicit ``element_ranges`` and
``--f-max`` limits are respected. Count likelihoods and chemical patterns are
soft ranking inputs; no PFAS or halogen classification gate removes candidates.
Formula generation still depends on the configured element bounds and fragment/neutral-loss
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

Calibration for the compact models and the standard isotope-only and
mass-only compatibility modes uses a positive query-dependent temperature and
an absent-answer state. The winning formula's precursor-mass fit and observed
isotope count help adjust confidence. The positive temperature preserves the
candidate ranking. Context inputs are bounded after standardization to limit
extrapolation. Calibration is fitted on a separate formula-disjoint role.

A per-query ``estimated_fdr <= 0.05`` cutoff accepts top-1 probabilities of at
least 0.95. This is different from accepting the largest pool whose *average*
predicted error is 0.05: that pool can include less-confident individual calls.
The latter operating point can be mildly optimistic under precursor mass
error, particularly with missing acquisition metadata. Validate pooled
reporting, halogen-rich samples and unusual acquisition conditions separately.
Neither operating point provides a distribution-free FDR-control guarantee.
For version 1.4, the nominal pooled 5% rule exceeded 5% observed error in
the external halogen MS2 subgroup (3 errors among 34 accepted). The missing
true-formula probability was also underestimated in external MS2 data. These
estimates should not be treated as validated 5% FDR control across sources;
see :doc:`model_report_1_4` for the complete denominators and other modes.

The ``normalized_estimated_prob`` output column is retained for compatibility.
It now preserves probability mass for an absent answer, rather than forcing
the reported candidates to sum to one. Independent legacy probabilities are
scaled down if their sum exceeds one, but weak probabilities are not inflated.

Compatibility
-------------

The compact models are validated for singly charged ppm inputs below 1500 m/z,
supported monomer/multimer adducts, and precursor/fragment tolerances of
5/10 ppm or 10/20 ppm. Known Orbitrap metadata use 5/10 ppm; QTOF/timsTOF use
10/20 ppm. The validated preprocessing uses four maximum isotope peaks,
0.02 Da isotope bins, 1% denoising and six peaks per 50 Da.

For other settings, the established numerical rankers remain available.
``MsbuddyConfig(use_spectral=False)`` or ``--no-spectral`` selects them
explicitly. Absolute-tolerance, multiply charged and other adduct inputs use
the optional legacy compatibility models as before.
Chemically equivalent adduct spellings share the same model routing. Custom legacy model files
remain supported through ``data_files`` or ``--ml-model``.

Standard model downloads are atomic and verified against the release's
SHA-256 digest. Existing custom model files are loaded without applying the
standard asset digest. Partial ``update_config`` calls preserve unspecified
settings and retain loaded models unless their data paths change.

The original method is described in
`BUDDY <https://doi.org/10.1038/s41592-023-01850-x>`_, including the published
Supplementary Note 10 isotope simulation.
The compact spectral representation follows the large-receptive-field and
auxiliary-task approach of
`FIDDLE <https://doi.org/10.1038/s41467-025-66060-9>`_.
