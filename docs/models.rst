Model settings and confidence
=============================

Version 2.0
-----------

``msbuddy 2.0`` replaces the separate mode-gated rankers with one unified
four-stage CPU engine packaged as a single checksum-verified ``format_version=5``
(backward-compatible with ``format_version=4``) ``.joblib`` bundle
(``ml_v2.0.0.joblib``, **21.4 MB**, trained on public libraries only; CPU ONNX
Runtime + LightGBM; PyTorch and CUDA are not required; **39.8 MB** total with
the 18.4 MB lossless ``.msbdb`` formula databases):

1. **Stage 0 (Exact-Mass Spectral Transformer + Isobaric Contrastive Alignment,
   ONNX)** predicts element-count distributions (continuous ``C``/``H`` and
   zero-inflated hurdle negative binomials over ``Br, Cl, F, I, K, N, Na, O, P, S``),
   global properties, and a 128-d L2-normalized contrastive spectrum embedding
   ``z_spec`` from top-64 exact-mass MS/MS peaks and 60-dimensional query metadata.
2. **Stage 1–2 (Compiled ``nogil`` Numba Engine + Batched ``FormulaEncoderV3``)**
   streams ``formula_db_v0.4.0.msbdb`` directly into contiguous ``int16`` arrays
   (~520 MB peak RSS), performs fast two-pass candidate generation and
   contrastive shortlisting (protecting physics-only candidates), assigns exact
   subformulas/neutral losses with 48-dimensional MS/MS physics (internal
   MS1–MS2 self-recalibration, pool-IDF discriminative peaks, subformula-nesting
   DAG support, and heteroatom/adduct fragment balance), and computes optional
   MS1 isotope pattern features across a single shared-memory thread pool when
   ``parallel=True``.
3. **Stage 3–4 (FiLM-Conditioned Hybrid Set Reranker + Absent-Mass Calibration)**
   combines a permutation-equivariant ``PureSetRerankerV3`` (ONNX, FiLM-conditioned
   on the 97-d physical spectrum context with an explicit ``[ABSENT]`` token) and
   ``LightGBM`` LambdaRank trained with scaffold-clustered 5-fold OOF stacking,
   calibrated with temperature and absent-bias scaling ``(T, b_null)``.

Halogens (``F, Cl, Br, I``) remain enabled by default with zero compound-class
or PFAS-specific heuristics. Standard instrument tolerances remain
``5 / 10 ppm`` (Orbitrap), ``10 / 20 ppm`` (Q-TOF / timsTOF), and ``2 / 5 ppm``
(FTICR). Queries without MS/MS spectra (MS1-only or mass-only), multiply-charged
precursors, or runs configured with ``use_spectral=False`` lazily load the
compact compatibility bundle (``ml_compatibility_v2.0.1.joblib``, **3.23 MB**,
bringing the total download across all modes to **43.1 MB**). MS1-only and
precursor mass-only queries run through the compiled ``ms1_v2`` engine using
shared-memory ``FastFormulaDB`` and 12-d analytical Numba isotope features
(including fine-structure M+1/M+2 shifts) with calibrated ``[ABSENT]``
posteriors, arbitrary charge/tolerance/adduct support, and opt-in
``adduct="auto"``.

Opt-in multi-adduct disambiguation (``adduct="auto"``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Known-adduct annotation remains the default. When polarity is known but the
adduct form is unknown, passing ``adduct="auto"`` expands each query across
common positive adducts (``[M+H]+``, ``[M+Na]+``, ``[M+NH4]+``, ``[M+K]+``,
``[M-H2O+H]+``, ``[2M+H]+``, ``[2M+Na]+``) or negative adducts (``[M-H]-``,
``[M+Cl]-``, ``[M+CH2O2-H]-``, ``[M+C2H4O2-H]-``, ``[2M-H]-``). Fixed Bayesian
adduct prevalence log-priors are added during ``adduct="auto"`` scoring only to
resolve MS/MS-isobaric adduct aliases (such as ``[M+H]+`` of ``M`` versus
``[M'-H2O+H]+`` of ``M + H2O``) before joint softmax normalization against one
shared ``[ABSENT]`` anchor. The top-ranked adduct is stored on
``MetaFeature.adduct``. Known-adduct queries do not apply adduct priors and
preserve identical posterior probabilities.

There is no hard biological-formula classifier or element-presence gate.
F, Cl, Br and I remain available for every query by default. Atom-count
predictions contribute ranking evidence without excluding candidate formulas.

See :ref:`update-2-0`, :ref:`update-1-4`, and :ref:`update-1-3` for version
updates and model sizes, and :doc:`database_format` for the lossless format.

Optional acquisition metadata
-----------------------------

``MsbuddyConfig(ms_instr=...)`` accepts ``orbitrap``, ``tof`` and ``fticr``.
Both ``qtof`` and ``timstof`` are accepted as aliases for ``tof``. It selects
the documented mass tolerances. For mixed or manually
configured inputs, ``MetaFeature(instrument=..., collision_energy=...,
collision_energy_unit=...)`` can supply a per-query instrument and energy.
Per-query metadata do not override the engine's mass tolerances.

Energy units must be explicit: ``"30 nce"``, ``"20 eV"``, or a number with
``collision_energy_unit="nce"`` or ``"ev"``. Bare numeric units are not guessed.
Ranged or stepped energies are masked. Supported MGF keys include
``INSTRUMENT_TYPE``, ``COLLISION_ENERGY`` and ``COLLISION_ENERGY_EV``.
Missing metadata remain valid inputs; instrument type is not inferred from
mass tolerance. QTOF and timsTOF metadata both use the same TOF model input.
The existing model was trained with separate indicators; the unified TOF input
uses the former QTOF indicator. Its predictions for timsTOF inputs can therefore
change, and the original per-instrument evaluation does not validate this change.

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

Use ``engine.estimate_batch_fdr(fdr_threshold=0.05)`` after annotating the full
collection to perform this pooling. The returned ``q_values`` align with input
spectra and select the largest complete confidence prefix whose mean posterior
error is at most the threshold. Equal-confidence annotations are selected
together; spectra with no candidates have no q-value and are never selected.
``get_summary()`` and the command-line summary include ``batch_q_value`` and
``posterior_error_probability``. Command-line q-values pool all processing
batches. A q-value describes the selected collection, not the error probability
of that individual annotation. Adding or removing queries can change q-values.

Calibration for the compact models and the standard isotope-only and
mass-only compatibility modes uses a positive query-dependent temperature and
an absent-answer state. The winning formula's precursor-mass fit and observed
isotope count help adjust confidence. The positive temperature preserves the
candidate ranking. Context inputs are bounded after standardization to limit
extrapolation. Calibration is fitted on a separate formula-disjoint role.

A per-query ``estimated_fdr <= 0.05`` cutoff accepts top-1 probabilities of at
least 0.95. This is different from accepting the largest pool whose *average*
predicted error is 0.05: that pool can include less-confident individual calls.
Neither operating point provides a distribution-free FDR-control guarantee.
Validate pooled reporting, halogen-rich samples and unusual acquisition
conditions separately. Version 1.4 confidence has not been validated as
general 5% FDR control across sources; see :ref:`update-1-4`.

The ``normalized_estimated_prob`` output column is retained for compatibility.
It now preserves probability mass for an absent answer, rather than forcing
the reported candidates to sum to one. Independent legacy probabilities are
scaled down if their sum exceeds one, but weak probabilities are not inflated.

Compatibility
-------------

The compact models are validated for singly charged ppm inputs below 1500 m/z,
supported monomer/multimer adducts, and precursor/fragment tolerances of
5/10 ppm or 10/20 ppm. Known Orbitrap metadata use 5/10 ppm; TOF metadata use
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
