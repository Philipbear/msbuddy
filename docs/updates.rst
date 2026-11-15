Version updates and validation
==============================

The sections below keep detailed release comparisons and limitations together.
For a short change list, see the `changelog
<https://github.com/Philipbear/msbuddy/blob/main/changelog.md>`_.
Jump to :ref:`update-1-4` or :ref:`update-1-3`.

.. _update-1-4:

Version 1.4.0 (16 November 2026)
--------------------------------

Version 1.4 selects the compact hybrid scorer reviewed on 16 November 2026.
It shares one CPU spectral encoder and one candidate ranker across supported
MS2 and joint MS1+MS2 inputs. Ranking weights and calibration were frozen
before the accuracy comparison. The 1.4.0 model bundle changes only the
version label and lossless compression of that frozen review artifact.

Accuracy
^^^^^^^^

The fresh accuracy comparison contains 1,696 formulas. Values below use all
queries in each condition and the same candidate policy for 1.3 and 1.4.
Joint conditions use **simulated** isotope intensities; they do not establish
performance with measured MS1 envelopes.

.. list-table:: Fresh accuracy comparison
   :header-rows: 1

   * - Condition
     - Version 1.3 top choice
     - Version 1.4 top choice
   * - Native precursor, MS2
     - 1,225/1,696 (72.23%)
     - 1,241/1,696 (73.17%)
   * - Native precursor, simulated MS1 + MS2
     - 1,447/1,696 (85.32%)
     - 1,461/1,696 (86.14%)
   * - Perturbed precursor, MS2
     - 1,146/1,696 (67.57%)
     - 1,187/1,696 (69.99%)

An independent 1,507-formula acquisition-family diagnostic was substantially
harder: native MS2 top choice was 900/1,507 (59.72%). With simulated MS1
evidence it was 1,133/1,507 (75.18%). This is why the fresh comparison must
not be read as a universal accuracy level. High-fluorine results remain
sparsely supported in several held-out conditions.

Confidence and estimated FDR
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The calibrator gives each candidate a posterior probability and reserves
probability for a correct formula absent from the candidate list. At rank
one, ``estimated_fdr`` equals one minus the top candidate probability. The
per-query 5% cutoff accepts probabilities of at least 0.95. A separate
pooled 5% rule accepts the largest complete confidence prefix whose average
predicted error is at most 5%; it may include less-confident individual calls.
Neither rule has a distribution-free FDR guarantee.

.. list-table:: Nominal 5% selections, acquisition metadata supplied
   :header-rows: 1

   * - Evaluation and evidence
     - Individual 5%: errors / accepted
     - Pooled 5%: errors / accepted
   * - Fresh comparison, MS2
     - 1/217 (0.46%)
     - 8/402 (1.99%)
   * - Formula-novel acquisition-family diagnostic, MS2
     - 3/80 (3.75%)
     - 6/153 (3.92%)
   * - Same diagnostic, halogen MS2 subgroup
     - 1/18 (5.56%)
     - 3/34 (8.82%)
   * - Same diagnostic, simulated MS1 + MS2
     - 3/237 (1.27%)
     - 5/435 (1.15%)

The all-query selections can appear favorable while chemically important
subgroups fail. In the formula-novel diagnostic with MS2 alone, the true
formula was absent in 8.29% of queries, versus 2.84% mean predicted absent
probability. The prespecified confidence audit passed 58 of 96 empirical
checks and 14 of 48 simultaneous uncertainty checks. Version 1.4.0 therefore
**does not claim validated 5% FDR control across sources**, particularly for
halogen-rich MS2 data. Treat its confidence values as model-based estimates
and validate error rates on data matching the intended acquisition and
chemical domain.

Download size and integrity
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The standard model is 8,751,792 bytes; the formula database is 18,405,871
bytes, and the common-fragment database is 2,496 bytes. The standard total is
27,160,159 bytes. The optional 169,787,342-byte compatibility model is
downloaded only when an older inference route is needed or explicitly
requested for offline use. The 749,644-parameter encoder itself is about
3.05 MB; the bundle also contains the scorer, candidate selector and
probability calibration.

The 0.4.0 formula database is a lossless recompression of the prior indexed
database. Every decoded array has the same shape, dtype and bytes. A matched
64-spectrum full API check reproduced every candidate order and probability
with the recompressed database and model bundle. Standard downloads use
pinned SHA-256 digests and atomic replacement.

In one 64-spectrum matched annotation run, the 0.3.0 database took 17.52
seconds and the 0.4.0 database took 16.41 seconds. These runs include other
inference work and do not establish a speedup. Random indexed reads can take
longer with LZMA.

Scope
^^^^^

Compact inference supports the documented singly charged, ppm-tolerance
conditions. Custom candidate budgets and unsupported conditions use the
optional compatibility route. Halogens remain enabled by default; set
``halogen=False`` only when prior knowledge rules them out. Formula generation
can still omit a true formula before ranking. The 1.4 release does not change
that candidate policy.

.. _update-1-3:

Version 1.3.0 (15 November 2026)
--------------------------------

Version 1.3 replaces the standard MS2 and joint scoring ensemble with one
shared candidate scorer and a compact spectral encoder. It improves the
prespecified general and fluorinated-formula evaluations while substantially
reducing the default downloads. A separate external PFAS challenge identifies
a remaining MS2-only ranking weakness, reported below.

Model and storage design
^^^^^^^^^^^^^^^^^^^^^^^^

The 749,644-parameter encoder combines the existing spectral representation
with a small chemical/count head. The shared 435-input tree scorer uses
physical candidate evidence, the spectrum representation, precise fragment
mass patterns, count discrepancies and count likelihoods. Fluorine has a
soft zero-versus-positive count distribution with unbounded positive support.
There is no binary biological-formula or PFAS gate that removes candidates.

The selected model retains a fixed spectral backbone. Structural fingerprint
and motif auxiliary supervision, gentle backbone fine-tuning, removal of
teacher supervision and a small neural candidate scorer were evaluated on
development data. None improved its overall selection score. The selected
scorer distills the earlier model's negative-candidate ordering alongside
correct-formula supervision; earlier model scores are unnecessary at inference.
The fingerprint prediction head is excluded from runtime.

.. list-table::
   :header-rows: 1

   * - Asset
     - Version 1.2 bytes
     - Version 1.3 bytes
   * - Default model
     - 197158291
     - 9371942
   * - Formula database
     - 436457267
     - 22990650
   * - Common database
     - 68290
     - 2496
   * - Default total
     - 633683848
     - 32365088
   * - Optional compatibility weights
     - included above
     - 169787342

Compatibility ranking parameters remain unchanged. Isotope-only and
mass-only confidence is refitted for the wider candidate pool. Optional
weights are fetched when a compatibility path is needed and can be prefetched
with ``download_data(include_compatibility=True)``. Default inference requires
CPU ONNX Runtime and LightGBM, without PyTorch, CUDA or RDKit.

The database redesign preserves every formula row, duplicate, dtype, index
and mass bit. All 4,194,390 rows and every stored array were checked against
the legacy files. An additional 8,644 searches preserved candidate order,
duplicates, masses and charges. See :doc:`database_format` for the format.

In five fresh-process trials per format on one machine, median peak process
memory during database loading decreased from 535 MB to 223 MB. Median loader
time increased from 0.067 to 0.198 seconds. These measurements exclude model
loading and candidate generation; operating-system file caches were not
controlled. Individual indexed lookups were slightly slower with lazy
decoding. These are loader and lookup measurements, not an end-to-end speed
claim.

Evaluation design
^^^^^^^^^^^^^^^^^

The ranking model was selected on development data. Confidence uses a
separate formula-disjoint calibration role with formula-group internal
validation. Ranking and confidence were frozen before final outcomes were
opened. Final predictions use the release CPU inference graphs.

The primary cohort contains 4,000 distinct formulas, including 2,164 formulas
with halogens. A separate fluorinated challenge contains 153 further distinct
formulas. Formula and structure groups were excluded from current fitting
roles and previously opened research cohorts. Historical pretraining overlap
cannot be fully reconstructed. These cohorts do not estimate the chemical
prevalence of routine untargeted samples.

Formula/structure consistency, supported adducts, precursor mass consistency
and spectral information were checked before role assignment. Fitting-label
quality checks use truth-compatible fragments. The final evaluation retains
all frozen queries, including weak spectra and candidate-generation failures.

Both versions receive the same spectra, mass tolerances and available metadata.
The released 1.2 default uses a fluorine ceiling of 20; version 1.3 uses 80 for
every query. Both allow halogens throughout. MS2 and joint candidate budgets
remain 25 and 50; isotope-only and mass-only budgets remain 50 and 75.

The measured-mass condition uses the recorded precursor and measured MS2.
The perturbed condition replaces precursor error using a zero-mean Gaussian
with standard deviation equal to tolerance/5, truncated at the tolerance.
Joint evaluations use simulated MS1 evidence, not measured isotope spectra.
Simulation uses the charged composition, including multimer and net adduct,
with the original method's relative isotope-intensity errors and correlated
signs. The stress condition increases intensity noise 1.5-fold and retains
two, three or four isotope peaks. No extra isotope m/z noise is introduced.

General cohort: 4,000 formulas
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Accuracy is unconditional top-1 accuracy, including absent correct formulas.
Intervals are paired 95% intervals from 4,096 formula-block bootstrap samples.
All eight primary checks improved in point accuracy and passed the predefined
0.5 percentage-point noninferiority margin. All eight gain intervals are
above zero on this cohort.

.. list-table::
   :header-rows: 1

   * - Condition
     - 1.2 (%)
     - 1.3 (%)
     - Gain (pp)
     - 95% interval (pp)
   * - Measured mass; MS2; metadata known
     - 86.55
     - 89.63
     - 3.08
     - 2.33 to 3.87
   * - Measured mass; MS2; metadata missing
     - 84.83
     - 88.88
     - 4.05
     - 3.28 to 4.88
   * - Measured mass; joint; metadata known
     - 94.38
     - 94.95
     - 0.58
     - 0.05 to 1.10
   * - Measured mass; joint; metadata missing
     - 93.95
     - 94.68
     - 0.73
     - 0.20 to 1.23
   * - Perturbed mass; MS2; metadata known
     - 79.00
     - 84.13
     - 5.13
     - 4.23 to 6.00
   * - Perturbed mass; MS2; metadata missing
     - 78.58
     - 83.23
     - 4.65
     - 3.78 to 5.58
   * - Perturbed mass; joint; metadata known
     - 90.63
     - 92.90
     - 2.28
     - 1.63 to 2.93
   * - Perturbed mass; joint; metadata missing
     - 90.30
     - 92.45
     - 2.15
     - 1.50 to 2.80

The isotope-stress condition also improved: with measured precursor mass
and known metadata, joint accuracy increased from 90.13% to 92.68%; with
perturbed precursor mass it increased from 84.73% to 88.63%. These results
still use simulated isotope evidence.

Isotope-only and mass-only ranking accuracy was exactly unchanged on this
general cohort. Their ranking parameters were retained, and the wider
fluorine bound did not change these top-1 decisions.

Separate fluorinated challenge: 153 formulas
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This cohort is separate from the 4,000-formula primary evaluation. Most of
its formulas contain a small number of fluorines; it does not establish
performance for long perfluorinated chains.

.. list-table::
   :header-rows: 1

   * - Condition
     - 1.2 (%)
     - 1.3 (%)
     - Gain interval (pp)
   * - Measured mass; MS2; metadata known
     - 65.36
     - 76.47
     - 5.88 to 16.99
   * - Measured mass; MS2; metadata missing
     - 62.09
     - 75.82
     - 7.84 to 19.61
   * - Measured mass; joint; metadata known
     - 81.05
     - 87.58
     - 2.61 to 11.11
   * - Measured mass; joint; metadata missing
     - 79.08
     - 86.93
     - 3.92 to 12.42
   * - Perturbed mass; MS2; metadata known
     - 58.82
     - 69.93
     - 5.23 to 16.99
   * - Perturbed mass; MS2; metadata missing
     - 57.52
     - 67.97
     - 5.23 to 15.69
   * - Perturbed mass; joint; metadata known
     - 73.86
     - 81.05
     - 1.31 to 12.42
   * - Perturbed mass; joint; metadata missing
     - 71.24
     - 77.12
     - 0.65 to 11.11

External PFAS standards
^^^^^^^^^^^^^^^^^^^^^^^

The external measured-MS2 challenge comes from the
`PFAS standards record <https://doi.org/10.5281/zenodo.15852752>`_. It was used
for evaluation only. After prespecified quality and input checks it contains
157 spectra representing 26 formulas. Nine formulas overlap current fitting
roles; the disjoint subset contains 108 spectra and 17 formulas. Acquisition
replicates are grouped by formula in uncertainty estimates. Instrument family
is unknown, and a tolerance profile is supplied explicitly.

The following table reports the disjoint subset. MS2-only top-1 accuracy is
lower in this small domain-specific cohort; the intervals are wide. The
joint results improve with simulated MS1 evidence. These findings limit a
claim of uniform PFAS improvement.

.. list-table::
   :header-rows: 1

   * - Condition
     - 1.2 (%)
     - 1.3 (%)
     - Gain interval (pp)
   * - Measured mass; MS2; metadata known
     - 80.56
     - 75.00
     - -14.70 to 7.19
   * - Measured mass; MS2; metadata missing
     - 81.48
     - 72.22
     - -20.95 to 5.42
   * - Measured mass; joint; metadata known
     - 92.59
     - 97.22
     - -1.22 to 14.89
   * - Measured mass; joint; metadata missing
     - 94.44
     - 98.15
     - -1.94 to 13.64
   * - Perturbed mass; MS2; metadata known
     - 65.74
     - 58.33
     - -17.65 to 6.67
   * - Perturbed mass; MS2; metadata missing
     - 63.89
     - 55.56
     - -17.90 to 4.72
   * - Perturbed mass; joint; metadata known
     - 85.19
     - 91.67
     - 0.00 to 16.37
   * - Perturbed mass; joint; metadata missing
     - 87.96
     - 92.59
     - -1.38 to 13.83

The disjoint subset includes one formula above 20 fluorines, measured in
four acquisitions. The old default excludes its correct formula; version 1.3
retrieves and ranks it first in all four measured-mass MS2 acquisitions.
One formula is insufficient to estimate general high-fluorine accuracy.

For measured-mass MS2 with metadata, the 5% per-query error cutoff retains
32 disjoint acquisitions with no observed errors, compared with 26 for the
old model. Correlated acquisitions and the small number of formulas limit
interpretation of this yield comparison. Zero observed errors does not imply
zero future error risk.

Confidence and FDR
^^^^^^^^^^^^^^^^^^

Candidate formulas share probability with an explicit absent-answer state.
A positive context-dependent temperature preserves ranking. The calibration
uses only observable score, mass, isotope and explanation evidence. Context
values are clipped after standardization to limit extreme extrapolation.
All four evidence-mode optimizers converged and passed finite-difference
gradient checks with maximum errors below 1e-9.

At rank one, ``estimated_fdr <= 0.05`` means posterior probability at least
0.95. It differs from selecting a larger pool with average predicted error
at most 5%. The table reports the actual per-query API rule on the 4,000-formula
cohort. These are empirical checks, not distribution-free FDR guarantees.

.. list-table::
   :header-rows: 1

   * - Condition
     - Accepted
     - Errors
     - Observed FDP (%)
     - Mean predicted error (%)
   * - Measured mass; MS2; metadata known
     - 2519
     - 11
     - 0.44
     - 0.86
   * - Measured mass; MS2; metadata missing
     - 2452
     - 10
     - 0.41
     - 0.93
   * - Measured mass; joint; metadata known
     - 3184
     - 9
     - 0.28
     - 0.69
   * - Measured mass; joint; metadata missing
     - 3135
     - 9
     - 0.29
     - 0.71
   * - Perturbed mass; MS2; metadata known
     - 2138
     - 24
     - 1.12
     - 0.99
   * - Perturbed mass; MS2; metadata missing
     - 2110
     - 22
     - 1.04
     - 1.03
   * - Perturbed mass; joint; metadata known
     - 2940
     - 17
     - 0.58
     - 0.90
   * - Perturbed mass; joint; metadata missing
     - 2894
     - 19
     - 0.66
     - 0.90

Pooled expected-error selection, isotope stress, compatibility modes and
the external domain were evaluated separately. Calibration remains dependent
on domain match. A wider candidate pool, new acquisition conditions or rare
chemistries can change the error distribution.

Development controls and limitations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The selected count model's eight-condition development mean was 90.902%.
Removing teacher ordering reduced it to 90.225%. Structural multitask
supervision and gentle encoder fine-tuning did not improve the selected
overall development result. A 96,129-parameter neural candidate scorer was
only 0.39 MB but lost accuracy and was rejected. This supports retaining one
small tree scorer rather than assuming a predicted fingerprint makes formula
identification unambiguous.

The remaining limitations are candidate-generation misses, the external
PFAS MS2 ranking result, limited independent high-fluorine examples, and the
use of simulated MS1 in the primary joint evaluation. On a small additional
measured-MS1/MS2 diagnostic of 42 spectra (35 formulas, no halogens), both
MS2-only and joint accuracy increased from 40/42 to 41/42. That diagnostic
has historical overlap limitations and does not establish performance for
new chemistries or measured isotope evidence generally. No matched new
FIDDLE benchmark is reported here; these release comparisons are against
msbuddy 1.2.

Runtime and software checks
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The CPU comparison uses the same 64 calibration queries in the same order,
with one inference thread on an AMD EPYC 9J14 Linux server. Three fresh-process
repeats alternate old/new order. The first two queries warm imports and JIT
compilation and are excluded from the warm-query statistics. Values below
are medians of the three repeat summaries. Operating-system file caches
and unrelated machine activity were not controlled.

.. list-table::
   :header-rows: 1

   * - Measurement
     - 1.2
     - 1.3
   * - MS2 median warm query (ms)
     - 63.3
     - 78.9
   * - MS2 p95 warm query (ms)
     - 1225.8
     - 1279.7
   * - MS2 total for 64 warm queries (s)
     - 23.06
     - 24.83
   * - MS2 initialization, excluding first-query JIT (s)
     - 5.46
     - 1.33
   * - MS2 process peak memory (MB)
     - 1588
     - 491
   * - Joint median warm query (ms)
     - 93.1
     - 121.9
   * - Joint p95 warm query (ms)
     - 1431.7
     - 1494.1
   * - Joint total for 64 warm queries (s)
     - 25.96
     - 28.34
   * - Joint initialization, excluding first-query JIT (s)
     - 5.53
     - 1.33
   * - Joint process peak memory (MB)
     - 1588
     - 501

The new default uses substantially less memory and initializes faster. Warm
queries are modestly slower because formula rows are decoded lazily and
additional accurate-mass/count evidence is computed. Both versions also
pay a first-query JIT cost of roughly 15–17 seconds in these fresh processes;
it is separate from initialization and warm-query latency. This release
makes no end-to-end speedup claim.

Validation passed 145 regression tests on each of Python 3.8, 3.9, 3.10,
3.11, 3.12 and 3.13 in isolated Linux environments. Each environment imported
the built wheel outside the source tree, decoded and checked every database
array SHA-256, and executed all four evidence modes with the actual CPU
assets offline. This covers NumPy 1.x and 2.x and compatible ONNX Runtime
versions. A separate macOS ARM64 wheel check passed all four modes.

Production checks reproduced original-database candidate arrays and physical
features on 128 fixed queries, and reproduced 3,742 candidate scores using an
independent CPU implementation. Standard spectral scoring never loaded the
optional compatibility weights. Serial/spawned execution, six compatibility
controls and 32 CLI/API controls passed; the latter produced identical
exported TSV bytes across all four evidence modes. Hosted CI is not claimed
as the basis of these completed independent checks.

Every database array and block was checked for bit identity, including all
4,194,390 formula rows. An additional 8,644 query comparisons preserved
formula order, duplicate entries, mass bits and charge. Model compression
preserves all score parameters, encoder bytes and candidate-selector values.
Compatibility ranking serializations remain identical while numerical-only
confidence is updated independently. Wheel/source packaging, strict package
metadata, documentation and public-content audits are included in the
release checks. Public download and installation verification are recorded
with the publication receipt.
