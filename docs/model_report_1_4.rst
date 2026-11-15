Version 1.4 model and confidence review
=======================================

Version 1.4 selects the compact hybrid scorer reviewed on 16 November 2026.
It shares one CPU spectral encoder and one candidate ranker across supported
MS2 and joint MS1+MS2 inputs. Ranking weights and calibration were frozen
before the accuracy comparison. The 1.4.0 model bundle changes only the
version label and lossless compression of that frozen review artifact.

Accuracy
--------

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
----------------------------

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
---------------------------

The standard model is 8,751,792 bytes; the formula database is 18,405,871
bytes, and the common-fragment database is 2,496 bytes. The standard total is
27,160,159 bytes. The optional 169,787,342-byte compatibility model is
downloaded only when an older inference route is needed or explicitly
requested for offline use.

The 0.4.0 formula database is a lossless recompression of the prior indexed
database. Every decoded array has the same shape, dtype and bytes. A matched
64-spectrum full API check reproduced every candidate order and probability
with the recompressed database and model bundle. Standard downloads use
pinned SHA-256 digests and atomic replacement.

Scope
-----

Compact inference supports the documented singly charged, ppm-tolerance
conditions. Custom candidate budgets and unsupported conditions use the
optional compatibility route. Halogens remain enabled by default; set
``halogen=False`` only when prior knowledge rules them out. Formula generation
can still omit a true formula before ranking. The 1.4 release does not change
that candidate policy.
