Candidate recall and ranking experiments
========================================

Run these research commands from a source checkout. They trace the actual
annotation functions, leave reference libraries read-only, and do not replace
released models. MassDesk search and propagated neutral-loss windows are
explicit experiments; ordinary annotation still uses database fragment search.

Prepare labelled inputs
-----------------------

Use Python 3.10 or newer for MassDesk. The research environment needs
``lightgbm``, ``torch``, ``pyarrow``, and ``pyteomics`` alongside the project
dependencies. MassDesk can be installed or loaded from a source checkout.

The sampler reads native NAS CASMI 2026 Parquet. It selects structures by a
deterministic hash within source library, instrument family, polarity, and
precursor-m/z groups (<400, 400-800, >800). Peak arrays are materialized only
from selected row groups. This stratified pilot does not estimate average
accuracy across the original library::

   python -m benchmarks.library_sample \
     --parquet /path/to/reference/ms2_libraries/kaggle_casmi2026/train.parquet \
     --libraries mona massbank pluskal_ms2 enveda-180 \
     --per-stratum 4 --seed 20261113 \
     --output results/recall/spectra.jsonl

The JSONL preserves source path and row, library, original instrument, formula,
structure group, adduct, collision energies, and peak arrays. Its adjacent
``.sampling.json`` records source size, settings, and stratum populations.
Original contributor attributions and usage terms still apply.

Other inputs can use the same schema: unique ``identifier``, neutral reference
``formula``, ``precursor_mz``, ``adduct``, ``polarity``, ``instrument``, ``mzs``,
and ``intensities``. Optional fields include ``collision_energy_ev``,
``source_library``, ``structure_group``, and ``ms1`` containing ``mzs`` and
``intensities``. Invalid labels are reported, not silently rewritten.

Trace generation and selection
------------------------------

Start with the existing workflow and pretrained ranking::

   python -m benchmarks.candidate_audit \
     --input results/recall/spectra.jsonl \
     --output results/recall/database \
     --backends database --export-features --timeout 30

Labels are used only by the observer and evaluator. They never constrain
generation or enter model inputs. The observed stages are:

* ``precursor_database`` and ``precursor_filtered``: parent retrieval and
  subsequent chemical restrictions.
* ``ms2_stitched`` and ``ms2_filtered``: precursor formulas reconstructed from
  fragment/loss pairs before and after chemical restrictions.
* ``merged``, ``valid``, and ``pre_pruning``: combined pools before the cutoff.
* ``retained`` and, when ranking is requested, ``annotated``: candidates that
  survive selection and subformula assignment.

Per-peak records distinguish a failed lookup from an unqueried partner. The
smaller mass is searched first; failure there skips the other side.
``truth_compatible_pairs`` describes returned compositions consistent with the
label and is never used for selection. Chemical-rule failures, precursor mass
mismatches, and multiply charged precursors are identified separately.

Outputs are ``audit.jsonl``, ``summary.json``, and ``summary.tsv``. Timeouts,
invalid inputs, and errors have distinct statuses. Stage recall counts completed,
applicable trials; operational accuracy must also count failed requests.
``--candidate-only`` skips expensive subformula assignment and ranking.
``--export-features`` saves every pre-pruning candidate with 16 cheap features:
12 element counts, precursor mass error, explained MS2 intensity, isotope
similarity, and precursor database membership. Full ranking additionally
exports BUDDY's 53-feature representation.

Parent retrieval can rescue missing MS2 candidates. Inspect ``ms2_stitched``
separately when assessing discovery of previously unreported formulas.

Compare fragment search
-----------------------

The adapter calls ``massdesk.core.decompose.decompose_mass`` on fragment ionic
mass and neutral-loss mass. Bounds use configuration and adduct, never reference
atom counts. It uses msbuddy's element masses for a controlled comparison,
includes the electron mass correction, and applies the existing nonnegative
fragment DBE restriction before the result cap::

   python -m benchmarks.candidate_audit \
     --input results/recall/spectra.jsonl \
     --output results/recall/search_comparison \
     --backends database fallback union de-novo \
     --massdesk-path /path/to/MassDesk \
     --candidate-only --max-results 1000 --peak-budget 3 --timeout 15

``fallback`` decomposes empty lookups only. It cannot recover a missing correct
formula when an incorrect database match already exists. ``union`` adds de novo
results for a fixed number of the highest-intensity processed peaks.
``de-novo`` replaces all fragment/loss lookups. Every mode still merges parent
database candidates: this is a fragment-search comparison, not total removal
of the database.

Decomposition uses a bounded cache. Result caps and truncated queries are
recorded; capped searches cannot establish exhaustive recall. A separate worker
enforces deadlines even if decomposition or subformula assignment stalls.
Loading and compilation are warmed before timed requests. Avoid concurrent
workloads when comparing runtimes. Timings include audit overhead and do not
constitute a FIDDLE speed benchmark.

Neutral-loss uncertainty
------------------------

A loss is the difference between two measured masses. The legacy window scales
MS2 ppm tolerance by loss mass, which can be too narrow for small losses even
when both ions satisfy their individual tolerances. The experimental option
``--propagate-loss-tolerance`` uses a conservative absolute bound::

   loss_tolerance_Da = precursor_tolerance_Da + fragment_tolerance_Da

This works with database and de novo search. The final precursor-mass check
is retained. Broader windows can increase candidate counts and runtime, so
compare identical inputs before changing production settings.

Replay the merge correction
---------------------------

The corrected merge preserves bottom-up explained intensity on formulas also
found by parent database search. Previously that evidence was lost before
pruning. Isolate its effect by replaying old and corrected scores on identical
exported pools::

   python -m benchmarks.pruning_ablation \
     --audit results/recall/database/audit.jsonl \
     --summary results/recall/database/summary.json \
     --output results/recall/pruning_ablation.json

The corrected replay must match recorded selection. It reports recoveries
and regressions in candidate retention, rather than final annotation accuracy.

Learning-to-rank controls
-------------------------

Compare a small LightGBM ranker, a numerical-feature/metadata neural control, and a
neural ranker with fragment and neutral-loss spectral channels. These are
feasibility baselines, not a FIDDLE reproduction::

   python -m benchmarks.train_ranker \
     --records results/recall/spectra.jsonl \
     --audit results/recall/database/audit.jsonl \
     --output results/recall/rankers \
     --candidate-pool retained --epochs 20

``retained`` uses the surviving candidates and all 53 BUDDY features.
``proposal`` uses the 16 cheap features before pruning, allowing experiments
on replacing the initial cutoff without enumerating all subformulas. Multiple
audit files can compare pools on the same split. Each spectrum/backend pair
must occur once.

Canonical formulas define deterministic 60/20/20 train/validation/test groups.
Normalization uses training data only; checkpoint selection uses validation
only. Negative sampling affects training, while test scoring uses entire
exported pools. Models, split manifest, learning curves, and metrics are saved
together. LightGBM runs separately from PyTorch to avoid native runtime conflicts.

Small pilots cannot establish a deep-learning advantage or calibrated FDR.
New scores are uncalibrated; pretrained models remain in place. Larger formula-
and source-held-out, calibration, and chimeric-spectrum evaluations are required
before adopting a new scorer.
