Signed delta-mass annotation
============================

``annotate_delta_mass`` enumerates signed elemental changes for a directed
edge: **target minus source**. For example, replacing a proton adduct with a
sodium adduct gives ``-H+Na``. Unlike a molecular ``Formula``, a ``DeltaFormula``
can contain negative atom counts, have negative mass, or be exactly zero.

``annotate_network_delta`` uses both precursor m/z values and their assigned
adducts to calculate the **neutral molecular** difference. It reports the
adduct change separately. Both functions work without downloading a model or
formula database.

An edge from modified cosine supplies a relationship worth examining; it does
not establish that the nodes are related by a reaction. The API returns
candidate net changes and possible explanations. It does not confirm a
biotransformation, its intermediates, or an ion-identity relationship, and its
ranks are not probabilities or FDR estimates.

A signed mass difference
------------------------

.. code-block:: python

   from msbuddy import DeltaFormula, annotate_delta_mass

   exchange = DeltaFormula({'H': -1, 'Na': 1})
   print(str(exchange))                 # -H+Na
   print(exchange.mass)                # 21.98194424977 Da

   matches = annotate_delta_mass(21.98194425, tolerance=0.001)
   print(matches[0].formula)            # -H+Na
   print(matches[0].delta.composition)  # {'H': -1, 'Na': 1}
   print(matches[0].hypotheses)         # a possible adduct exchange

The input is an atomic mass difference in Da. For ions with the **same signed
charge**, the atomic difference is ``abs(charge) * (target_mz - source_mz)``.
A raw m/z difference must not be interpreted as a neutral molecular difference
when the adducts, charges, or multimer counts differ. Use the network function
when those assignments are available.

``DeltaFormula.from_string('-H2O+O')`` combines signed formula terms and produces
``-H2``. A sign applies to the entire following term. ``str(delta)`` formats
every nonzero element with its own sign; ``str(DeltaFormula({}))`` is ``'0'``.
Isotope labels and charge suffixes are not part of this notation. A leading
integer is a formula multiplier, so ``13C`` means thirteen carbon atoms, not
carbon-13.

Propagating from a known formula
--------------------------------

.. code-block:: python

   from msbuddy import annotate_delta_mass

   matches = annotate_delta_mass(
       15.99491462,
       tolerance=0.001,
       source_formula='C10H16O2',
   )
   print(matches[0].formula)         # +O
   print(matches[0].target_formula)  # C10H16O3

``source_formula`` can be a neutral formula string, a nonnegative element-count
mapping, or a neutral ``Formula`` object. It excludes changes that would
subtract more atoms than the source contains. This check is necessary, but
does not prove that the propagated target is chemically possible. No molecular
DBE or valence rule is applied to a signed difference: it is a difference of
compositions, not a molecule.

When an adduct is unknown, applying an ion-composition delta such as ``-H+Na``
directly to a **neutral** source formula is not a neutral annotation. Either
keep the raw result as an ion-change hypothesis, or supply both adducts to
``annotate_network_delta``.

Separating adducts from molecular changes
-----------------------------------------

.. code-block:: python

   from msbuddy import annotate_network_delta

   edge = annotate_network_delta(
       source_mz=181.0706645565,
       target_mz=203.0526088063,
       source_adduct='[M+H]+',
       target_adduct='[M+Na]+',
       source_formula='C6H12O6',
       mz_tolerance=5,
       ppm=True,
   )
   best = edge[0]
   print(best.formula)              # 0: no neutral elemental change
   print(best.target_formula)       # C6H12O6
   print(str(edge.adduct_delta))     # -H+Na
   print(str(best.ion_delta))        # -H+Na

For an ion ``[nM+A]z``, the function calculates::

   neutral_mass = (abs(z) * mz - atomic_mass(A) + z * electron_mass) / n

This handles different charges, opposite polarities, multimers, losses such as
``[M+H-H2O]+``, and bare molecular ions such as ``[M]+``. It parses adducts
strictly: an invalid adduct raises an error instead of falling back to a proton
adduct. Double-precision atomic masses are used throughout this API.

The result's ``adduct_delta`` is the target adduct composition minus the source
adduct composition. Each candidate's ``ion_delta`` is the complete atomic ion
composition difference. With unequal multimer counts, calculating that ion
composition requires a source formula; without one, ``ion_delta`` is ``None``.
Electron mass is reported through charge context, not as an atom in a formula.

Adducts are input assumptions. This function does not choose an adduct from a
mass difference. A zero neutral delta is also consistent with isomers, and does
not show that two nodes are the same compound. A retention-time, peak-shape,
isotope-pattern, or other experimental check can provide additional evidence.

Precursor errors and isotope alternatives
-----------------------------------------

``annotate_delta_mass`` defaults to an absolute Da tolerance. If ``ppm=True``,
it requires an explicit positive ``reference_mass``; ppm is never silently
scaled by a small or zero delta mass. ``mass_error`` is theoretical minus
observed mass. ``mass_error_ppm`` is ``None`` unless a reference mass was
explicitly supplied.

The network function accepts one ``mz_tolerance`` for both nodes, or a pair
``(source_tolerance, target_tolerance)``. It converts each precursor's error to
a neutral-mass bound, then adds those bounds conservatively. For example,
5 ppm at each of two approximately 500 Da singly charged precursors gives
approximately 0.005 Da of difference tolerance, even when their difference is
almost zero.

If a source formula is supplied, it must match the source precursor within its
own tolerance, and every propagated target must match the target precursor
within its own tolerance. An inconsistent source raises ``ValueError``. The
returned ``source_mass_error`` and candidate ``target_mass_error`` use neutral
mass units. Small floating-point arithmetic allowances are used at these
precursor-boundary checks.

``result.phenomena`` separately reports matching one- or two-carbon-13-like
isotope offsets. It does not force an isotope mass difference into an ordinary
elemental formula. These labels are alternative hypotheses, not a complete
isotope assignment or a substitute for a measured isotope pattern.

Search scope and cost
---------------------

The search is de novo and exhaustive **within the returned constraints**.
It does not use the legacy formula database. Default signed bounds are:

.. list-table:: Default per-element net changes
   :header-rows: 1
   :widths: 30 30

   * - Element
     - Inclusive bounds
   * - C
     - -6 to +6
   * - H
     - -18 to +18
   * - N
     - -3 to +3
   * - O
     - -6 to +6
   * - S
     - -2 to +2
   * - F
     - -12 to +12
   * - P, Cl, Br, I, Na, K
     - -1 to +1

The defaults also require ``max_changed_atoms=24`` (sum of the absolute net
counts) and ``max_changed_elements=4`` (number of nonzero element counts).
All supported elements, including halogens, are enabled. These limits describe
a small-change search; changes outside them are not covered. In particular,
long multi-step changes may require wider bounds.

For annotation propagation where larger multi-step changes are expected, use
``preset='broad'``. It increases the limits to 40 changed atoms and 6 changed
element types, with wider per-element bounds. For example:

.. code-block:: python

   edge = annotate_network_delta(
       source_mz, target_mz, source_adduct, target_adduct,
       source_formula=known_source_formula,
       preset='broad',
   )

Broad bounds are C ±12, H ±30, N ±6, O ±10, S ±3, P ±2, F ±16,
Cl and Br ±2, and I, Na and K ±1. ``DEFAULT_DELTA_RANGES`` and
``BROAD_DELTA_RANGES`` are available from ``msbuddy.delta`` for inspection.
This preset is most useful with a known source formula, which removes
impossible atom subtractions. Without an anchor, common mass differences can
have more than 10,000 matches and exceed the default result limit. Tighten the
scientific search scope before increasing that limit for a large network.

Explicit ``max_changed_atoms`` and ``max_changed_elements`` override a preset's
limits. Pass ``element_ranges`` to **replace** its per-element ranges; omitted
elements are then fixed at zero. The returned effective bounds always show what
was actually searched, including source-formula constraints.

Set ``max_changed_elements=12`` when every supported element may change
simultaneously. Wider domains increase both ambiguity and computation. The
meet-in-the-middle enumerator caches bounded half-spaces. Before allocating
them it checks ``max_half_states=2_000_000`` for each half; more than
``max_candidates=10_000`` valid matches also raises ``DeltaSearchSpaceError``.
There is no hidden result cap or silent truncation. A limit error means the
caller must narrow the domain/tolerance or explicitly raise the relevant limit.

The maximum absolute signed atom count is 32767 per element. Propagated
nonnegative target formulas use the same 32767-per-element representation limit
as the rest of msbuddy. An empty result is complete within its specified domain;
it is not evidence that no formula difference exists outside that domain.

Transformation explanations and ranking
---------------------------------------

Built-in hypotheses cover common additions/removals such as oxygen, hydrogen,
methylene, water, carbon dioxide, acetyl, glucuronide, sulfate, phosphate,
hexosyl and isoprenyl groups; halogen substitutions; and common ion adduct
exchanges. Names describe possible **net changes**, often with multiple
chemical interpretations. ``max_steps=2`` combines up to two named changes.
It can be set from zero to three. Only shortest unordered additive explanations
are returned; intermediate molecules and the order of steps are not tested.

Changing ``max_steps`` changes hypothesis labels and ordering, not the
enumerated formula space. Opposing transformations that cancel to zero do not
constitute evidence for a pathway and are not listed. The raw API includes ion
exchange hypotheses by default (``include_ion=True``); the neutral network API
does not use them as neutral-transformation explanations.

Candidates sort by zero net change first, then availability and length of a
named explanation, fewer changed atoms, and lower absolute mass error. This
simple ordering is deterministic and intentionally exposed as a heuristic.
Inspect multiple candidates instead of treating the first as a certain
annotation. Formula recall is limited by the declared search domain, while
top-ranked accuracy additionally depends on this ambiguous ordering.

For a network edge with a known source formula, an optional alternative is
``annotate_network_delta(..., source_formula=..., ranking='balanced')``. The
default remains ``ranking='rules'``. The balanced score is::

   0.8 * log1p(changed_atoms) + 0.2 * changed_element_types
   - 1.5 * named_prior + 0.5 * min(10, max(0, -target_DBE))

Here ``named_prior`` is one for zero change, the reciprocal of the shortest
named step count for an explained change, and zero otherwise. Lower scores are
preferred; ties preserve the rules ordering. The DBE term is a soft penalty on
the propagated **neutral target**, never a filter on a signed delta. It does
not remove candidates. This alternative can improve overall ranking while
reducing accuracy in some chemistry groups, including halogen changes; it is
therefore opt-in. It requires a source formula. Custom transformation rules or
bounds can also change its behavior.

``result.ranking_method`` and each candidate's optional ``heuristic_score``
record the chosen ordering. The score has no probability interpretation and
must not be used as an FDR. This option adds no model download or inference
runtime dependency.

Custom rules replace the built-in hypothesis list:

.. code-block:: python

   from msbuddy import DeltaFormula, DeltaTransformation, annotate_delta_mass

   rules = [
       DeltaTransformation('oxygen addition', DeltaFormula({'O': 1})),
       DeltaTransformation('oxygen removal', DeltaFormula({'O': -1})),
   ]
   matches = annotate_delta_mass(31.98982924, transformations=rules, max_steps=2)

The caller supplies reverse rules explicitly if desired. ``kind='ion'`` marks
ion-change rules, and ``kind='chemical'`` is the default. Excessively large
rule/step combinations raise a resource error. Custom rules do not expand the
elemental bounds automatically.

Saving results
--------------

``DeltaMassResult`` acts as a sequence of candidates and retains the mass-space,
tolerance, effective bounds, isotope alternatives and node context. Export it
without serializing Python objects:

.. code-block:: python

   import json

   payload = edge.to_dict()
   print(json.dumps(payload, allow_nan=False))
   assert payload['complete_within_bounds']
   assert not payload['truncated']

The JSON includes the search limits and transformation rules, every candidate's
signed composition, optional target and ion compositions, and the input adduct
and charge context. No probability or FDR field is inferred from a mass match.

