Migrating from 0.x to 1.0
=========================

Version 1.0 changes the Python API and is not fully compatible with 0.x.
Existing scripts that use element arrays, individual element-range arguments,
or repeated calls to ``add_data`` should be updated before upgrading.

Formula compositions
--------------------

``read_formula`` now returns a dictionary instead of a fixed-order NumPy
array. Construct ``Formula`` with an element-count mapping, or use
``Formula.from_string``::

   from msbuddy import Formula, formula_to_string, read_formula

   counts = read_formula('C6H12O6')
   formula = Formula(counts, charge=0)
   same_formula = Formula.from_string('C6H12O6')
   print(formula.composition, formula_to_string(counts))

Replace the old top-level ``form_arr_to_str`` with ``formula_to_string`` and
``enumerate_subform_arr`` with ``enumerate_subformulas``. The latter returns
a lazy iterator of dictionaries, including the empty composition. It can
produce a very large number of results; use ``itertools.islice`` when only
the first few are needed.

Counts must be nonnegative integers. Supported symbols are C, H, Br, Cl, F,
I, K, N, Na, O, P, and S; ``SUPPORTED_ELEMENTS`` provides this list in code.
Unsupported symbols raise ``UnsupportedElementError``, which is a
``ValueError``. Invalid syntax and atom counts also raise errors rather than
returning ``None``. This applies to formula strings, composition dictionaries,
and element ranges. Unsupported elements in supplied adducts also raise an
error, even when there are no MS/MS peaks.

``assign_subformula`` accepts either a formula string or an element-count
mapping. Its results have descriptive ``peak_index`` and ``formulas``
properties; the existing ``idx`` and ``subform_list`` names remain available.
Both ``assign_subformula`` and ``mz_to_formula`` reject malformed adducts
instead of silently using a default.

Element ranges
--------------

Replace separate parameters such as ``c_range`` and ``o_range`` with one
mapping::

   from msbuddy import MsbuddyConfig

   config = MsbuddyConfig(element_ranges={'C': (0, 40), 'O': (1, 10)})
   halogenated = MsbuddyConfig(halogen=True, element_ranges={'Cl': (0, 2)})

Each pair is inclusive. Unspecified elements keep their defaults; K and Na
default to zero. Explicit nonzero halogen ranges require ``halogen=True``.
The existing command-line element-limit options remain available.

Adding data
-----------

``Msbuddy.add_data`` now appends a feature or an iterable of features. In 0.x
it replaced the dataset. To replace existing data, call ``clear_data`` first::

   engine.clear_data()
   engine.add_data(new_features)

``load_mgf`` and ``load_usi`` continue to replace the currently loaded data.
