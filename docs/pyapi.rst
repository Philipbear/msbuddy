Python API
-------------

Version 1.0 changes several Python APIs. See the :doc:`migration` guide before
upgrading from 0.x. Public formula compositions are dictionaries mapping
supported element symbols to nonnegative integer counts.

Formula functions
~~~~~~~~~~~~~~~~~

.. function:: read_formula(formula: str)

   Parse a molecular formula string into an element-count dictionary. Supports
   parentheses, component multipliers, and dot-separated components. A dot
   separates components; it does not denote a fractional atom count. Invalid
   syntax raises ``ValueError`` and unsupported elements raise
   ``UnsupportedElementError``, a subclass of ``ValueError``.

   :returns: A dictionary containing nonzero element counts.

.. function:: formula_to_string(composition: Mapping[str, int])

   Format an element-count dictionary as a formula string. Unsupported elements
   and invalid atom counts raise errors.

.. function:: enumerate_subformulas(composition: Mapping[str, int])

   Return a lazy iterator of element-count dictionaries, including the empty
   composition. The number of results can grow rapidly with atom counts.

.. code-block:: python

   from itertools import islice
   from msbuddy import Formula, enumerate_subformulas, formula_to_string, read_formula

   counts = read_formula('C6H12O6')
   formula = Formula(counts)  # or Formula.from_string('C6H12O6')
   print(formula.composition, formula.mass, formula_to_string(counts))
   print(list(islice(enumerate_subformulas(counts), 5)))

``SUPPORTED_ELEMENTS`` contains the symbols C, H, Br, Cl, F, I, K, N, Na,
O, P, and S. Counts must be nonnegative integers no greater than 32767.
The helpers ``msbuddy.utils.read_formula_str`` and
``msbuddy.utils.add_formula_str`` also validate these symbols.

.. function:: assign_subformula(ms2_mz, precursor_formula, adduct, ms2_tol=10, ppm=True, dbe_cutoff=0.0)

   Assign singly charged fragment formulas to an MS/MS spectrum. The precursor
   may be a formula string or an element-count dictionary. Radical fragments
   are considered and candidate subformulas are filtered by DBE and chemical
   rules. Invalid formulas, unsupported elements and invalid adducts raise
   errors, including when the peak list is empty.

   :param ms2_mz: Sequence of MS/MS m/z values.
   :param precursor_formula: Uncharged formula string or element-count mapping.
   :param adduct: Adduct string, such as ``[M+H]+``.
   :param ms2_tol: Fragment m/z tolerance, by default 10 ppm.
   :param ppm: If false, use Da for the tolerance.
   :param dbe_cutoff: Minimum double bond equivalent.
   :returns: A list of :class:`msbuddy.utils.SubformulaResult` objects. Use
      each result's ``peak_index`` and ``formulas`` to inspect matches.

.. code-block:: python

   from msbuddy import assign_subformula

   matches = assign_subformula([107.05, 149.02], {'C': 15, 'H': 16, 'O': 5},
                               '[M+H]+', ms2_tol=0.02, ppm=False)
   for peak in matches:
       print(peak.peak_index, [match.formula for match in peak.formulas])

Mass search functions
~~~~~~~~~~~~~~~~~~~~~

.. function:: mass_to_formula (mass: float, mass_tol: float, ppm: bool, halogen: bool, dbe_cutoff: float, integer_dbe: bool)

   Convert a monoisotopic mass (neutral) to formula, return list of :class:`msbuddy.utils.FormulaResult`. This function relies on the global dependencies within the :class:`msbuddy.Msbuddy`. It works by database searching. Formula results are sorted by the absolute mass error.

   :param mass: float. Target mass, should be <1500 Da.
   :param mass_tol: float. The mass tolerance for searching. Default is 10 ppm.
   :param ppm: bool. If True, the mass tolerance is in ppm. If False, the mass tolerance is in Da. Default is True.
   :param halogen: bool. If True, the halogen elements (F, Cl, Br, I) are considered. Default is False.
   :param dbe_cutoff: float. The DBE cutoff for filtering out formula results. Default is 0.0.
   :param integer_dbe: bool. If True, only return formulas with interger DBE values. Default is True.
   :returns: A list of :class:`msbuddy.utils.FormulaResult` objects.

Example Usage:

.. code-block:: python

   from msbuddy import Msbuddy

   # create a Msbuddy object
   engine = Msbuddy()

   # convert mass to formula
   formula_list = engine.mass_to_formula(300.0000, 10, True, True, 0.0, True)

   # print results
   for f in formula_list:
      print(f.formula, f.mass_error, f.mass_error_ppm)


.. function:: mz_to_formula (mz: float, adduct: str, mz_tol: float, ppm: bool, halogen: bool, dbe_cutoff: float, integer_dbe: bool)

   Convert a m/z value to formula, return list of :class:`msbuddy.utils.FormulaResult`. This function relies on the global dependencies within the :class:`msbuddy.Msbuddy`. It works by database searching. Formula results are sorted by the absolute mass error.

   :param mz: float. Target m/z value, should be <1500.
   :param adduct: str. Precursor type string, e.g. "[M+H]+", "[M-H]-".
   :param mz_tol: float. The m/z tolerance for searching. Default is 10 ppm.
   :param ppm: bool. If True, the m/z tolerance is in ppm. If False, the m/z tolerance is in Da. Default is True.
   :param halogen: bool. If True, the halogen elements (F, Cl, Br, I) are considered. Default is False.
   :param dbe_cutoff: float. The DBE cutoff for filtering out formula results. Default is 0.0.
   :param integer_dbe: bool. If True, only return formulas with interger DBE values. Default is True.
   :returns: A list of :class:`msbuddy.utils.FormulaResult` objects.

Example Usage:

.. code-block:: python

   from msbuddy import Msbuddy

   # create a Msbuddy object
   engine = Msbuddy()

   # convert mz to formula
   formula_list = engine.mz_to_formula(300.0000, "[M+H]+", 10, True, True, 0.0, True)

   # print results
   for f in formula_list:
      print(f.formula, f.mass_error, f.mass_error_ppm)



Classes
~~~~~~~~~~~~~~~
.. class:: msbuddy.Msbuddy (config: Union[MsbuddyConfig, None] = None)

   Main annotation engine. Instances maintain independent data and configuration.

   :param config: :class:`msbuddy.MsbuddyConfig` object. Default is None.

   .. attribute:: config

      :class:`msbuddy.MsbuddyConfig` object. The configuration for the Msbuddy object.

   .. attribute:: data

      A list of :class:`msbuddy.base.MetaFeature` objects. Data loaded into the :class:`msbuddy.Msbuddy` object.

   .. method:: update_config (config: **kwargs)

      Replace the configuration with a new one. Unspecified options return to
      their defaults, and the database dependencies are reloaded.

      :param config: ``**kwargs``, attributes in :class:`msbuddy.MsbuddyConfig` class.
      :returns: None. The ``config`` attribute of the :class:`msbuddy.Msbuddy` object will be updated.

   .. method:: load_usi (usi_list: Union[str, List[str]], adduct_list: Union[None, str, List[str]] = None)

      Read USIs and replace the currently loaded data.

      :param usi_list: str or List[str]. A single USI string or a sequence of USI strings.
      :param optional adduct_list: str or List[str]. A single adduct string or a sequence of adduct strings, which will be applied to all USI strings accordingly.
      :returns: None. A list of :class:`msbuddy.base.MetaFeature` objects will be stored in the ``data`` attribute of the :class:`msbuddy.Msbuddy` object.

   .. method:: load_mgf (mgf_file: str)

      Read a single MGF file and replace the currently loaded data.

      :param mgf_file: str. The path to the mgf file.
      :returns: None. A list of :class:`msbuddy.base.MetaFeature` objects will be stored in the ``data`` attribute of the :class:`msbuddy.Msbuddy` object.

   .. method:: add_data (data)

      Append one feature or an iterable of features to the loaded data.

      :param data: A :class:`msbuddy.base.MetaFeature` or iterable of features.
      :returns: None. A list of :class:`msbuddy.base.MetaFeature` objects will be stored in the ``data`` attribute of the :class:`msbuddy.Msbuddy` object.

   .. method:: clear_data

      Clear the ``data`` attribute of the :class:`msbuddy.Msbuddy` object.

      :returns: None. The ``data`` attribute of the :class:`msbuddy.Msbuddy` object will be cleared to None.

   .. method:: annotate_formula

      Perform formula annotation for loaded data.

      :returns: None. The ``candidate_formula_list`` attribute of each :class:`msbuddy.base.MetaFeature` object in the ``data`` attribute of the :class:`msbuddy.Msbuddy` object will be updated.

   .. method:: get_summary

      Summarize the annotation results.

      :returns: A list of Python dictionaries. Each dictionary contains the summary information for a single :class:`msbuddy.base.MetaFeature` object.


Example Usage:

.. code-block:: python

   from msbuddy import Msbuddy

   # create a Msbuddy object with default configuration
   engine = Msbuddy()

   # load some data here
   engine.load_mgf("demo.mgf")
   # add custom data (List[MetaFeature])
   engine.add_data(...)

   # clear data
   engine.clear_data()

   # update configuration
   engine.update_config(ms_instr="fticr", halogen=True, timeout_secs=100)


.. class:: msbuddy.MsbuddyConfig (ms_instr=None, ppm=True, ms1_tol=5, ms2_tol=10, halogen=False, parallel=False, n_cpu=-1, timeout_secs=300, batch_size=1000, element_ranges=None, isotope_bin_mztol=0.02, max_isotope_cnt=4, rel_int_denoise_cutoff=0.01, top_n_per_50_da=6, data_dir=None, data_files=None)

   It is a class to store all the configurations for **msbuddy**.

   :param ms_instr: str. The mass spectrometry instrument type, used for automated mass tolerance setting. Supported instruments are "orbitrap", "fticr" and "qtof". Default is None. If None, parameters ``ppm``, ``ms1_tol`` and ``ms2_tol`` will be used.
   :param ppm: bool. If True, the mass tolerance is in ppm. If False, the mass tolerance is in Da. Default is True.
   :param ms1_tol: float. The mass tolerance for MS1 spectra. Default is 5 ppm.
   :param ms2_tol: float. The mass tolerance for MS/MS spectra. Default is 10 ppm.
   :param halogen: bool. If True, the halogen elements (F, Cl, Br, I) are considered. Default is False.
   :param parallel: bool. If True, the annotation is performed in parallel. Default is False.
   :param n_cpu: int. The number of CPUs to use. Default is -1, which means all available CPUs.
   :param timeout_secs: float. The timeout in seconds for each query. Default is 300 seconds.
   :param batch_size: int. The batch size for formula annotation; a larger batch size takes more memory. Default is 1000.
   :param element_ranges: Mapping of supported element symbols to inclusive
      ``(minimum, maximum)`` pairs. Unspecified elements retain their defaults.
      Nonzero halogen ranges require ``halogen=True``.
   :param isotope_bin_mztol: float. The mass tolerance for MS1 isotope binning, in Da. Default is 0.02 Da.
   :param max_isotope_cnt: int. The maximum number of isotopes to consider. Default is 4.
   :param rel_int_denoise_cutoff: float. The cutoff for relative intensity denoising. Default is 0.01 (1%).
   :param top_n_per_50_da: int. The maximum number of fragments to reserve in every 50 Da. Default is 6.

Example Usage:

.. code-block:: python

    from msbuddy import Msbuddy, MsbuddyConfig

    # create a MsbuddyConfig object
    msb_config = MsbuddyConfig(
        ms_instr="orbitrap",
        halogen=True,
        element_ranges={"C": (0, 40), "Cl": (0, 2)},
        timeout_secs=100)

    # create a Msuddy object with the specified configuration
    msb_engine = Msbuddy(msb_config)



.. class:: msbuddy.base.Spectrum (mz_array: np.array, int_array: np.array)

    A class to represent a mass spectrum.

   :param mz_array: A numpy array of m/z values.
   :param int_array: A numpy array of intensity values.

   .. attribute:: mz_array

      A numpy array of m/z values.

   .. attribute:: int_array

      A numpy array of intensity values.

Example usage:

.. code-block:: python

    import numpy as np
    from msbuddy.base import Spectrum

    mz_array = np.array([100, 200, 300, 400, 500])
    int_array = np.array([10, 20, 30, 40, 50])
    spectrum = Spectrum(mz_array, int_array)



.. class:: msbuddy.base.Formula (composition, charge=0, mass=None, isotope=0)

    A class to represent a molecular formula.

   :param composition: Mapping from element symbols to nonnegative integer counts.
   :param charge: int. The charge of the molecular formula; defaults to zero.
   :param optional mass: float. The exact mass of the molecular formula. Default is None, exact mass will be calculated.
   :param isotope: int. The isotopologue of the formula (e.g., 1 for M+1). Default is 0, which means M+0.


   .. classmethod:: from_string(formula, charge=0)

      Construct a formula from a molecular formula string.

   .. attribute:: composition

      A new dictionary containing the nonzero element counts. Array storage is
      an internal implementation detail.

   .. attribute:: charge

      int. The charge of the molecular formula.

   .. attribute:: mass

      float. The exact mass of the molecular formula.

   .. attribute:: isotope

      int. The isotopologue of the formula.

   .. attribute:: dbe

      float. The double bond equivalent (DBE) of the formula.



.. class:: msbuddy.base.Adduct (string: Union[str, None], pos_mode: bool, report_invalid: bool)

    A class to represent an adduct type. Malformed adducts use the default
    unless ``report_invalid=True``. Unsupported elements always raise
    ``UnsupportedElementError``.

   :param optional string: str. The adduct type. Default is [M+H]+ for positive mode and [M-H]- for negative mode.
   :param pos_mode: bool. True for positive mode and False for negative mode.
   :param report_invalid: bool. If True, an error will be raised if the input adduct type cannot be parsed. If False, the default adduct type will be used. Default is False.


   .. attribute:: string

      str. The adduct type.

   .. attribute:: pos_mode

      bool. True for positive mode and False for negative mode.

   .. attribute:: charge

      int. The charge of the adduct.

   .. attribute:: m

      int. The count of multimer (M) in the adduct. e.g. [M+H]+ has m=1, [2M+H]+ has m=2.

   .. attribute:: net_formula

      :class:`msbuddy.base.Formula` object describing the net change. Its
      ``composition`` can contain negative counts; for example, [M+H-H2O]+
      has ``{'H': -1, 'O': -1}``.



.. class:: msbuddy.base.ProcessedMS1 (mz: float, raw_spec: Spectrum, charge: int, mz_tol: float, ppm: bool, isotope_bin_mztol: float, max_isotope_cnt: int)

    A class to represent a processed MS1 spectrum, for MS1 isotopic pattern extraction.

   :param mz: float. Precursor ion m/z.
   :param raw_spec: :class:`msbuddy.base.Spectrum` object. Raw MS1 spectrum.
   :param charge: int. Precursor ion charge.
   :param mz_tol: float. The mass tolerance for MS1 spectra.
   :param ppm: bool. If True, the mass tolerance is in ppm. If False, the mass tolerance is in Da.
   :param isotope_bin_mztol: float. The mass tolerance for MS1 isotope binning, in Da.
   :param max_isotope_cnt: int. The maximum number of isotopes to consider.


   .. attribute:: mz_tol

      float. The mass tolerance for MS1 spectra.

   .. attribute:: ppm

      bool. If True, the mass tolerance is in ppm. If False, the mass tolerance is in Da.

   .. attribute:: idx_array

      A numpy array of raw indices of selected peaks.

   .. attribute:: mz_array

      A numpy array of m/z values of selected peaks.

   .. attribute:: int_array

      A numpy array of intensity values of selected peaks.



.. class:: msbuddy.base.ProcessedMS2 (mz: float, raw_spec: Spectrum, mz_tol: float, ppm: bool, rel_int_denoise_cutoff: float, top_n_per_50_da: int, charge: int = 1)

    A class to represent a processed MS/MS spectrum, for MS/MS preprocessing (deprecursor, denoise, reserve top N fragments).

   :param mz: float. Precursor ion m/z.
   :param raw_spec: :class:`msbuddy.base.Spectrum` object. Raw MS/MS spectrum.
   :param mz_tol: float. The mass tolerance for MS/MS spectra.
   :param ppm: bool. If True, the mass tolerance is in ppm. If False, the mass tolerance is in Da.
   :param rel_int_denoise_cutoff: float. The cutoff for relative intensity denoising.
   :param top_n_per_50_da: int. The maximum number of fragments to reserve in every 50 Da.
   :param charge: int. The precursor ion charge; defaults to 1.

   .. attribute:: mz_tol

      float. The mass tolerance for MS1 spectra.

   .. attribute:: ppm

      bool. If True, the mass tolerance is in ppm. If False, the mass tolerance is in Da.

   .. attribute:: idx_array

      A numpy array of raw indices of selected peaks.

   .. attribute:: mz_array

      A numpy array of m/z values of selected peaks.

   .. attribute:: int_array

      A numpy array of intensity values of selected peaks.

   .. method:: normalize_intensity(method: str)

      Normalize the intensity of the MS/MS spectrum.

      :param method: str. The normalization method, either "sum" or "max".
      :returns: None. The ``int_array`` attribute of the :class:`msbuddy.base.ProcessedMS2` object will be updated.



.. class:: msbuddy.base.MS2Explanation (idx_array: np.array, explanation_array: List[Union[Formula, None]])

    A class to represent MS/MS explanation.

   :param idx_array: numpy array. The indices of the fragments.
   :param explanation_list: A list of :class:`msbuddy.base.Formula` objects. The explanations for the fragments.

   .. attribute:: idx_array

      A numpy array of the indices of the fragments being explained.

   .. attribute:: explanation_list

      A list of :class:`msbuddy.base.Formula` objects. The explanations for the fragments.



.. class:: msbuddy.base.CandidateFormula (formula: Formula, ms1_isotope_similarity: Union[float, None] = None, ms2_raw_explanation: Union[MS2Explanation, None] = None)

    A class to represent a candidate formula.

   :param formula: :class:`msbuddy.base.Formula` object. The candidate formula (in neutral form).
   :param charged_formula: :class:`msbuddy.base.Formula` object. The candidate formula (in charged form).
   :param optional ms1_isotope_similarity: float. The isotope similarity between the candidate formula and the MS1 isotopic pattern.
   :param optional ms2_raw_explanation: :class:`msbuddy.base.MS2Explanation` object. The MS/MS explanation for the candidate formula.

   .. attribute:: formula

      :class:`msbuddy.base.Formula` object. The candidate formula (in neutral form).

   .. attribute:: charged_formula

      :class:`msbuddy.base.Formula` object. The candidate formula (in charged form).

   .. attribute:: ms1_isotope_similarity

      float. The isotope similarity between the candidate formula and the MS1 isotopic pattern.

   .. attribute:: ms2_raw_explanation

      :class:`msbuddy.base.MS2Explanation` object. The MS/MS explanation for the candidate formula.

   .. attribute:: estimated_prob

      float. The estimated formula probability predicted by the ML-b model.

   .. attribute:: normed_estimated_prob

      float. The normalized estimated formula probability considering all the candidate formulas for the same metabolic feature.

   .. attribute:: estimated_fdr

      float. The estimated FDR of the candidate formula.



.. class:: msbuddy.base.MetaFeature (identifier: Union[str, int], mz: float, charge: int, rt: Union[float, None] = None, adduct: Union[str, None] = None, ms1: Union[Spectrum, None] = None, ms2: Union[Spectrum, None] = None)

    A class to represent a metabolic feature.

   :param identifier: str or int. A unique identifier for the metabolic feature.
   :param mz: float. Precursor ion m/z.
   :param charge: int. Precursor ion charge.
   :param optional rt: float. Retention time in seconds. Default is None.
   :param optional adduct: str. Adduct type. If omitted, the default adds or removes one proton per charge, such as [M+H]+ or [M+2H]2+.
   :param optional ms1: :class:`msbuddy.base.Spectrum` object. MS1 spectrum containing the isotopic pattern information. Default is None.
   :param optional ms2: :class:`msbuddy.base.Spectrum` object. MS/MS spectrum. Default is None.

   .. attribute:: identifier

      str. The unique identifier for the metabolic feature.

   .. attribute:: mz

      float. Precursor ion m/z.

   .. attribute:: charge

      int. Precursor ion charge.

   .. attribute:: rt

      float. Retention time in seconds.

   .. attribute:: adduct

      :class:`msbuddy.base.Adduct` object representing the adduct type.

   .. attribute:: ms1_raw

      :class:`msbuddy.base.Spectrum` object. Raw MS1 spectrum.

   .. attribute:: ms2_raw

      :class:`msbuddy.base.Spectrum` object. Raw MS/MS spectrum.

   .. attribute:: ms1_processed

      :class:`msbuddy.base.ProcessedMS1` object. Processed MS1 spectrum.

   .. attribute:: ms2_processed

      :class:`msbuddy.base.ProcessedMS2` object. Processed MS/MS spectrum.

   .. attribute:: candidate_formula_list

      A list of :class:`msbuddy.base.CandidateFormula` objects. Candidate formulas generated for the metabolic feature.



.. class:: msbuddy.utils.FormulaResult (formula: str, mass: float, t_mass: float)

    FormulaResult class, for API output usage.

   :param formula: str. The molecular formula string.
   :param mass: float. The exact mass of the formula
   :param t_mass: float. The target mass.

   .. attribute:: formula

      str. The molecular formula string.

   .. attribute:: mass

      float. The calculated mass or m/z of the formula.

   .. attribute:: target_mass

      float. The input mass or m/z used for the search.

   .. attribute:: mass_error

      float. Mass error (Da) between the formula and the target mass.

   .. attribute:: mass_error_ppm

      float. Mass error in ppm.


.. class:: msbuddy.utils.SubformulaResult (idx: int, subform_list: List[FormulaResult])

    SubformulaResult class, for API output usage.

   :param idx: int. The index of the fragment ion.
   :param subform_list: List[FormulaResult]. A list of :class:`msbuddy.utils.FormulaResult` objects.

   .. attribute:: peak_index

      Index of the fragment peak in the input m/z sequence.

   .. attribute:: formulas

      Formula matches for the peak, as :class:`msbuddy.utils.FormulaResult` objects.

   .. attribute:: idx

      Alias of ``peak_index``.

   .. attribute:: subform_list

      Alias of ``formulas``.
