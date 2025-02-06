Python API
-------------

Functions
~~~~~~~~~~~~~~~
.. function:: read_formula (formula_string: str)

   Read a formula string (in neutral form) and return a numpy array ([C, H, Br, Cl, F, I, K, N, Na, O, P, S]), return None if invalid. For a more general approach (no element restriction), use :func:`msbuddy.utils.read_formula_str`.

   :param formula_string: str. The molecular formula string.
   :returns: A numpy array of the molecular formula array in the format of [C, H, Br, Cl, F, I, K, N, Na, O, P, S]. None if invalid.

Example Usage:

.. code-block:: python

   from msbuddy.utils import read_formula

   formula_array = read_formula("C10H20O5")
   print(formula_array)


.. function:: read_formula_str (formula_string: str)

   Read a formula string and return a dictionary. It can deal with cases such as "2H2O" and "C5H7NO2.HCl".

   :param formula_string: str. The molecular formula string.
   :returns: A dictionary of the parsed molecular formula.

Example Usage:

.. code-block:: python

   from msbuddy.utils import read_formula_str

   formula_dict = read_formula_str("2H2O")
   print(formula_dict)


.. function:: add_formula_str (formula_str1: str, formula_str2: str)

   Add up two formula strings and return a dictionary.

   :param formula_str1: str. The molecular formula string.
   :param formula_str2: str. The molecular formula string.
   :returns: A dictionary of the summed molecular formula.

Example Usage:

.. code-block:: python

   from msbuddy.utils import add_formula_str

   formula_dict = add_formula_str("C6H12O5", "H2O")
   print(formula_dict)


.. function:: form_arr_to_str (formula_array: List[int])

   Read a neutral formula array and return the Hill string.

   :param formula_array: List[int]. The molecular formula array in the format of [C, H, Br, Cl, F, I, K, N, Na, O, P, S].
   :returns: The Hill string of the molecular formula.

Example Usage:

.. code-block:: python

   from msbuddy.utils import form_arr_to_str

   formula_str = form_arr_to_str([10, 20, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0])
   print(formula_str)


.. function:: assign_subformula (ms2_mz: List[float], precursor_formula: str, adduct: str, ms2_tol: float, ppm: bool, dbe_cutoff: float)

   Assign subformulas to an MS/MS spectrum with a given precursor formula and adduct. Radical fragment ions are considered. Double bond equivalent (DBE) cutoff is used to filter out subformulas.
   A soft version of SENIOR rules and other rules (remove invalid subformulas such as "C4", "N4") are also applied. Note that input precursor formula strings should only contain CHNOPSFClBrINaK.

   :param ms2_mz: List[float]. A list-like object (or 1D numpy array) of the m/z values of the MS/MS spectrum.
   :param precursor_formula: str. The precursor formula string (in uncharged form). e.g., "C10H20O5".
   :param adduct: str. The adduct type string. e.g., "[M+H]+". If the input adduct is not recognized, the default adduct type (M +/- H) will be used.
   :param ms2_tol: float. The m/z tolerance for MS/MS spectra. Default is 10 ppm.
   :param ppm: bool. If True, the m/z tolerance is in ppm. If False, the m/z tolerance is in Da. Default is True.
   :param dbe_cutoff: float. The DBE cutoff for filtering out subformulas. Default is 0.0.
   :returns: A list of :class:`msbuddy.utils.SubformulaResult` objects.

Example Usage:

.. code-block:: python

   from msbuddy import assign_subformula

   subformla_list = assign_subformula([107.05, 149.02, 209.04, 221.04, 230.96],
                                      precursor_formula="C15H16O5", adduct="[M+H]+",
                                      ms2_tol=0.02, ppm=False, dbe_cutoff=0.0)


.. function:: enumerate_subform_arr (formula_array: List[int])

   Enumerate all possible sub-formula arrays of a given formula array.

   :param formula_array: List[int]. A list-like object (or 1D numpy array) of the molecular formula array.
   :returns: A 2D numpy array, with each row being a sub-formula array.

Example Usage:

.. code-block:: python

   from msbuddy.utils import enumerate_subform_arr

   all_subform_arr = enumerate_subform_arr([10, 20, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0])
   print(all_subform_arr)

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

   Buddy main class. Note that the Buddy class is singleton, which means only one Buddy object can be created.

   :param config: :class:`msbuddy.MsbuddyConfig` object. Default is None.

   .. attribute:: config

      :class:`msbuddy.MsbuddyConfig` object. The configuration for the Msbuddy object.

   .. attribute:: data

      A list of :class:`msbuddy.base.MetaFeature` objects. Data loaded into the :class:`msbuddy.Msbuddy` object.

   .. attribute:: db_loaded

      bool. True if the database is loaded.

   .. method:: update_config (config: **kwargs)

      Update the configuration for the :class:`msbuddy.Msbuddy` object.

      :param config: **kwargs, attributes in :class:`msbuddy.MsbuddyConfig` class.
      :returns: None. The ``config`` attribute of the :class:`msbuddy.Msbuddy` object will be updated.

   .. method:: load_usi (usi_list: Union[str, List[str]], adduct_list: Union[None, str, List[str]] = None)

      Read from a single USI string or a sequence of USI strings, and load the data into the ``data`` attribute of the :class:`msbuddy.Msbuddy` object.

      :param usi_list: str or List[str]. A single USI string or a sequence of USI strings.
      :param optional adduct_list: str or List[str]. A single adduct string or a sequence of adduct strings, which will be applied to all USI strings accordingly.
      :returns: None. A list of :class:`msbuddy.base.MetaFeature` objects will be stored in the ``data`` attribute of the :class:`msbuddy.Msbuddy` object.

   .. method:: load_mgf (mgf_file: str)

      Read a single mgf file, and load the data into the ``data`` attribute of the :class:`msbuddy.Msbuddy` object.

      :param mgf_file: str. The path to the mgf file.
      :returns: None. A list of :class:`msbuddy.base.MetaFeature` objects will be stored in the ``data`` attribute of the :class:`msbuddy.Msbuddy` object.

   .. method:: add_data (data: List[MetaFeature])

      Add data into the ``data`` attribute of the :class:`msbuddy.Msbuddy` object.

      :param data: A list of :class:`msbuddy.base.MetaFeature` objects. The data to be added.
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


.. class:: msbuddy.MsbuddyConfig (ms_instr: str = None, ppm: bool = True, ms1_tol: float = 5, ms2_tol: float = 10, halogen: bool = False, parallel: bool = False, n_cpu: int = -1, timeout_secs: float = 300, batch_size: int = 1000, c_range: Tuple[int, int] = (0, 80), h_range: Tuple[int, int] = (0, 150), n_range: Tuple[int, int] = (0, 20), o_range: Tuple[int, int] = (0, 30), p_range: Tuple[int, int] = (0, 10), s_range: Tuple[int, int] = (0, 15), f_range: Tuple[int, int] = (0, 20), cl_range: Tuple[int, int] = (0, 15), br_range: Tuple[int, int] = (0, 10), i_range: Tuple[int, int] = (0, 10), isotope_bin_mztol: float = 0.02, max_isotope_cnt: int = 4, rel_int_denoise_cutoff: float = 0.01, top_n_per_50_da: int = 6)

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
   :param c_range: Tuple[int, int]. The range of carbon atoms. Default is (0, 80).
   :param h_range: Tuple[int, int]. The range of hydrogen atoms. Default is (0, 150).
   :param n_range: Tuple[int, int]. The range of nitrogen atoms. Default is (0, 20).
   :param o_range: Tuple[int, int]. The range of oxygen atoms. Default is (0, 30).
   :param p_range: Tuple[int, int]. The range of phosphorus atoms. Default is (0, 10).
   :param s_range: Tuple[int, int]. The range of sulfur atoms. Default is (0, 15).
   :param f_range: Tuple[int, int]. The range of fluorine atoms. Default is (0, 20).
   :param cl_range: Tuple[int, int]. The range of chlorine atoms. Default is (0, 15).
   :param br_range: Tuple[int, int]. The range of bromine atoms. Default is (0, 10).
   :param i_range: Tuple[int, int]. The range of iodine atoms. Default is (0, 10).
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



.. class:: msbuddy.base.Formula (array: np.array, charge: int, mass: Union[float, None] = None, isotope: int = 0)

    A class to represent a molecular formula.

   :param array: numpy array. The molecular formula array in the format of [C, H, Br, Cl, F, I, K, N, Na, O, P, S].
   :param charge: int. The charge of the molecular formula.
   :param optional mass: float. The exact mass of the molecular formula. Default is None, exact mass will be calculated.
   :param isotope: int. The isotopologue of the formula (e.g., 1 for M+1). Default is 0, which means M+0.


   .. attribute:: array

      A numpy array of the molecular formula array.

   .. attribute:: charge

      int. The charge of the molecular formula.

   .. attribute:: mass

      float. The exact mass of the molecular formula.

   .. attribute:: isotope

      int. The isotopologue of the formula.

   .. attribute:: dbe

      float. The double bond equivalent (DBE) of the formula.



.. class:: msbuddy.base.Adduct (string: Union[str, None], pos_mode: bool, report_invalid: bool)

    A class to represent an adduct type. If a invalid string is given, the default adduct type will be used.

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

      :class:`msbuddy.base.Formula` object. The net formula of the adduct. For example, [M+H-H2O]+ has net formula H-1O-1.



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



.. class:: msbuddy.base.ProcessedMS2 (mz: float, raw_spec: Spectrum, mz_tol: float, ppm: bool, denoise: bool, rel_int_denoise_cutoff: float, top_n_per_50_da: int)

    A class to represent a processed MS/MS spectrum, for MS/MS preprocessing (deprecursor, denoise, reserve top N fragments).

   :param mz: float. Precursor ion m/z.
   :param raw_spec: :class:`msbuddy.base.Spectrum` object. Raw MS1 spectrum.
   :param mz_tol: float. The mass tolerance for MS1 spectra.
   :param ppm: bool. If True, the mass tolerance is in ppm. If False, the mass tolerance is in Da.
   :param rel_int_denoise_cutoff: float. The cutoff for relative intensity denoising.
   :param top_n_per_50_da: int. The maximum number of fragments to reserve in every 50 Da.

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
   :param optional adduct: str. Adduct type. Default is [M+H]+ for positive mode and [M-H]- for negative mode.
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

   .. attribute:: mass_error

      float. Mass error (Da) between the formula and the target mass.

   .. attribute:: mass_error_ppm

      float. Mass error in ppm.


.. class:: msbuddy.utils.SubformulaResult (idx: int, subform_list: List[FormulaResult])

    SubformulaResult class, for API output usage.

   :param idx: int. The index of the fragment ion.
   :param subform_list: List[FormulaResult]. A list of :class:`msbuddy.utils.FormulaResult` objects.

   .. attribute:: idx

      int. The index of the fragment ion.

   .. attribute:: subform_list

      A list of :class:`msbuddy.utils.FormulaResult` objects.