Python API
-------------

Classes
~~~~~~~~~~~~~~~
.. class:: msbuddy.Buddy (param_set: Union[BuddyParamSet, None] = None)

   Buddy main class.

   :param param1: :class:`msbuddy.BuddyParamSet` object. Default is None.

   .. attribute:: param_set

      :class:`msbuddy.BuddyParamSet` object. The parameter set for the Buddy object.

   .. attribute:: data

      A list of :class:`msbuddy.base.MetaFeature` objects. Data loaded into the Buddy object.

   .. attribute:: db_loaded

      bool. True if the database is loaded.

   .. method:: load_usi (usi_list: Union[str, List[str]], adduct_list: Union[None, str, List[str]] = None)

      Read from a single USI string or a sequence of USI strings, and load the data into the ``data`` attribute of the :class:`Buddy` object.

      :param usi_list: str or List[str]. A single USI string or a sequence of USI strings.
      :param optional adduct_list: str or List[str]. A single adduct string or a sequence of adduct strings, which will be applied to all USI strings accordingly.
      :returns: None. A list of :class:`msbuddy.base.MetaFeature` objects will be stored in the ``data`` attribute of the :class:`Buddy` object.

   .. method:: load_mgf (mgf_file: str)

      Read a single mgf file, and load the data into the ``data`` attribute of the :class:`msbuddy.Buddy` object.

      :param mgf_file: str. The path to the mgf file.
      :returns: None. A list of :class:`msbuddy.base.MetaFeature` objects will be stored in the ``data`` attribute of the :class:`Buddy` object.

   .. method:: add_data (data: List[MetaFeature])

      Add data into the ``data`` attribute of the :class:`msbuddy.Buddy` object.

      :param data: A list of :class:`msbuddy.base.MetaFeature` objects. The data to be added.
      :returns: None. A list of :class:`msbuddy.base.MetaFeature` objects will be stored in the ``data`` attribute of the :class:`Buddy` object.

   .. method:: annotate_formula

      Perform formula annotation for loaded data.

      :returns: None. The ``candidate_formula_list`` attribute of each :class:`msbuddy.base.MetaFeature` object in the ``data`` attribute of the :class:`Buddy` object will be updated.

   .. method:: get_summary

      Summarize the annotation results.

      :returns: A list of Python dictionaries. Each dictionary contains the summary information for a single :class:`msbuddy.base.MetaFeature` object.



.. class:: msbuddy.BuddyParamSet (ppm: bool = True, ms1_tol: float = 5, ms2_tol: float = 10, halogen: bool = False, parallel: bool = False, n_cpu: int = -1, timeout_secs: float = 300, c_range: Tuple[int, int] = (0, 80), h_range: Tuple[int, int] = (0, 150), n_range: Tuple[int, int] = (0, 20), o_range: Tuple[int, int] = (0, 30), p_range: Tuple[int, int] = (0, 10), s_range: Tuple[int, int] = (0, 15), f_range: Tuple[int, int] = (0, 20), cl_range: Tuple[int, int] = (0, 15), br_range: Tuple[int, int] = (0, 10), i_range: Tuple[int, int] = (0, 10), isotope_bin_mztol: float = 0.02, max_isotope_cnt: int = 4, ms2_denoise: bool = True, rel_int_denoise: bool = True, rel_int_denoise_cutoff: float = 0.01, max_noise_frag_ratio: float = 0.85, max_noise_rsd: float = 0.20, max_frag_reserved: int = 50, use_all_frag: bool = False)

   It is a class to store all the parameter settings for **msbuddy**.

   :param ppm: bool. If True, the mass tolerance is in ppm. If False, the mass tolerance is in Da. Default is True.
   :param ms1_tol: float. The mass tolerance for MS1 spectra. Default is 5 ppm.
   :param ms2_tol: float. The mass tolerance for MS/MS spectra. Default is 10 ppm.
   :param halogen: bool. If True, the halogen elements (F, Cl, Br, I) are considered. Default is False.
   :param parallel: bool. If True, the annotation is performed in parallel. Default is False.
   :param n_cpu: int. The number of CPUs to use. Default is -1, which means all available CPUs.
   :param timeout_secs: float. The timeout in seconds for each query. Default is 300 seconds.
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
   :param ms2_denoise: bool. If True, the MS/MS spectra are denoised (see details in `our paper <https://doi.org/10.1038/s41592-023-01850-x>`_). Default is True.
   :param rel_int_denoise: bool. If True, the MS/MS spectra are denoised based on relative intensity. Default is True.
   :param rel_int_denoise_cutoff: float. The cutoff for relative intensity denoising. Default is 0.01 (1%).
   :param max_noise_frag_ratio: float. The maximum ratio of noise fragments to total fragments. Default is 0.85 (85%).
   :param max_noise_rsd: float. The maximum relative standard deviation of noise fragments. Default is 0.20 (20%).
   :param max_frag_reserved: int. The maximum number of fragments to reserve. Default is 50.
   :param use_all_frag: bool. If True, all fragments are used. If False, only the top fragments are used. Default is False.

Example Usage:

.. code-block:: python

    from msbuddy import Buddy, BuddyParamSet

    # create a parameter set
    buddy_param_set = BuddyParamSet(
        ppm=True,
        ms1_tol=10,
        ms2_tol=20,
        halogen=True,
        parallel=True,
        n_cpu=4,
        timeout_secs=600)

    # create a Buddy object with the specified parameter set
    buddy = Buddy(buddy_param_set)



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
   :param isotope: int. The isotopologue of the formula. Default is 0, which means M+0.


   .. attribute:: array

      A numpy arrat of the molecular formula array.

   .. attribute:: charge

      int. The charge of the molecular formula.

   .. attribute:: mass

      float. The exact mass of the molecular formula.

   .. attribute:: isotope

      int. The isotopologue of the formula.

   .. attribute:: dbe

      float. The double bond equivalent (DBE) of the formula.



.. class:: msbuddy.base.Adduct (string: Union[str, None], pos_mode: bool)

    A class to represent an adduct type. If a invalid string is given, the default adduct type will be used.

   :param optional string: str. The adduct type. Default is [M+H]+ for positive mode and [M-H]- for negative mode.
   :param pos_mode: bool. True for positive mode and False for negative mode.


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



.. class:: msbuddy.base.ProcessedMS2 (mz: float, raw_spec: Spectrum, mz_tol: float, ppm: bool, denoise: bool, rel_int_denoise: bool, rel_int_denoise_cutoff: float, max_noise_frag_ratio: float, max_noise_rsd: float, max_frag_reserved: int, use_all_frag: bool = False)

    A class to represent a processed MS/MS spectrum, for MS/MS preprocessing (deprecursor, denoise, reserve top N fragments).

   :param mz: float. Precursor ion m/z.
   :param raw_spec: :class:`msbuddy.base.Spectrum` object. Raw MS1 spectrum.
   :param mz_tol: float. The mass tolerance for MS1 spectra.
   :param ppm: bool. If True, the mass tolerance is in ppm. If False, the mass tolerance is in Da.
   :param denoise: bool. If True, the MS/MS spectrum is denoised (see details in `our paper <https://doi.org/10.1038/s41592-023-01850-x>`_).
   :param rel_int_denoise: bool. If True, the MS/MS spectrum is denoised based on relative intensity.
   :param rel_int_denoise_cutoff: float. The cutoff for relative intensity denoising.
   :param max_noise_frag_ratio: float. The maximum ratio of noise fragments to total fragments.
   :param max_noise_rsd: float. The maximum relative standard deviation of noise fragments.
   :param max_frag_reserved: int. The maximum number of fragments to reserve.
   :param use_all_frag: bool. If True, all fragments are used. If False, only the top fragments are used.

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
   :param explanation_array: A list of :class:`msbuddy.base.Formula` objects. The explanations for the fragments.

   .. attribute:: idx_array

      A numpy array of the indices of the fragments being explained.

   .. attribute:: explanation_array

      A list of :class:`msbuddy.base.Formula` objects. The explanations for the fragments.



.. class:: msbuddy.base.CandidateFormula (formula: Formula, ms1_isotope_similarity: Union[float, None] = None, ms2_raw_explanation: Union[MS2Explanation, None] = None)

    A class to represent a candidate formula.

   :param formula: :class:`msbuddy.base.Formula` object. The candidate formula (in neutral form).
   :param optional ms1_isotope_similarity: float. The isotope similarity between the candidate formula and the MS1 isotopic pattern.
   :param optional ms2_raw_explanation: :class:`msbuddy.base.MS2Explanation` object. The MS/MS explanation for the candidate formula.

   .. attribute:: formula

      :class:`msbuddy.base.Formula` object. The candidate formula (in neutral form).

   .. attribute:: ms1_isotope_similarity

      float. The isotope similarity between the candidate formula and the MS1 isotopic pattern.

   .. attribute:: ms2_raw_explanation

      :class:`msbuddy.base.MS2Explanation` object. The MS/MS explanation for the candidate formula.

   .. attribute:: ml_a_prob

      float. The formula feasibility predicted by the ML-a model.

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




Functions
~~~~~~~~~~~~~~~
.. function:: generate_candidate_formula (meta_feature: MetaFeature, param_set: BuddyParamSet)

   Generate candidate formula for a given metabolic feature based on the given parameter set.

   :param meta_feature: :class:`msbuddy.base.MetaFeature` object.
   :param param_set: :class:`msbuddy.BuddyParamSet` object.
   :returns: :class:`msbuddy.base.MetaFeature` object, with :class:`msbuddy.base.CandidateFormula` objects updated.

Example Usage:

.. code-block:: python

   from msbuddy import generate_candidate_formula

   # generate candidate formulas for a given metabolic feature based on the given parameter set
   mf = generate_candidate_formula(meta_feature, param_set)

   # print all the candidate formula strings and their estimated FDRs
   for candidate_formula in mf.candidate_formula_list:
      print(candidate_formula.formula.__str__() + "\t" + str(candidate_formula.estimated_fdr))


.. function:: read_formula (formula_string: str)

   Read a neutral formula string and return a numpy array, return None if invalid

   :param formula_string: str. The molecular formula string.
   :returns: A numpy array of the molecular formula array in the format of [C, H, Br, Cl, F, I, K, N, Na, O, P, S]. None if invalid.

Example Usage:

.. code-block:: python

   from msbuddy.base import read_formula

   formula_array = read_formula("C10H20O5")
   print(formula_array)
