Python API
-------------

Base Classes
~~~~~~~~~~~~~~~
.. class:: Buddy (param_set: Union[BuddyParamSet, None] = None)

   Buddy main class.

   :param param1: :class:`BuddyParamSet` object. Default is None.

   .. attribute:: param_set

      :class:`BuddyParamSet` object. The parameter set for the Buddy object.

   .. attribute:: data

      A list of :class:`MetaFeature` objects. Data loaded into the Buddy object.

   .. attribute:: db_loaded

      bool. True if the database is loaded.

   .. method:: load_usi (usi_list: Union[str, List[str]], adduct_list: Union[None, str, List[str]] = None)

      Read from a single USI string or a sequence of USI strings, and load the data into the ``data`` attribute of the :class:`Buddy` object.

      :param usi_list: str or List[str]. A single USI string or a sequence of USI strings.
      :param optional adduct_list: str or List[str]. A single adduct string or a sequence of adduct strings, which will be applied to all USI strings accordingly.
      :returns: None. A list of :class:`MetaFeature` objects will be stored in the ``data`` attribute of the :class:`Buddy` object.

   .. method:: load_mgf (mgf_file: str)

      Read a single mgf file, and load the data into the ``data`` attribute of the :class:`Buddy` object.

      :param mgf_file: str. The path to the mgf file.
      :returns: None. A list of :class:`MetaFeature` objects will be stored in the ``data`` attribute of the :class:`Buddy` object.


.. class:: BuddyParamSet (ppm: bool = True, ms1_tol: float = 5, ms2_tol: float = 10, halogen: bool = False, timeout_secs: float = 300, c_range: Tuple[int, int] = (0, 80), h_range: Tuple[int, int] = (0, 150), n_range: Tuple[int, int] = (0, 20), o_range: Tuple[int, int] = (0, 30), p_range: Tuple[int, int] = (0, 10), s_range: Tuple[int, int] = (0, 15), f_range: Tuple[int, int] = (0, 20), cl_range: Tuple[int, int] = (0, 15), br_range: Tuple[int, int] = (0, 10), i_range: Tuple[int, int] = (0, 10), isotope_bin_mztol: float = 0.02, max_isotope_cnt: int = 4, ms2_denoise: bool = True, rel_int_denoise: bool = True, rel_int_denoise_cutoff: float = 0.01, max_noise_frag_ratio: float = 0.85, max_noise_rsd: float = 0.20, max_frag_reserved: int = 50, use_all_frag: bool = False)

   It is a class to store all the parameter settings for **msbuddy**.

   :param ppm: bool. If True, the mass tolerance is in ppm. If False, the mass tolerance is in Da. Default is True.
   :param ms1_tol: float. The mass tolerance for MS1 spectra. Default is 5 ppm.
   :param ms2_tol: float. The mass tolerance for MS/MS spectra. Default is 10 ppm.
   :param halogen: bool. If True, the halogen elements (F, Cl, Br, I) are considered. Default is False.
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
        timeout_secs=600)

    # create a Buddy object with the specified parameter set
    buddy = Buddy(buddy_param_set)


.. class:: Spectrum (mz_array: np.array, int_array: np.array)

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
    from msbuddy import Spectrum

    mz_array = np.array([100, 200, 300, 400, 500])
    int_array = np.array([10, 20, 30, 40, 50])
    spectrum = Spectrum(mz_array, int_array)



.. class:: Adduct (string: Union[str, None], pos_mode: bool)

    A class to represent an adduct type.

   :param optional string: str. The adduct type. Default is [M+H]+ for positive mode and [M-H]- for negative mode.
   :param pos_mode: bool. True for positive mode and False for negative mode.


   .. attribute:: string

      The adduct type.

   .. attribute:: pos_mode

      True for positive mode and False for negative mode.

   .. attribute:: charge

      The charge of the adduct.

   .. attribute:: m

      The count of M in the adduct. e.g. [M+H]+ has m=1, [2M+H]+ has m=2.




.. class:: MetaFeature (identifier: Union[str, int], mz: float, charge: int, rt: Union[float, None] = None, adduct: Union[str, None] = None, ms1: Union[Spectrum, None] = None, ms2: Union[Spectrum, None] = None)

    A class to represent a metabolic feature.

   :param identifier: str or int. A unique identifier for the metabolic feature.
   :param mz: float. Precursor ion m/z.
   :param charge: int. Precursor ion charge.
   :param optional rt: float. Retention time in seconds. Default is None.
   :param optional adduct: str. Adduct type. Default is [M+H]+ for positive mode and [M-H]- for negative mode.
   :param optional ms1: :class:`Spectrum` object. MS1 spectrum containing the isotopic pattern information. Default is None.
   :param optional ms2: :class:`Spectrum` object. MS/MS spectrum. Default is None.

   .. attribute:: identifier

      The unique identifier for the metabolic feature.

   .. attribute:: mz

      Precursor ion m/z.

   .. attribute:: charge

      Precursor ion charge.

   .. attribute:: rt

      Retention time in seconds.

   .. attribute:: adduct

      :class:`Adduct` object representing the adduct type.

   .. attribute:: ms1_raw

      :class:`Spectrum` object. Raw MS1 spectrum.

   .. attribute:: ms2_raw

      :class:`Spectrum` object. Raw MS/MS spectrum.

   .. attribute:: ms1_processed

      :class:`ProcessedMS1` object. Processed MS1 spectrum.

   .. attribute:: ms2_processed

      :class:`ProcessedMS2` object. Processed MS/MS spectrum.

   .. attribute:: candidate_formula_list

      :class: A list of `CandidateFormula` objects. Candidate formulas generated for the metabolic feature.



.. class:: ClassName(param1, param2)

   Brief description of the class and its purpose.

   :param param1: (type) Description of the first constructor parameter. Default: default_value1.
   :param param2: (type) Description of the second constructor parameter. Default: default_value2.

   .. attribute:: attribute1

      Description of the first attribute.

   .. attribute:: attribute2

      Description of the second attribute.

   .. method:: method1(arg1, arg2)

      Description of the first method.

      :param arg1: (type) Description of the first argument. Default: default_value1.
      :param arg2: (type) Description of the second argument. Default: default_value2.
      :returns: (type) Description of the return value.

   .. method:: method2(arg1)

      Description of the second method.

      :param arg1: (type) Description of the argument. Default: default_value1.
      :returns: (type) Description of the return value.

Functions
~~~~~~~~~~~~~~~
.. function:: generate_candidate_formula (meta_feature: MetaFeature, param_set: BuddyParamSet)

   Generate candidate formula for a given metabolic feature based on the given parameter set.

   :param meta_feature: :class:`MetaFeature` object.
   :param param_set: :class:`BuddyParamSet` object.
   :returns: A list of :class:`CandidateFormula` objects will be generated within the :class:`MetaFeature` object.

Example Usage:

.. code-block:: python

   # generate candidate formulas for a given metabolic feature
   generate_candidate_formula(meta_feature, param_set)

   # print all the candidate formula strings and their estimated FDRs
   for candidate_formula in meta_feature.candidate_formula_list:
      print(candidate_formula.formula.__str__() + "\t" + str(candidate_formula.estimated_fdr))
