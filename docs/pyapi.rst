Python API
-------------

.. class:: Spectrum (mz_array: np.array, int_array: np.array)

    A class to represent a mass spectrum.

   :param mz_array: A numpy array of m/z values.
   :param int_array: A numpy array of intensity values.

   .. attribute:: mz_array

      A numpy array of m/z values.

   .. attribute:: int_array

      A numpy array of intensity values.


Example usage:

.. code-block::

    import numpy as np
    from msbuddy import Spectrum

    mz_array = np.array([100, 200, 300, 400, 500])
    int_array = np.array([1, 2, 3, 4, 5])
    spectrum = Spectrum(mz_array, int_array)

    print(spectrum.mz_array)
    print(spectrum.int_array)



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


.. function:: generate_candidate_formula (meta_feature: MetaFeature, param_set: BuddyParamSet)

   Generate candidate formula for a given metabolic feature.

   :param meta_feature: :class:`MetaFeature` object.
   :param param_set: :class:`BuddyParamSet` object.
   :returns: A list of :class:`CandidateFormula` objects will be generated within the :class:`MetaFeature` object.

Example Usage:

.. code-block:: python

   # generate candidate formulas for a given metabolic feature
   generate_candidate_formula(meta_feature, param_set)

   # print all the candidate formula strings
   for candidate_formula in meta_feature.candidate_formula_list:
      print(candidate_formula.formula.__str__())
