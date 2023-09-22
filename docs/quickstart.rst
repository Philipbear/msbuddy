Quick Start
===========

As a quick start, we here load MS/MS spectra from a mgf file, perform molecular formula annotations, and retrieve the annotation result summary:

.. code-block:: python

   from msbuddy import Buddy, BuddyParamSet

   # create a parameter set
   buddy_param_set = BuddyParamSet(ms_instr="orbitrap", # supported: "qtof", "orbitrap" and "fticr"
                                                        # highly recommended to fill in the instrument type
                                   halogen=True,
                                   parallel=True,
                                   n_cpu=12,
                                   timeout_secs=200)

   # instantiate a Buddy object with the parameter set
   buddy = Buddy(buddy_param_set)

   # load data, here we use a mgf file as an example
   buddy.load_mgf('input_file.mgf')

   # annotate molecular formula
   buddy.annotate_formula()

   # retrieve the annotation result summary
   results = buddy.get_summary()

   # print the result, results is a list of dictionaries
   for individual_result in results:
       for key, value in individual_result.items():
           print(key, value)


It is **highly recommended** to set up the ``ms_instr`` parameter in the :class:`msbuddy.BuddyParamSet` to obtain the best annotation performance.
Please see `Parameter Settings <paramset.html>`_ session for more details.



Within the result summary, ``results`` is a list of Python dictionaries. ``individual_result`` is a dictionary containing the following keys:

- ``identifier``: Identifier of the metabolic feature
- ``mz``: Precursor m/z
- ``rt``: Retention time in seconds
- ``adduct``: Adduct type
- ``formula_rank_1``: Molecular formula annotation ranked in the first place
- ``estimated_fdr``: Estimated false discovery rate (FDR)
- ``formula_rank_2``: Molecular formula annotation ranked in the second place
- ``formula_rank_3``: Molecular formula annotation ranked in the third place
- ``formula_rank_4``: Molecular formula annotation ranked in the fourth place
- ``formula_rank_5``: Molecular formula annotation ranked in the fifth place

