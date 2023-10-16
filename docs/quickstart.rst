Quick Start
===========

As a quick start, we here load MS/MS spectra from a mgf file, perform molecular formula annotations, and retrieve the annotation result summary:

.. code-block:: python

   from msbuddy import Msbuddy, MsbuddyConfig

   # create a MsbuddyConfig object
   msb_config = MsbuddyConfig(ms_instr="orbitrap", # supported: "qtof", "orbitrap" and "fticr"
                                                   # highly recommended to fill in the instrument type
                              halogen=True,
                              timeout_secs=200)

   # instantiate a Msbuddy object with the parameter set
   msb_engine = Msbuddy(msb_config)

   # load data, here we use a mgf file as an example
   msb_engine.load_mgf('input_file.mgf')

   # annotate molecular formula
   msb_engine.annotate_formula()

   # retrieve the annotation result summary
   results = msb_engine.get_summary()

   # print the result, results is a list of dictionaries
   for individual_result in results:
       for key, value in individual_result.items():
           print(key, value)


.. note::
    It is **highly recommended** to set up the ``ms_instr`` parameter in the :class:`msbuddy.MsbuddyConfig` to obtain the best annotation performance.
    Please see `Configuration <config.html>`_ session for more details.



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


MS/MS spectra can also be loaded via their USIs:

.. code-block:: python

   # you can load multiple USIs at once
   msb_engine.load_usi(['mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00003740036',
                        'mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00003740037'])

   # load USIs with adducts specified, otherwise the default adducts ([M+H]+, [M-H]-) will be used
   msb_engine.load_usi(usi_list=['mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00003740036',
                                 'mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00000845027'],
                       adduct_list=['[M+H]+', '[M-H2O+H]+'])

.. note::
    **msbuddy** does not perform adduct annotation. Please make sure the adduct type is correctly specified in the input file if necessary, otherwise default adducts ([M+H]+, [M-H]-) will be used.
    We claim that adduct annotation should be performed on the MS1 level, where chromatographic peak profiles must be involved.


If parallel computing is needed, you can specify the number of CPUs to be used, but the code has to be run in ``if __name__ == '__main__':`` block:
.. code-block:: python

   if __name__ == '__main__':
       from msbuddy import Msbuddy, MsbuddyConfig
       # create a MsbuddyConfig object
       msb_config = MsbuddyConfig(ms_instr="orbitrap", # supported: "qtof", "orbitrap" and "fticr"
                                                       # highly recommended to fill in the instrument type
                                  halogen=True,
                                  parallel=True, # enable parallel computing
                                  n_cpu=12) # number of CPUs to be used
       ...(other code remains the same)