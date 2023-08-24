Quick Start
===========

As a quick start, we here load MS/MS spectra from a mgf file, perform molecular formula annotations, and retrieve the annotation result summary:

.. code-block:: python

   from msbuddy import Buddy

   # instantiate a Buddy object
   buddy = Buddy()

   # load data, here we use a mgf file as an example
   buddy.load_mgf('input_file.mgf')

   # annotate molecular formula
   buddy.annotate_formula()

   # retrieve the annotation result summary
   result = buddy.get_summary()

   # print the result
   for key, value in result.items():
       print(key, value)


Here, ``result`` is a Python dictionary with the following keys:

- ``identifier``: Identifier of the metabolic feature
- ``mz``: Precursor ion m/z
- ``rt``: Retention time in seconds
- ``adduct``: Adduct type
- ``formula_rank_1``: Molecular formula annotation ranked in the first place
- ``estimated_fdr``: Estimated false discovery rate (FDR)


