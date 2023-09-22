Result Export
----------------

After molecular formula annotation, you can access the result summary as demonstrated in the `Quick Start <quickstart.html>`_ session.
All the top 5 formulas will be included in the ``results``.

.. code-block:: python

   # retrieve the annotation result summary
   results = buddy.get_summary()

In case you want to access all the annotation results, you can iterate through the :class:`CandidateFormula` objects stored in each :class:`MetaFeature`.

.. code-block:: python

   # retrieve all the annotation results for each metabolic feature
   for meta_feature in buddy.data:
       for i, candidate in enumerate(meta_feature.candidate_formula_list):
           print('MetaFeature mz' + str(meta_feature.mz) + '  rt: ' + str(meta_feature.rt) + \
           '  rank: ' + str(i+1) + 'Formula: ' + candidate.formula.__str__() + \
           '  estimated FDR: ' + str(candidate.estimated_fdr))



Please see :class:`MetaFeature` and :class:`CandidateFormula` in `Python API <pyapi.html>`_ for more details.


If you are using the command-line interface, the result summary will be automatically written in the output directory as a ``.tsv`` file.
To access more detailed annotation results, you can use the ``-details`` option.

