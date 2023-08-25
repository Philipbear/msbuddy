Result Export
----------------

After molecular formula annotation, you can access the result summary as demonstrated in the `Quick Start <quickstart.rst>`_ session.
All the first-ranked formulas will be included in the ``result``.

.. code-block:: python

   # retrieve the annotation result summary
   result = buddy.get_summary()

In case you want to access all the annotation results, you can iterate through the :class:`CandidateFormula` objects stored in each :class:`MetaFeature`.

.. code-block:: python
    # retrieve all the annotation results for each metabolic feature
    for meta_feature in buddy.data:
        for i, candidate in enumerate(meta_feature.candidate_formula_list):
            print('MetaFeature:  mz' + str(meta_feature.mz) + '  rt: ' + str(meta_feature.rt) + '  rank: ' + str(i+1) + \
            'Formula: ' + candidate.formula.__str__() + '  estimated FDR: ' + str(candidate.estimated_fdr))


Please see :class:`MetaFeature` and :class:`CandidateFormula` for more details.
