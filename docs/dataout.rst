Result Export
----------------

After molecular formula annotation, you can access the result summary as demonstrated in the `Quick Start <quickstart.html>`_ session.
All the top 5 formulas will be included in the ``results``.

.. code-block:: python

   # retrieve the annotation result summary
   results = msb_engine.get_summary()

The summary includes ``posterior_error_probability`` for each top formula and
``batch_q_value`` pooled across the full loaded collection. To select annotations
at a desired model-estimated false fraction:

.. code-block:: python

   confidence = msb_engine.estimate_batch_fdr(fdr_threshold=0.05)
   selected = [row for row, keep in zip(results, confidence.selected) if keep]

The individual error probability and pooled q-value answer different questions.
The legacy ``estimated_fdr`` is a within-spectrum candidate-prefix estimate;
at rank one it equals the individual error probability. These quantities depend
on calibrated model probabilities and do not guarantee FDR under arbitrary
changes in chemistry or acquisition conditions. Compute pooled q-values once
over the collection you intend to report, rather than separately per processing
chunk. Empty candidate sets are excluded from selections and have missing
q-values. Candidate detail files contain individual probabilities; global
q-values are in the summary file.

In case you want to access all the annotation results, you can iterate through the :class:`msbuddy.base.CandidateFormula` objects stored in each :class:`msbuddy.base.MetaFeature`.

.. code-block:: python

   # retrieve all the annotation results for each metabolic feature
   for meta_feature in msb_engine.data:
       for i, candidate in enumerate(meta_feature.candidate_formula_list):
           print('MetaFeature mz' + str(meta_feature.mz) + '  rt: ' + str(meta_feature.rt) + \
           '  rank: ' + str(i+1) + 'Formula: ' + candidate.formula.__str__() + \
           '  estimated FDR: ' + str(candidate.estimated_fdr))



Please see :class:`msbuddy.base.MetaFeature` and :class:`msbuddy.base.CandidateFormula` in `Python API <pyapi.html>`_ for more details.

.. note::
    If you are using the command-line interface, the result summary will be automatically written in the output directory as a ``.tsv`` file.
    To access more detailed annotation results, you can use the ``-details`` option. See `Command-line API <cmdapi.html>`_ for more details.

