"""Batch error semantics, probability preservation and processing invariance."""
import importlib
import itertools
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from msbuddy import estimate_batch_fdr
from msbuddy.base import CandidateFormula, Formula, MetaFeature
from msbuddy.ml import calc_fdr


def feature(identifier, probability):
    query = MetaFeature(identifier, 100., 1)
    query.candidate_formula_list = []
    if probability is not None:
        candidate = CandidateFormula(Formula.from_string('C6H6'))
        candidate.estimated_prob = probability
        query.candidate_formula_list = [candidate]
        calc_fdr([query], 0, 1)
    return query


class BatchConfidenceTests(unittest.TestCase):
    def test_expected_fdp_matches_enumeration_of_truth_outcomes(self):
        probabilities = [.99, .97, .90, .30]
        result = estimate_batch_fdr(probabilities, .05)
        self.assertEqual(result.selected, (True, True, True, False))
        expectation = 0.
        for truth in itertools.product([False, True], repeat=4):
            weight = np.prod([p if correct else 1-p for p, correct in zip(probabilities, truth)])
            fdp = sum(selected and not correct for selected, correct in zip(result.selected, truth)) / 3
            expectation += weight * fdp
        self.assertAlmostEqual(result.estimated_fdr, expectation)
        self.assertAlmostEqual(result.expected_false_discoveries, .14)
        self.assertAlmostEqual(result.q_values[2], .14/3)
        self.assertGreater(result.posterior_error_probabilities[2], .05)

    def test_ties_are_complete_and_permutation_invariant(self):
        probabilities = [.99, .9, .9, .9]
        reference = estimate_batch_fdr(probabilities, .06)
        self.assertEqual(reference.selected, (True, False, False, False))
        self.assertEqual(len(set(reference.q_values[1:])), 1)
        for order in itertools.permutations(range(4)):
            result = estimate_batch_fdr([probabilities[i] for i in order], .06)
            for i, j in enumerate(order):
                self.assertEqual(result.q_values[i], reference.q_values[j])
                self.assertEqual(result.selected[i], reference.selected[j])

    def test_missing_queries_and_weak_probability_are_not_promoted(self):
        result = estimate_batch_fdr([None, .01, None])
        self.assertEqual(result.q_values, (None, .99, None))
        self.assertEqual(result.selected_count, 0)
        self.assertEqual(result.eligible_count, 1)
        self.assertEqual(result.estimated_fdr, 0.)
        self.assertEqual(json.loads(json.dumps(result.to_dict()))['q_values'], [None, .99, None])
        self.assertEqual(estimate_batch_fdr([]).selected_count, 0)

    def test_boundaries_and_input_are_preserved(self):
        p = [.95, None, .95]
        self.assertEqual(estimate_batch_fdr(p).selected, (True, False, True))
        self.assertEqual(p, [.95, None, .95])
        self.assertEqual(estimate_batch_fdr([1., .999, None], 0.).selected, (True, False, False))
        self.assertEqual(estimate_batch_fdr([0., None], 1.).selected, (True, False))

    def test_invalid_values_fail(self):
        for value in [True, np.bool_(False), float('nan'), float('inf'), -.1, 1.1, '.95']:
            with self.subTest(value=value):
                with self.assertRaises(ValueError): estimate_batch_fdr([value])
                with self.assertRaises(ValueError): estimate_batch_fdr([.9], value)
        with self.assertRaises(ValueError): estimate_batch_fdr('.99')

    def test_engine_matches_helper_without_mutating_candidates(self):
        main = importlib.import_module('msbuddy.main')
        engine = object.__new__(main.Msbuddy)
        engine.data = [feature(str(i), p) for i, p in enumerate([.99, None, .90])]
        before = [(id(q.candidate_formula_list), [c.normed_estimated_prob for c in q.candidate_formula_list]) for q in engine.data]
        result = engine.estimate_batch_fdr(.06)
        self.assertEqual(result, estimate_batch_fdr([.99, None, .90], .06))
        summary = engine.get_summary()
        self.assertEqual([r['batch_q_value'] for r in summary], list(result.q_values))
        self.assertEqual([r['posterior_error_probability'] for r in summary], list(result.posterior_error_probabilities))
        after = [(id(q.candidate_formula_list), [c.normed_estimated_prob for c in q.candidate_formula_list]) for q in engine.data]
        self.assertEqual(before, after)
        engine.data[0].candidate_formula_list[0].normed_estimated_prob = None
        with self.assertRaises(ValueError): engine.estimate_batch_fdr()

    def test_cli_q_values_do_not_depend_on_processing_chunks(self):
        main = importlib.import_module('msbuddy.main')
        probabilities = [.99, .90, None, .98, .9]
        expected = estimate_batch_fdr(probabilities).q_values
        for batch_size in [1, 2, 5]:
            engine = object.__new__(main.Msbuddy)
            engine.config = SimpleNamespace(data_dir=None, data_files=None, batch_size=batch_size, parallel=False)
            engine._dependencies = {}
            engine.data = [feature(str(i), p) for i, p in enumerate(probabilities)]
            def annotate_batch(batch, total):
                return batch * batch_size, min((batch + 1) * batch_size, len(engine.data))
            with tempfile.TemporaryDirectory() as directory:
                with patch.object(main, 'init_ml_models', side_effect=lambda gd, **kwargs: gd), \
                     patch.object(engine, '_annotate_formula_main_batch', side_effect=annotate_batch):
                    engine.annotate_formula_cmd(Path(directory))
                result = pd.read_csv(Path(directory)/'msbuddy_result_summary.tsv', sep='\t')
            np.testing.assert_allclose(result['batch_q_value'], [np.nan if q is None else q for q in expected])
            self.assertIsNone(engine.data)


if __name__ == '__main__':
    unittest.main()
