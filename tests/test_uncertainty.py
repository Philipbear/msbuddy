"""Regression tests for preserving uncertainty outside the candidate pool."""
from types import SimpleNamespace
import unittest

import numpy as np

from msbuddy.ml import calc_fdr, _platt_calibrated_probability


class UncertaintyTests(unittest.TestCase):
    def estimate(self, probabilities):
        feature = SimpleNamespace(candidate_formula_list=[SimpleNamespace(estimated_prob=p)
                                                         for p in probabilities])
        calc_fdr([feature], 0, 1)
        return feature.candidate_formula_list

    def test_single_weak_candidate_does_not_become_certain(self):
        candidate, = self.estimate([.01])
        self.assertAlmostEqual(candidate.normed_estimated_prob, .01)
        self.assertAlmostEqual(candidate.estimated_fdr, .99)

    def test_missing_candidate_probability_mass_is_preserved(self):
        candidates = self.estimate([.1, .2])
        self.assertAlmostEqual(sum(c.normed_estimated_prob for c in candidates), .3)
        self.assertAlmostEqual(candidates[0].estimated_fdr, .8)
        self.assertAlmostEqual(candidates[1].estimated_fdr, .85)

    def test_incoherent_legacy_probabilities_are_only_scaled_down(self):
        candidates = self.estimate([.8, .7])
        self.assertAlmostEqual(sum(c.normed_estimated_prob for c in candidates), 1.)
        self.assertAlmostEqual(candidates[1].estimated_fdr, .5)

    def test_zero_probabilities_do_not_produce_nan(self):
        self.assertTrue(all(c.estimated_fdr == 1 for c in self.estimate([0., 0.])))

    def test_invalid_probabilities_are_rejected(self):
        for value in [np.nan, np.inf, -.01, 1.01]:
            with self.assertRaises(ValueError):
                self.estimate([value])

    def test_calibration_is_finite_for_extreme_scores(self):
        with np.errstate(over='raise'):
            result = _platt_calibrated_probability(np.array([-1e6, 1e6]), 1., 0.)
        np.testing.assert_array_equal(result, [0., 1.])


if __name__ == '__main__':
    unittest.main()
