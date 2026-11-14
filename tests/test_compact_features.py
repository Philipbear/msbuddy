import numpy as np
import unittest

from msbuddy._compact_features import candidate_matrix, fluorine_log_probability


def check_hurdle_distribution_normalizes_and_supports_large_counts():
    for raw in [[-3., -2., -1.], [0., 2., 1.], [3., 15., 3.]]:
        counts = np.arange(1000)
        probability = np.exp(fluorine_log_probability(raw, counts))
        assert np.isclose(probability.sum(), 1., atol=1e-10)
        assert np.isclose(probability[0], 1 / (1 + np.exp(raw[0])))
        assert np.isfinite(fluorine_log_probability(raw, 80))
    assert np.isfinite(fluorine_log_probability([-100., -100., -100.], 80))


def check_shared_modes_have_identical_nonisotope_features():
    evidence = np.arange(2 * 94).reshape(2, 94).astype(float)
    counts = np.ones((2, 12));counts[1, 4] = 40
    args = (counts, np.ones(24), np.zeros(64), np.ones(128), np.zeros(72), np.zeros(12), np.ones(12))
    ms2 = candidate_matrix(evidence, *args, 2, 0)
    joint = candidate_matrix(np.column_stack([np.zeros(2), evidence]), *args, 0, 4)
    assert ms2.shape == joint.shape == (2, 435)
    assert np.array_equal(ms2[:, :-3], joint[:, :-3])
    assert np.array_equal(ms2[:, -3:], [[0, 1, 0], [0, 1, 0]])
    assert np.array_equal(joint[:, -3:], [[1, 1, 1], [1, 1, 1]])
    assert np.isfinite(ms2).all()


def check_invalid_counts_and_evidence_are_rejected():
    args = (np.ones((1, 12)), np.zeros(24), np.zeros(64), np.zeros(128), np.zeros(72), np.zeros(12), np.ones(12))
    with np.testing.assert_raises(ValueError):candidate_matrix(np.zeros((1, 93)), *args, 2, 0)
    args[0][0, 4] = -1
    with np.testing.assert_raises(ValueError):candidate_matrix(np.zeros((1, 94)), *args, 2, 0)


class CompactFeatureTests(unittest.TestCase):
    test_distribution = staticmethod(check_hurdle_distribution_normalizes_and_supports_large_counts)
    test_shared_inputs = staticmethod(check_shared_modes_have_identical_nonisotope_features)
    test_invalid_inputs = staticmethod(check_invalid_counts_and_evidence_are_rejected)
