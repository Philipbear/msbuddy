"""Validate split integrity, label-blind inputs, and candidate-retention replay."""

import unittest
from types import SimpleNamespace

import numpy as np

from benchmarks.pruning_ablation import retained_truth
from benchmarks.library_sample import instrument_family as sample_instrument_family
from benchmarks.train_ranker import (encode_metadata, encode_spectrum, formula_split,
                                      rank_metrics, training_indices)
from msbuddy.cand import _retain_top_cand_form


class ResearchRankerTests(unittest.TestCase):
    def test_formula_group_is_independent_of_formula_string_order(self):
        self.assertEqual(formula_split('C6H12O6'), formula_split('H12O6C6'))
        groups = {formula_split('C{}H{}O2'.format(c, 2 * c)) for c in range(2, 100)}
        self.assertEqual(groups, {'train', 'validation', 'test'})

    def test_model_inputs_do_not_depend_on_reference_labels(self):
        record = {'identifier': 'example', 'precursor_mz': 181.070665,
                  'mzs': [61.0284, 91.03897, float('nan')], 'intensities': [100, 50, 1],
                  'adduct': '[M+H]+', 'polarity': 'positive', 'instrument': 'orbitrap',
                  'collision_energy_ev': [20, 30], 'formula': 'C6H12O6'}
        changed = {**record, 'formula': 'C8H16O4', 'structure_group': 'a different molecule'}
        np.testing.assert_array_equal(encode_spectrum(record), encode_spectrum(changed))
        np.testing.assert_array_equal(encode_metadata(record), encode_metadata(changed))
        self.assertEqual(encode_spectrum(record).shape, (2, 7500))

    def test_tof_instruments_use_one_research_metadata_feature(self):
        record = {'precursor_mz': 181.070665, 'adduct': '[M+H]+', 'polarity': 'positive'}
        expected = encode_metadata({**record, 'instrument': 'tof'})
        self.assertEqual(expected.shape, (7,))
        self.assertEqual(expected[-1], 1)
        for instrument in ('qtof', 'timstof', 'Q-TOF', 'timsTOF'):
            np.testing.assert_array_equal(
                encode_metadata({**record, 'instrument': instrument}), expected)
            self.assertEqual(sample_instrument_family(instrument), 'tof')
        self.assertEqual(sample_instrument_family('FTMS'), 'orbitrap')
        self.assertEqual(sample_instrument_family('FTICR'), 'unknown')

    def test_negative_sampling_preserves_the_positive_and_has_no_duplicates(self):
        row = {'proposal_candidates': list(map(str, range(500))), 'formula': '499',
               'proposal_features': np.zeros((500, 16)).tolist()}
        indices = training_indices(row, np.random.default_rng(17), 64)
        self.assertEqual(len(indices), 64)
        self.assertEqual(len(set(indices)), 64)
        self.assertIn(499, indices)

    def test_rank_metrics_include_failed_requests_in_operational_accuracy(self):
        rows = [{'formula': 'B', 'proposal_candidates': ['A', 'B']},
                {'status': 'timeout'}]
        metrics = rank_metrics(rows, [np.array([0, 1]), None])
        self.assertEqual((metrics['attempted'], metrics['completed']), (2, 1))
        self.assertEqual(metrics['top1_all_attempts'], .5)
        self.assertEqual(metrics['top1_given_candidate'], 1)

    def test_pruning_replay_matches_actual_selection(self):
        names = [str(index) for index in range(350)] + ['truth']
        features = np.zeros((351, 16))
        features[:350, 13] = .1
        features[350, 13] = .9
        features[350, 15] = 1
        row = {'formula': 'truth', 'proposal_candidates': names,
               'proposal_features': features.tolist(), 'processed_peaks': 3}
        for preserve in (False, True):
            candidates = [SimpleNamespace(name=name, mz_error=0.0,
                                           exp_ms2_sum_int=float(features[index, 13]) *
                                           (1 if preserve else 1 - features[index, 15]))
                          for index, name in enumerate(names)]
            actual = _retain_top_cand_form(candidates, 5, False, True, 350)
            self.assertEqual(retained_truth(row, 5, preserve_evidence=preserve),
                             any(candidate.name == 'truth' for candidate in actual))


if __name__ == '__main__':
    unittest.main()
