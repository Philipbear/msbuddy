"""Check model selection for missing, empty and usable MS1/MS2 data."""

import unittest
from types import SimpleNamespace

import numpy as np

from msbuddy.base import CandidateFormula, Formula, MetaFeature, Spectrum
from msbuddy.ml import predict_formula_probability
from msbuddy.utils import read_formula


class SpectrumModeTests(unittest.TestCase):
    def test_multicharged_precursor_keeps_high_mz_fragments(self):
        for charge in (2, -2):
            with self.subTest(charge=charge):
                feature = MetaFeature('double', 90, charge,
                                      ms2=Spectrum([40, 90, 91, 130, 180, 181],
                                                   [10, 100, 30, 20, 10, 10]))
                feature.data_preprocess(True, 5, 10, 0.02, 4, 0.01, 6)
                np.testing.assert_array_equal(feature.ms2_processed.mz_array, [40, 130])
                np.testing.assert_array_equal(feature.ms2_processed.idx_array, [0, 3])

    def test_custom_feature_infers_a_multiply_charged_default_adduct(self):
        for charge, expected in ((1, '[M+H]+'), (-1, '[M-H]-'),
                                 (2, '[M+2H]2+'), (-2, '[M-2H]2-'), (3, '[M+3H]3+')):
            with self.subTest(charge=charge):
                feature = MetaFeature('feature', 100, charge)
                self.assertEqual((feature.adduct.string, feature.adduct.charge), (expected, charge))

    def test_mismatched_adduct_charge_is_reported(self):
        with self.assertLogs(level='WARNING') as records:
            feature = MetaFeature('mismatch', 100, 2, adduct='[M+H]+')
        self.assertEqual(feature.adduct.charge, 1)
        self.assertIn('differs from adduct', records.output[0])

    def test_nonintegral_charge_is_rejected(self):
        for charge in (1.5, float('nan'), float('inf')):
            with self.subTest(charge=charge), self.assertRaisesRegex(ValueError, 'nonzero integer'):
                MetaFeature('bad', 100, charge)

    def test_reprocessing_clears_previous_spectra_when_raw_input_is_removed(self):
        feature = MetaFeature('reused', 181.0707, 1,
                              ms1=Spectrum([181.0707, 182.074055], [100, 7]),
                              ms2=Spectrum([40, 70], [5, 8]))
        feature.data_preprocess(True, 5, 10, 0.02, 4, 0.01, 6)
        self.assertTrue(feature.ms1_processed)
        self.assertTrue(feature.ms2_processed)
        feature.ms1_raw = None
        feature.ms2_raw = Spectrum([], [])
        feature.data_preprocess(True, 5, 10, 0.02, 4, 0.01, 6)
        self.assertIsNone(feature.ms1_processed)
        self.assertIsNone(feature.ms2_processed)

    def test_models_follow_usable_spectra_after_preprocessing(self):
        mz = 181.0707
        ms1 = Spectrum([mz, mz + 1.003355], [100, 7])
        single_ms1 = Spectrum([mz], [100])
        unmatched_ms1 = Spectrum([100, 101], [100, 7])
        zero_m0_ms1 = Spectrum([mz, mz + 1.003355], [0, 7])
        valid_ms2 = Spectrum([40, 70, 100], [4, 8, 12])
        precursor_ms2 = Spectrum([mz], [100])
        zero_ms2 = Spectrum([40, 70], [0, 0])
        invalid_ms2 = Spectrum([40, 70], [-5, float('nan')])
        mixed_ms2 = Spectrum([100, 40, 70], [-2, 4, 0])
        cases = [
            ('neither', None, None, 3),
            ('empty', Spectrum([], []), Spectrum([], []), 3),
            ('ms1_only', ms1, None, 1),
            ('ms2_only', None, valid_ms2, 2),
            ('both', ms1, valid_ms2, 0),
            ('unmatched_ms1', unmatched_ms1, None, 3),
            ('one_isotope', single_ms1, None, 3),
            ('precursor_only', None, precursor_ms2, 3),
            ('zero_intensity', None, zero_ms2, 3),
            ('invalid_intensity', None, invalid_ms2, 3),
            ('ms1_empty_ms2', ms1, precursor_ms2, 1),
            ('zero_m0', zero_m0_ms1, None, 3),
            ('mixed_intensities', None, mixed_ms2, 2),
        ]
        features = []
        for name, ms1_raw, ms2_raw, _ in cases:
            mf = MetaFeature(name, mz, 1, ms1=ms1_raw, ms2=ms2_raw)
            mf.data_preprocess(True, 5, 10, 0.02, 4, 0.01, 6)
            mf.candidate_formula_list = [CandidateFormula(
                Formula(read_formula('C6H12O6'), 0), Formula(read_formula('C6H13O6'), 1))]
            features.append(mf)

        class Model:
            def __init__(self, group):
                self.group = group
                self.shapes = []

            def predict(self, matrix):
                self.shapes.append(matrix.shape)
                return np.full(len(matrix), self.group, dtype=float)

        names = ['model_ms1_ms2', 'model_ms1_noms2', 'model_noms1_ms2', 'model_noms1_noms2']
        models = [Model(group) for group in range(4)]
        gd = dict(zip(names, models))
        for group in range(4):
            gd[f'platt_a_{group}'] = 1.0
            gd[f'platt_b_{group}'] = 0.0
        config = SimpleNamespace(ppm=True, ms1_tol=5, ms2_tol=10)
        predict_formula_probability(features, 0, len(features), config, gd)

        for feature, (_, _, _, group) in zip(features, cases):
            expected = 1 / (1 + np.exp(-group))
            self.assertAlmostEqual(feature.candidate_formula_list[0].estimated_prob, expected,
                                   msg=feature.identifier)
        self.assertEqual([model.shapes[0][1] for model in models], [53, 29, 52, 28])
        self.assertFalse(features[8].ms2_processed)
        self.assertFalse(features[9].ms2_processed)
        np.testing.assert_array_equal(features[12].ms2_processed.mz_array, [40])
        np.testing.assert_array_equal(features[12].ms2_processed.idx_array, [0])


if __name__ == '__main__':
    unittest.main()
