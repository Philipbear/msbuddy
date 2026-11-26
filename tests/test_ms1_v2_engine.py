import hashlib
import unittest
from unittest.mock import patch

import numpy as np

from msbuddy import Adduct, Formula, MetaFeature, Msbuddy, MsbuddyConfig, Spectrum
from msbuddy._fast_db import calc_isotope_envelope_numba, compute_ms1_features_numba
from msbuddy._ms1_v2_engine import (
    CAL_CTX_DIM,
    ISO_FEAT_DIM,
    MASS_FEAT_DIM,
    MS1_V2_FORMAT_VERSION,
    annotate_batch_ms1_v2,
    build_ms1_cal_context,
    build_ms1_v2_features,
    calibrated_ms1_log_odds,
    validate_ms1_v2_spec,
)
from msbuddy.load import _load_versioned_models
from msbuddy.utils import read_formula


def _fake_cal():
    return {
        'context_center': [0.0] * CAL_CTX_DIM,
        'context_scale': [1.0] * CAL_CTX_DIM,
        'context_clip': 6.0,
        'null_coefficients': [-1.0] + [0.0] * CAL_CTX_DIM,
        'temperature_coefficients': [0.0] + [0.0] * CAL_CTX_DIM,
    }


def _fake_ms1_v2_spec():
    iso_txt = 'tree\nversion=v4\n'
    mass_txt = 'tree\nversion=v4\n'
    return {
        'format_version': MS1_V2_FORMAT_VERSION,
        'version': '2.0.1',
        'isotope_feat_dim': ISO_FEAT_DIM,
        'mass_feat_dim': MASS_FEAT_DIM,
        'isotope_lgb_txt': iso_txt,
        'isotope_lgb_sha256': hashlib.sha256(iso_txt.encode('utf-8')).hexdigest(),
        'mass_lgb_txt': mass_txt,
        'mass_lgb_sha256': hashlib.sha256(mass_txt.encode('utf-8')).hexdigest(),
        'isotope_calibration': _fake_cal(),
        'mass_calibration': _fake_cal(),
    }


class TestMs1V2Engine(unittest.TestCase):
    def test_feature_dimensions_and_finiteness_for_ms1_and_mass_modes(self):
        f1 = Formula(read_formula('C15H12O4'), 0).array.astype(np.int16)
        f2 = Formula(read_formula('C12H10Cl2O2'), 0).array.astype(np.int16)
        c_arr = np.stack([f1, f2], axis=0)
        c_err = np.array([-0.5, 1.2], dtype=np.float32)
        adduct = Adduct('[M+H]+', True, report_invalid=True)

        X_mass, masses, abs_ppm = build_ms1_v2_features(c_arr, c_err, adduct, 257.08, True, 5.0, 2, ms1_f12=None)
        self.assertEqual(X_mass.shape, (2, MASS_FEAT_DIM))
        self.assertTrue(np.all(np.isfinite(X_mass)))
        self.assertAlmostEqual(float(abs_ppm[0]), 0.5, places=4)

        dp, sh = calc_isotope_envelope_numba(f2)
        obs_mzs = np.array([257.08, 257.08 + sh[1], 257.08 + sh[2], 257.08 + sh[3]], dtype=np.float64)
        ms1_f12 = compute_ms1_features_numba(c_arr, dp[:4], obs_mzs, 4, 1)
        X_iso, _, _ = build_ms1_v2_features(c_arr, c_err, adduct, 257.08, True, 5.0, 2, ms1_f12=ms1_f12)
        self.assertEqual(X_iso.shape, (2, ISO_FEAT_DIM))
        self.assertTrue(np.all(np.isfinite(X_iso)))
        # Pool-relative MS1 similarity gap favors the matching Cl2 candidate
        self.assertGreater(float(X_iso[1, MASS_FEAT_DIM + 2]), float(X_iso[0, MASS_FEAT_DIM + 2]))
        self.assertGreater(float(X_iso[1, MASS_FEAT_DIM + 12]), 0.0)

    def test_validate_ms1_v2_spec_rejects_tampering_and_bad_dims(self):
        spec = _fake_ms1_v2_spec()
        validate_ms1_v2_spec(spec)
        with self.assertRaisesRegex(ValueError, 'checksum'):
            validate_ms1_v2_spec(dict(spec, isotope_lgb_txt='tampered'))
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            validate_ms1_v2_spec(dict(spec, mass_feat_dim=MASS_FEAT_DIM - 1))
        bad_cal = _fake_cal()
        bad_cal['context_scale'][0] = 0.0
        with self.assertRaisesRegex(ValueError, 'context_scale'):
            validate_ms1_v2_spec(dict(spec, isotope_calibration=bad_cal))

    def test_calibration_preserves_absent_mass_and_ordering(self):
        scores = np.array([2.5, 0.5, -1.0], dtype=np.float64)
        abs_ppm = np.array([0.4, 2.0, 3.5], dtype=np.float32)
        ctx = build_ms1_cal_context(scores, abs_ppm, 5.0, 3, 300.1, 1, False, 0, None)
        self.assertEqual(ctx.shape, (CAL_CTX_DIM,))
        lo = calibrated_ms1_log_odds(scores, ctx, _fake_cal())
        self.assertTrue(np.all(np.diff(lo) < 0))
        all_s = np.append(lo, 0.0)
        probs = np.exp(all_s - np.log(np.exp(all_s).sum()))
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=6)
        self.assertGreater(float(probs[-1]), 0.0)

    def test_annotate_batch_ms1_v2_supports_ms1_mass_multicharge_da_and_auto(self):
        f_glu = Formula(read_formula('C6H12O6'), 0)
        f_alt = Formula(read_formula('C5H12N2O5'), 0)
        arrs = np.stack([f_glu.array.astype(np.int64), f_alt.array.astype(np.int64)], axis=0)
        masses = np.array([f_glu.mass, f_alt.mass], dtype=np.float64)
        order = np.argsort(masses)
        masses = masses[order]
        arrs = arrs[order]
        gd = {
            'basic_db_mass': masses,
            'basic_db_formula': arrs,
            'basic_db_idx': np.searchsorted(masses, np.arange(15000) / 10.0),
            'halogen_db_mass': np.array([], dtype=np.float64),
            'halogen_db_formula': np.empty((0, 12), dtype=np.int64),
            'halogen_db_idx': np.zeros(15000, dtype=np.int64),
            'common_frag_db': np.empty((0, 12), dtype=np.int16),
            'common_loss_db': np.empty((0, 12), dtype=np.int16),
            'model_ms1_v2': _fake_ms1_v2_spec(),
            'model_ms1_v2_cache': {},
        }

        class DummyBooster:
            def __init__(self, expected_dim):
                self.expected_dim = expected_dim
                self.calls = 0

            def predict(self, X, num_threads=1):
                self.calls += 1
                assert X.shape[1] == self.expected_dim
                # Higher score for lower abs_ppm (col 71)
                return -X[:, 71].astype(np.float64)

        iso_b = DummyBooster(ISO_FEAT_DIM)
        mass_b = DummyBooster(MASS_FEAT_DIM)
        gd['model_ms1_v2_cache'] = {
            'isotope': iso_b,
            'mass': mass_b,
            'isotope_cal': _fake_cal(),
            'mass_cal': _fake_cal(),
        }

        ion1 = Adduct('[M+H]+', True)
        mz1 = f_glu.mass + ion1.net_formula.mass - 0.00054858
        dp, sh = calc_isotope_envelope_numba((f_glu.array + ion1.net_formula.array).astype(np.int16))
        ms1_spec = Spectrum([mz1, mz1 + sh[1], mz1 + sh[2]], [dp[0] * 100.0, dp[1] * 100.0, dp[2] * 100.0])

        ion2 = Adduct('[M+2H]2+', True)
        mz2 = (f_glu.mass + ion2.net_formula.mass - 2 * 0.00054858) / 2.0

        features = [
            MetaFeature('ms1_single', mz1, 1, adduct='[M+H]+', ms1=ms1_spec),
            MetaFeature('mass_single', mz1, 1, adduct='[M+H]+'),
            MetaFeature('mass_double', mz2, 2, adduct='[M+2H]2+'),
            MetaFeature('auto_ms1', mz1, 1, adduct='auto', ms1=ms1_spec),
        ]
        cfg = MsbuddyConfig(ppm=False, ms1_tol=0.05, halogen=False)
        annotate_batch_ms1_v2(features, cfg, gd)

        self.assertEqual(iso_b.calls, 1)
        self.assertEqual(mass_b.calls, 1)
        for mf in features:
            self.assertTrue(mf.candidate_formula_list, mf.identifier)
            self.assertEqual(str(mf.candidate_formula_list[0].formula), 'C6H12O6')
            self.assertGreater(mf.candidate_formula_list[0].normed_estimated_prob, 0.0)
            self.assertLess(mf.candidate_formula_list[0].normed_estimated_prob, 1.0)
        self.assertEqual(features[3].adduct.string, '[M+H]+')

    def test_load_versioned_models_loads_optional_ms1_v2_from_compat_bundle(self):
        from tests.test_compact_bundle import compatibility_bundle
        compat = compatibility_bundle()
        compat['ms1_v2'] = _fake_ms1_v2_spec()
        deps = {}
        _load_versioned_models(compat, deps)
        self.assertIn('model_ms1_v2', deps)
        self.assertIn('model_ms1_v2_cache', deps)


if __name__ == '__main__':
    unittest.main()
