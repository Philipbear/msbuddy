import hashlib
import unittest

import numpy as np

from msbuddy import Formula, MetaFeature, Spectrum
from msbuddy._fast_db import (
    build_query_meta_60,
    calc_isotope_envelope_numba,
    common_nl_lookup_fast,
    compute_ms1_features_numba,
    general_chemistry_summary_numba,
    mass_exists_fast,
    pack_common_key,
)
from msbuddy._v2_engine import (
    CAND_FEAT_DIM,
    POOL_FEAT_DIM,
    QUERY_META_60_DIM,
    featurize_peaks_v2,
    validate_v2_bundle,
)
from msbuddy.cand import calc_isotope_pattern
from msbuddy.utils import form_arr_to_str, read_formula


class TestMsbuddyV2Components(unittest.TestCase):
    def test_numba_isotope_envelope_matches_brainpy(self):
        for formula_str in (
            'C20H30O4',
            'C15H10Cl2O4',
            'C12H8Br2O',
            'C8HF15O2',
            'C25H30N3O5PS2',
            'C10H14I2O2',
            'C20H30O5K',
        ):
            f = Formula(read_formula(formula_str), 0)
            bp = calc_isotope_pattern(f, iso_peaks=4)
            bp = bp / bp.sum()
            dp, dm = calc_isotope_envelope_numba(f.array.astype(np.int16))
            dp4 = dp[:4] / dp[:4].sum()
            self.assertLess(float(np.max(np.abs(bp - dp4))), 1e-4, msg=formula_str)
            self.assertGreater(float(dm[1]), 0.99)
            self.assertGreater(float(dm[2]), 1.99)

    def test_ms1_features_zero_when_absent_and_discriminates_chlorine(self):
        f_ch = Formula(read_formula('C15H12O4'), 0).array.astype(np.int16)
        f_cl = Formula(read_formula('C12H10Cl2O2'), 0).array.astype(np.int16)
        cands = np.stack([f_ch, f_cl], axis=0)
        zeros = np.zeros(4, dtype=np.float64)
        out_empty = compute_ms1_features_numba(cands, zeros, zeros, 0, 1)
        self.assertTrue(np.all(out_empty == 0.0))

        # True Cl2 envelope: strong M+2 peak (~0.65 of M0) at +1.9973 Da
        dp_cl, sh_cl = calc_isotope_envelope_numba(f_cl)
        obs_mzs = np.array([300.0, 300.0 + sh_cl[1], 300.0 + sh_cl[2], 300.0 + sh_cl[3]], dtype=np.float64)
        out_cl = compute_ms1_features_numba(cands, dp_cl[:4], obs_mzs, 4, 1)
        self.assertEqual(float(out_cl[1, 0]), 1.0)
        self.assertGreater(float(out_cl[1, 2]), 0.99)  # Cl2 similarity ~ 1.0
        self.assertLess(float(out_cl[0, 2]), 0.60)     # CHO mismatch

    def test_mass_exists_and_common_nl_lookup(self):
        masses = np.array([18.010565, 44.026215, 180.063388, 300.123456], dtype=np.float64)
        idx = np.searchsorted(masses, np.arange(4000) / 10.0).astype(np.int64)
        self.assertTrue(mass_exists_fast(masses, idx, 180.06340, 1e-4))
        self.assertFalse(mass_exists_fast(masses, idx, 180.06500, 1e-4))

        h2o = Formula(read_formula('H2O'), 0).array.astype(np.int16)
        co2 = Formula(read_formula('CO2'), 0).array.astype(np.int16)
        ch4 = Formula(read_formula('CH4'), 0).array.astype(np.int16)
        keys = np.sort(np.array([pack_common_key(h2o), pack_common_key(co2)], dtype=np.uint64))
        self.assertTrue(common_nl_lookup_fast(h2o, keys))
        self.assertTrue(common_nl_lookup_fast(co2, keys))
        self.assertFalse(common_nl_lookup_fast(ch4, keys))

    def test_fast_form_arr_to_str(self):
        arr = Formula(read_formula('C8HF15O2'), 0).array
        self.assertEqual(form_arr_to_str(arr), 'C8HF15O2')

    def test_meta_feature_validation_and_auto_adduct(self):
        with self.assertRaises(ValueError):
            MetaFeature('neg_mz', -10.0, 1)
        with self.assertRaises(ValueError):
            MetaFeature('nan_mz', float('nan'), 1)
        with self.assertRaises(ValueError):
            MetaFeature('auto_double', 300.0, 2, adduct='auto')

        mf_pos = MetaFeature('auto_p', 300.1, 1, adduct='auto')
        self.assertTrue(mf_pos.adduct_auto)
        self.assertEqual(mf_pos.adduct.string, '[M+H]+')

        mf_neg = MetaFeature('auto_n', 300.1, -1, adduct='auto')
        self.assertTrue(mf_neg.adduct_auto)
        self.assertEqual(mf_neg.adduct.string, '[M-H]-')

    def test_featurizer_high_mass_and_empty_spectrum(self):
        # Verify zero 1500 Da ceiling and safe handling of empty MS2
        pf0, pm0, _, mk0, _, nv0 = featurize_peaks_v2(np.array([]), np.array([]), 1850.75)
        self.assertEqual(nv0, 0)
        self.assertEqual(pf0.shape, (64, 40))
        self.assertFalse(mk0.any())

        mzs = np.array([150.05, 500.25, 1620.80, 1750.90], dtype=np.float64)
        ints = np.array([10.0, 80.0, 100.0, 45.0], dtype=np.float64)
        pf, pm, _, mk, _, nv = featurize_peaks_v2(mzs, ints, 1850.75)
        self.assertEqual(nv, 4)
        self.assertAlmostEqual(float(pm[3]), 1750.90, places=2)
        qm = build_query_meta_60(
            mzs, ints, 1850.75,
            np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            1, 1, 1.007825, instrument='orbitrap', collision_energy=35.0,
        )
        self.assertEqual(qm.shape, (QUERY_META_60_DIM,))
        self.assertTrue(np.all(np.isfinite(qm)))
        chem = general_chemistry_summary_numba(mzs, ints, 1850.75, 1, 20.0)
        self.assertEqual(chem.shape, (32,))
        self.assertTrue(np.all(np.isfinite(chem)))

    def test_v2_bundle_checksum_validation(self):
        s0 = b'fake-stage0-onnx-bytes'
        s3 = b'fake-stage3-onnx-bytes'
        lgb = 'tree\nversion=v3\n'
        valid_bundle = {
            'format_version': 4,
            'cand_feat_dim': CAND_FEAT_DIM,
            'pool_feat_dim': POOL_FEAT_DIM,
            'query_meta_dim': QUERY_META_60_DIM,
            'stage0_encoder_onnx': s0,
            'stage0_encoder_sha256': hashlib.sha256(s0).hexdigest(),
            'stage3_reranker_onnx': s3,
            'stage3_reranker_sha256': hashlib.sha256(s3).hexdigest(),
            'stage3_lgb_txt': lgb,
            'stage3_lgb_sha256': hashlib.sha256(lgb.encode('utf-8')).hexdigest(),
            'feat_center': np.zeros(CAND_FEAT_DIM, dtype=np.float32),
            'feat_scale': np.ones(CAND_FEAT_DIM, dtype=np.float32),
            'fusion': {'w_nn': 0.45, 'lgb_scale': 1.2},
            'calibration': {'temp': 1.4, 'b_null': -1.5},
        }
        validate_v2_bundle(valid_bundle)

        tampered = dict(valid_bundle, stage0_encoder_onnx=b'tampered')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            validate_v2_bundle(tampered)

        bad_scale = dict(valid_bundle, feat_scale=np.zeros(CAND_FEAT_DIM, dtype=np.float32))
        with self.assertRaises(ValueError):
            validate_v2_bundle(bad_scale)


if __name__ == '__main__':
    unittest.main()
