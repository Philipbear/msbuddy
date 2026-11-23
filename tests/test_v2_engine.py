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

    def test_materialize_int16_streams_database_without_cache_bloat(self):
        import tempfile
        from pathlib import Path
        from msbuddy._database import Database, write_database
        from msbuddy._fast_db import _materialize_int16

        arr = np.arange(12000 * 12, dtype=np.int64).reshape(12000, 12) % 97
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / 'mini.msbdb'
            write_database(db_path, {'forms': arr}, role='test', block_rows=2048)
            db = Database(db_path)
            try:
                out = _materialize_int16(db.array('forms'))
                self.assertEqual(out.dtype, np.int16)
                self.assertTrue(out.flags['C_CONTIGUOUS'])
                np.testing.assert_array_equal(out, arr.astype(np.int16))
                self.assertEqual(len(db._cache), 0)
                self.assertEqual(db._cached_bytes, 0)
            finally:
                db.close()

    def test_v2_peak_fourier_stacked_channel_order_and_auto_adduct_priors(self):
        import math
        from msbuddy._v2_engine import (
            AUTO_ADDUCT_LOG_PRIORS,
            AUTO_ADDUCTS_NEG,
            AUTO_ADDUCTS_POS,
            PEAK_WAVELENGTHS,
        )

        mzs = np.array([123.4567, 250.1234], dtype=np.float64)
        ints = np.array([50.0, 100.0], dtype=np.float64)
        pmz = 350.25
        pf, _, _, _, _, nv = featurize_peaks_v2(mzs, ints, pmz)
        self.assertEqual(nv, 2)
        two_pi = 2.0 * math.pi
        losses = pmz - mzs
        # Channels 5..11: sin(mz), 12..18: cos(mz), 19..25: sin(nl), 26..32: cos(nl)
        for j, wl in enumerate(PEAK_WAVELENGTHS):
            self.assertAlmostEqual(float(pf[0, 5 + j]), math.sin((two_pi / wl) * mzs[0]), places=5)
            self.assertAlmostEqual(float(pf[0, 12 + j]), math.cos((two_pi / wl) * mzs[0]), places=5)
            self.assertAlmostEqual(float(pf[0, 19 + j]), math.sin((two_pi / wl) * losses[0]), places=5)
            self.assertAlmostEqual(float(pf[0, 26 + j]), math.cos((two_pi / wl) * losses[0]), places=5)

        for a in AUTO_ADDUCTS_POS + AUTO_ADDUCTS_NEG:
            self.assertIn(a, AUTO_ADDUCT_LOG_PRIORS)
        self.assertEqual(AUTO_ADDUCT_LOG_PRIORS['[M+H]+'], 0.0)
        self.assertEqual(AUTO_ADDUCT_LOG_PRIORS['[M-H]-'], 0.0)

    def test_v5_ms2_evidence_parity_and_bundle_validation(self):
        from msbuddy._fast_db import compute_ms2_evidence_fast, compute_ms2_evidence_fast_v3
        from msbuddy._v2_engine import (
            CAND_V5_DIM,
            CL_DIM,
            POOL_V5_DIM,
            SPEC_CTX_DIM,
            _sha256_np_dict,
            build_formula_features_v3,
            extract_adduct_meta5,
            formula_enc_forward_np,
        )

        # Verify compute_ms2_evidence_fast_v3 cols 0..23 match compute_ms2_evidence_fast
        f1 = Formula(read_formula('C15H12O4'), 0).array.astype(np.int16)
        f2 = Formula(read_formula('C13H12N4O2'), 0).array.astype(np.int16)
        c_neut = np.stack([f1, f2], axis=0)
        c_ch = c_neut.copy()
        c_ch[:, 1] += 1
        pmzs = np.array([105.0335, 137.0233, 181.0495, 225.0546], dtype=np.float32)
        pints = np.array([0.4, 0.8, 1.0, 0.6], dtype=np.float32)
        masses = np.array([18.010565, 27.994915, 43.98983, 104.026215, 136.016044, 180.042259], dtype=np.float64)
        idx = np.searchsorted(masses, np.arange(3000) / 10.0).astype(np.int64)
        cl_keys = np.sort(np.array([pack_common_key(Formula(read_formula('H2O'), 0).array.astype(np.int16))], dtype=np.uint64))
        cf_db = np.zeros((1, 12), dtype=np.int16)

        f24, bf24, bfd24 = compute_ms2_evidence_fast(c_neut, c_ch, pmzs, pints, 4, 1, True, 20.0, masses, idx, masses, idx, cl_keys, cf_db)
        f48, bf48, bfd48 = compute_ms2_evidence_fast_v3(c_neut, c_ch, pmzs, pints, 4, 1, True, 20.0, masses, idx, masses, idx, cl_keys, cf_db)
        self.assertEqual(f48.shape, (2, 48))
        self.assertTrue(np.all(np.isfinite(f48)))
        np.testing.assert_allclose(f48[:, :24], f24, atol=1e-5)
        np.testing.assert_array_equal(bfd48, bfd24)
        np.testing.assert_array_equal(bf48, bf24)

        # Verify pure-NumPy FormulaEncoderV3 forward pass & v5 bundle validation
        rng = np.random.default_rng(7)
        fenc_np = {
            'in_proj.0.weight': rng.normal(scale=0.1, size=(16, 32)).astype(np.float32),
            'in_proj.0.bias': np.zeros(16, dtype=np.float32),
            'in_proj.1.weight': np.ones(16, dtype=np.float32),
            'in_proj.1.bias': np.zeros(16, dtype=np.float32),
            'res1.0.weight': rng.normal(scale=0.1, size=(16, 16)).astype(np.float32),
            'res1.0.bias': np.zeros(16, dtype=np.float32),
            'res1.1.weight': np.ones(16, dtype=np.float32),
            'res1.1.bias': np.zeros(16, dtype=np.float32),
            'res1.4.weight': rng.normal(scale=0.1, size=(16, 16)).astype(np.float32),
            'res1.4.bias': np.zeros(16, dtype=np.float32),
            'res2.0.weight': rng.normal(scale=0.1, size=(16, 16)).astype(np.float32),
            'res2.0.bias': np.zeros(16, dtype=np.float32),
            'res2.1.weight': np.ones(16, dtype=np.float32),
            'res2.1.bias': np.zeros(16, dtype=np.float32),
            'res2.4.weight': rng.normal(scale=0.1, size=(16, 16)).astype(np.float32),
            'res2.4.bias': np.zeros(16, dtype=np.float32),
            'out_norm.weight': np.ones(16, dtype=np.float32),
            'out_norm.bias': np.zeros(16, dtype=np.float32),
            'out_proj.weight': rng.normal(scale=0.1, size=(CL_DIM, 16)).astype(np.float32),
            'out_proj.bias': np.zeros(CL_DIM, dtype=np.float32),
        }
        qm = np.zeros(QUERY_META_60_DIM, dtype=np.float32)
        qm[15] = 1.0; qm[16] = 0.5
        am5 = np.broadcast_to(extract_adduct_meta5(qm)[None, :], (2, 5))
        zf = formula_enc_forward_np(build_formula_features_v3(c_neut, am5), fenc_np)
        self.assertEqual(zf.shape, (2, CL_DIM))
        np.testing.assert_allclose(np.linalg.norm(zf, axis=1), np.ones(2), atol=1e-5)

        s0 = b'fake-s0-v5'; s3 = b'fake-s3-v5'; lgb = 'tree\nversion=v3\n'
        v5_bundle = {
            'format_version': 5,
            'cand_feat_dim': CAND_V5_DIM,
            'pool_feat_dim': POOL_V5_DIM,
            'query_meta_dim': QUERY_META_60_DIM,
            'cl_dim': CL_DIM,
            'spec_ctx_dim': SPEC_CTX_DIM,
            'logit_scale': 14.0,
            'stage0_encoder_onnx': s0,
            'stage0_encoder_sha256': hashlib.sha256(s0).hexdigest(),
            'formula_enc_np': fenc_np,
            'formula_enc_sha256': _sha256_np_dict(fenc_np),
            'stage3_reranker_onnx': s3,
            'stage3_reranker_sha256': hashlib.sha256(s3).hexdigest(),
            'stage3_lgb_txt': lgb,
            'stage3_lgb_sha256': hashlib.sha256(lgb.encode('utf-8')).hexdigest(),
            'feat_center': np.zeros(CAND_V5_DIM, dtype=np.float32),
            'feat_scale': np.ones(CAND_V5_DIM, dtype=np.float32),
            'fusion': {'w_nn': 0.20, 'lgb_scale': 1.2},
            'calibration': {'temp': 0.73, 'b_null': -4.0},
        }
        validate_v2_bundle(v5_bundle)
        with self.assertRaisesRegex(ValueError, 'checksum'):
            validate_v2_bundle(dict(v5_bundle, formula_enc_sha256='0' * 64))

        # Regression test: MS1 signed ppm in _v2_engine (obs - theo) must match MS2 signed ppm sign
        # so MS1-MS2 self-recalibration |sub_err_ppm - ms2_wmean_ppm| cancels systematic calibration drift.
        drift_ppm = +5.0
        f_ch = c_ch[0:1]
        from msbuddy._fast_db import ELEM_MASSES_F64, MASS_E
        frag_theo = float(Formula(read_formula('C7H5O'), 1).mass)
        shifted_pmzs = np.array([frag_theo * (1.0 + drift_ppm * 1e-6)], dtype=np.float32)
        shifted_ints = np.array([1.0], dtype=np.float32)
        f48_shift, _, bfd_shift = compute_ms2_evidence_fast_v3(
            c_neut[0:1], f_ch, shifted_pmzs, shifted_ints, 1, 1, True, 20.0,
            masses, idx, masses, idx, cl_keys, cf_db,
        )
        self.assertTrue(bool(bfd_shift[0, 0]))
        self.assertAlmostEqual(float(f48_shift[0, 24]), drift_ppm, delta=0.5)
        # Simulated generate_candidates_fast_numba out_err = (theo - obs)/obs * 1e6 ~ -drift_ppm;
        # negating gives +drift_ppm so |(-out_err) - f48_shift[0,24]| ~ 0!
        self.assertLess(abs(drift_ppm - float(f48_shift[0, 24])), 0.5)


if __name__ == '__main__':
    unittest.main()
