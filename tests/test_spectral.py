"""Compact inference, optional metadata, confidence, and fallback regressions."""
import hashlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

import numpy as np

from msbuddy import Adduct, Formula, MetaFeature, Msbuddy, MsbuddyConfig, Spectrum
from msbuddy._acquisition import collision_energy, instrument_family
from msbuddy._calibration import (CONTEXT_NAMES, CONDITIONAL_CONTEXT_NAMES,
                                  calibrated_probabilities, feature_context)
from msbuddy._spectral import encode, in_scope, rank, _session, _adduct_index, ADDUCTS
from msbuddy.load import _load_versioned_models, load_mgf


class Model:
    def __init__(self, size):
        self.size = size
        self.matrix = None

    def num_feature(self):
        return self.size

    def predict(self, matrix, **kwargs):
        self.matrix = matrix.copy()
        return matrix[:, 94] if self.size == 286 else np.zeros(len(matrix))


def conditional():
    return {'method': 'conditional_null_softmax', 'context_names': list(CONDITIONAL_CONTEXT_NAMES),
            'context_center': [0.] * 11, 'context_scale': [1.] * 11,
            'temperature_coefficients': [np.log(2.)] + [0.] * 11,
            'null_coefficients': [-2.] + [0.] * 11}


def bundle():
    calibration = {'method': 'null_softmax', 'context_names': list(CONTEXT_NAMES),
                   'context_center': [0.] * 9, 'context_scale': [1.] * 9,
                   'coefficients': [-2.] + [0.] * 9, 'inverse_temperature': 1.}
    encoder = b'unit-test-runtime-substitute'
    spectral = {'schema_version': 1, 'encoder_onnx': encoder,
                'encoder_sha256': hashlib.sha256(encoder).hexdigest(),
                'target_center': [0.] * 12, 'target_scale': [1.] * 12,
                'modes': {group: {'model': Model(size),
                                  'blend': {'weight': 1., 'baseline_scale': 1., 'new_scale': 1.},
                                  'calibration': conditional()} for group, size in [(0, 288), (2, 286)]}}
    return {'format_version': 2, 'feature_version': 2, 'version': 'test',
            'models': {name: Model(size) for name, size in [('joint', 95), ('isotope', 71), ('fragment', 94), ('mass', 70)]},
            'calibrations': {name: calibration for name in ['joint', 'isotope', 'fragment', 'mass']},
            'candidate_selector': Model(73), 'candidate_limits': [50, 100, 25, 150],
            'spectral_models': spectral}


def feature(identifier='query', instrument=None, **kwargs):
    f = MetaFeature(identifier, 250., 1, adduct='[M+H]+', instrument=instrument,
                    ms2=Spectrum([40., 80., 120., 249.], [1., .5, .1, .2]), **kwargs)
    f.data_preprocess(True, 5, 10, .02, 4, .01, 6)
    return f


class AcquisitionTests(unittest.TestCase):
    def test_explicit_units_and_ambiguous_energy(self):
        self.assertEqual(collision_energy(30), (None, None))
        self.assertEqual(collision_energy(30, 'eV'), (30., 'ev'))
        self.assertEqual(collision_energy('35 NCE (20 eV)'), (20., 'ev'))
        self.assertEqual(collision_energy('35 nce'), (35., 'nce'))
        for value in ['10-40 eV', '10–40 eV', '10, 20, 40 eV', '6-50 ramp', '-20 eV', '−20 eV', 'unknown']:
            self.assertEqual(collision_energy(value), (None, None))
        self.assertEqual(collision_energy(float('nan'), 'ev'), (None, None))

    def test_instrument_is_explicit_and_unknown_tolerance_does_not_infer_it(self):
        self.assertEqual(instrument_family('Q-Exactive'), 'orbitrap')
        self.assertEqual(instrument_family('timsTOF'), 'timstof')
        self.assertIsNone(instrument_family('FTMS'))
        _, vector = encode(feature())
        self.assertFalse(vector[:3].any())

    def test_mgf_preserves_optional_metadata_and_precursor_type(self):
        text = 'BEGIN IONS\nTITLE=x\nPEPMASS=250\nPRECURSOR_TYPE=[M+Na]+\nINSTRUMENT_TYPE=QTOF\nCOLLISION_ENERGY_EV=20\n40 1\nEND IONS\n'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.mgf'
            path.write_text(text)
            f, = load_mgf(path)
        self.assertEqual(f.adduct.string, '[M+Na]+')
        self.assertEqual((f.instrument, f.collision_energy, f.collision_energy_unit), ('qtof', 20., 'ev'))

    def test_config_update_preserves_metadata_and_explicit_tolerances(self):
        with patch('msbuddy.main.init_db', return_value={}):
            engine = Msbuddy(MsbuddyConfig(ms_instr='qtof', use_spectral=False))
            engine.update_config(ms1_tol=3)
        self.assertEqual(engine.config.ms_instr, 'qtof')
        self.assertEqual(engine.config.ms1_tol, 3)
        self.assertFalse(engine.config.use_spectral)
        self.assertEqual((MsbuddyConfig(ms_instr='timstof').ms1_tol, MsbuddyConfig(ms_instr='timstof').ms2_tol), (10, 20))


class SpectralRuntimeTests(unittest.TestCase):
    def test_all_canonical_adduct_codes_match_the_frozen_schema(self):
        for index, text in enumerate(ADDUCTS):
            self.assertEqual(_adduct_index(Adduct(text, text.endswith('+'), report_invalid=True)), index)

    def test_scope_retains_established_models_for_unsupported_settings(self):
        f = feature()
        self.assertTrue(in_scope(f, MsbuddyConfig()))
        for config in [MsbuddyConfig(use_spectral=False), MsbuddyConfig(ms_instr='fticr'),
                       MsbuddyConfig(max_candidates=10), MsbuddyConfig(ms1_tol=3),
                       MsbuddyConfig(max_isotope_cnt=6), MsbuddyConfig(ppm=False)]:
            self.assertFalse(in_scope(f, config))
        self.assertFalse(in_scope(feature(instrument='qtof'), MsbuddyConfig()))
        self.assertTrue(in_scope(feature(instrument='qtof'), MsbuddyConfig(ms_instr='qtof')))

    def test_equivalent_adduct_spelling_encodes_identically(self):
        a, b = feature(), feature()
        a.adduct = Adduct('[M+HCOO]-', False, report_invalid=True)
        b.adduct = Adduct('[M+CH2O2-H]-', False, report_invalid=True)
        xa, ma = encode(a)
        xb, mb = encode(b)
        np.testing.assert_array_equal(xa, xb)
        np.testing.assert_array_equal(ma, mb)

    def test_runtime_is_lazy_cpu_only_and_isolated_per_engine(self):
        factory = Mock(side_effect=lambda *args, **kwargs: object())
        runtime = SimpleNamespace(SessionOptions=lambda: SimpleNamespace(), InferenceSession=factory)
        a = {'model_spectral_cache': {}, 'model_spectral': {'encoder_onnx': b'one'}}
        b = {'model_spectral_cache': {}, 'model_spectral': {'encoder_onnx': b'one'}}
        with patch.dict(sys.modules, {'onnxruntime': runtime}):
            first = _session(a)
            self.assertIs(first, _session(dict(a)))
            self.assertIsNot(first, _session(b))
        self.assertEqual(factory.call_count, 2)
        self.assertEqual(factory.call_args.kwargs['providers'], ['CPUExecutionProvider'])

    def test_candidate_offsets_batching_and_encoder_mass_mask(self):
        class Runtime:
            def __init__(self): self.inputs = []
            def run(self, _, inputs):
                self.inputs.append(inputs)
                n = len(inputs['metadata'])
                return np.zeros((n, 128), dtype=np.float32), np.zeros((n, 15), dtype=np.float32)
        features = []
        for i in range(33):
            f = feature(str(i))
            f.candidate_formula_list = [SimpleNamespace(formula=Formula.from_string(name)) for name in ['C6H12O6', 'C5H8O7']]
            features.append(f)
        model = Model(286)
        runtime = Runtime()
        specification = bundle()['spectral_models']
        specification['modes'][2]['model'] = model
        deps = {'model_spectral': specification, 'model_spectral_cache': {'session': runtime}}
        baseline = np.arange(66, dtype=float)
        scores = rank(features, np.zeros((66, 94)), baseline, 2, deps)
        np.testing.assert_array_equal(scores, baseline)
        self.assertEqual([len(x['metadata']) for x in runtime.inputs], [32, 1])
        self.assertTrue(all(not x['metadata'][:, 17:19].any() for x in runtime.inputs))
        self.assertEqual(model.matrix.shape, (66, 286))
        self.assertEqual(model.matrix.dtype, np.float32)
        self.assertTrue(np.all(model.matrix[:, 95 + 17] > 0))

    def test_invalid_compact_bundle_does_not_partially_initialize_engine(self):
        data = bundle()
        data['spectral_models']['encoder_sha256'] = 'invalid'
        dependencies = {'sentinel': 1}
        with self.assertRaises(ValueError):
            _load_versioned_models(data, dependencies)
        self.assertEqual(dependencies, {'sentinel': 1})

    def test_valid_bundle_has_independent_empty_runtime_cache(self):
        a, b = {}, {}
        _load_versioned_models(bundle(), a)
        _load_versioned_models(bundle(), b)
        self.assertEqual(a['model_spectral_cache'], {})
        self.assertIsNot(a['model_spectral_cache'], b['model_spectral_cache'])


class ConditionalConfidenceTests(unittest.TestCase):
    def test_context_clipping_bounds_extrapolation_without_changing_old_specs(self):
        spec = conditional();spec['context_clip'] = 6.
        spec['temperature_coefficients'][1] = 1.
        spec['null_coefficients'][2] = 1.
        scores = np.array([1., 0.])
        a = calibrated_probabilities(scores, np.full(11, 100.), spec)
        b = calibrated_probabilities(scores, np.full(11, 6.), spec)
        np.testing.assert_array_equal(a[0], b[0]);self.assertEqual(a[1], b[1])
        self.assertAlmostEqual(float(a[0].sum()) + a[1], 1.)
        for value in [0., -1., np.nan, True, 'six']:
            spec['context_clip'] = value
            with self.assertRaises(ValueError):calibrated_probabilities(scores, np.zeros(11), spec)

    def test_positive_temperature_preserves_ranking_and_absent_mass(self):
        scores = np.array([-2., 3., 0.])
        spec = conditional()
        spec['temperature_coefficients'][-2] = .1
        context = np.zeros(11)
        p, absent = calibrated_probabilities(scores, context, spec)
        self.assertAlmostEqual(float(p.sum()) + absent, 1.)
        self.assertGreater(absent, 0)
        np.testing.assert_array_equal(np.argsort(-scores), np.argsort(-p))
        spec['temperature_coefficients'][-1] = float('nan')
        with self.assertRaises(ValueError):
            calibrated_probabilities(scores, context, spec)

    def test_conditional_context_uses_observed_winner_mass_fit(self):
        f = feature()
        f.candidate_formula_list = [SimpleNamespace(formula=Formula.from_string('C6H12O6'),
             charged_formula=Formula.from_string('C6H13O6', charge=1), ms1_isotope_similarity=None,
             ms2_raw_explanation=None)]
        context = feature_context(f, np.array([1.]), 5, CONDITIONAL_CONTEXT_NAMES)
        self.assertEqual(context.shape, (11,))
        self.assertEqual(context[-2], -2.)
        self.assertEqual(context[-1], 0)

    def test_importing_encoder_does_not_import_inference_or_training_runtime(self):
        subprocess.run([sys.executable, '-c', "import msbuddy._spectral, sys; assert 'torch' not in sys.modules; assert 'onnxruntime' not in sys.modules"], check=True)


if __name__ == '__main__':
    unittest.main()
