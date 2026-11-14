"""Protect small default loading, lazy compatibility and shared scoring routes."""
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from msbuddy import Formula
from msbuddy.load import init_ml_models, init_compatibility_models, _load_versioned_models
from msbuddy.ml import _predict_ml
from tests.test_spectral import Model, bundle, conditional, feature, ADDUCTS


def core_bundle():
    old = bundle()
    spectral = copy.deepcopy(old['spectral_models'])
    spectral.update(schema_version=2, ranker=Model(435),
                    modes={group: {'calibration': conditional()} for group in [0, 2]})
    return {'format_version': 3, 'feature_version': 2, 'version': 'test-compact',
            'candidate_selector': Model(73), 'candidate_limits': [50, 50, 25, 75],
            'validated_adducts': list(ADDUCTS), 'spectral_models': spectral,
            'compatibility': {'filename': 'optional.joblib', 'url': 'https://example.org/optional.joblib',
                              'sha256': '1' * 64}}


def compatibility_bundle():
    old = bundle()
    old.pop('spectral_models')
    old['legacy_bundle'] = [Model(n) for n in [53, 52, 29, 28]] + [-1., 0.] * 4
    return old


class CompactBundleTests(unittest.TestCase):
    def test_default_core_does_not_load_optional_estimators(self):
        with patch('msbuddy.load.check_download_joblibload', return_value=core_bundle()) as loader:
            dependencies = init_ml_models({}, '/tmp/example', {'ml_model': '/tmp/custom/core.joblib'})
            self.assertIs(init_ml_models(dependencies), dependencies)
        self.assertEqual(loader.call_count, 1)
        self.assertTrue(dependencies['model_core_loaded'])
        self.assertNotIn('model_ms1_ms2', dependencies)
        self.assertNotIn('legacy_dependencies', dependencies)
        self.assertEqual(dependencies['model_compatibility_files']['path'], Path('/tmp/custom/optional.joblib'))
        self.assertEqual(dependencies['model_compatibility_files']['expected_sha256'], '1' * 64)

    def test_optional_load_is_once_and_preserves_shared_state(self):
        dependencies = {}
        _load_versioned_models(core_bundle(), dependencies)
        original = dependencies['model_spectral']
        dependencies['model_compatibility_files'] = dict(path=Path('/tmp/optional.joblib'), url='https://example.org/optional.joblib',
                                                       download_missing=False, expected_sha256='1' * 64)
        with patch('msbuddy.load.check_download_joblibload', return_value=compatibility_bundle()) as loader:
            init_compatibility_models(dependencies)
            init_compatibility_models(dependencies)
        self.assertEqual(loader.call_count, 1)
        self.assertIs(dependencies['model_spectral'], original)
        self.assertEqual(dependencies['model_bundle_version'], 'test-compact')
        self.assertEqual(dependencies['candidate_limits'], (50, 50, 25, 75))
        self.assertIn('legacy_dependencies', dependencies)
        self.assertTrue(dependencies['model_compatibility_loaded'])

    def test_invalid_optional_file_does_not_partially_initialize(self):
        dependencies = {'model_compatibility_files': {}}
        with patch('msbuddy.load.check_download_joblibload', return_value=core_bundle()):
            with self.assertRaises(ValueError):init_compatibility_models(dependencies)
        self.assertEqual(set(dependencies), {'model_compatibility_files'})

    def test_conditional_confidence_is_supported_in_numerical_modes(self):
        data = compatibility_bundle()
        for mode in ['isotope', 'mass']:
            data['calibrations'][mode] = dict(conditional(), context_clip=6.)
        dependencies = {}
        _load_versioned_models(data, dependencies)
        self.assertEqual(dependencies['calibration_1']['method'], 'conditional_null_softmax')
        self.assertEqual(dependencies['calibration_3']['context_clip'], 6.)

    def test_shared_ranking_skips_legacy_scores_and_uses_three_encoder_inputs(self):
        f = feature()
        f.candidate_formula_list = [SimpleNamespace(formula=Formula.from_string(name)) for name in ['C6H12O6', 'C5H8O7']]
        class Runtime:
            def run(self, _, inputs):
                self.inputs = inputs
                return np.zeros((1, 128), dtype=np.float32), np.zeros((1, 72), dtype=np.float32)
        runtime = Runtime()
        dependencies = {}
        _load_versioned_models(core_bundle(), dependencies)
        dependencies.update(model_spectral_active=True, model_spectral_tolerance=10,
                            model_spectral_cache={'session': runtime})
        with patch('msbuddy.ml.gen_ml_feature', return_value=np.zeros((2, 95))):
            scores = _predict_ml([f], 2, True, 5, 10, dependencies)
        self.assertEqual(len(scores), 2)
        self.assertNotIn('model_noms1_ms2', dependencies)
        self.assertEqual(set(runtime.inputs), {'spectra', 'metadata', 'chemistry'})
        self.assertEqual(runtime.inputs['chemistry'].shape, (1, 64))
        self.assertFalse(runtime.inputs['metadata'][:, 17:19].any())
        self.assertEqual(dependencies['model_spectral']['ranker'].matrix.shape, (2, 435))
        self.assertTrue(np.all(dependencies['model_spectral']['ranker'].matrix[:, 95 + 17] > 0))

    def test_corrupt_core_does_not_partially_initialize(self):
        for edit in [lambda x: x['spectral_models'].update(ranker=Model(434)),
                     lambda x: x['compatibility'].update(filename='../outside'),
                     lambda x: x['compatibility'].update(sha256='bad')]:
            data = core_bundle();edit(data);dependencies = {'sentinel': 1}
            with self.assertRaises(ValueError):_load_versioned_models(data, dependencies)
            self.assertEqual(dependencies, {'sentinel': 1})
