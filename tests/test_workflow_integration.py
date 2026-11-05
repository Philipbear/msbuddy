"""Exercise complete annotation with synthetic data and deterministic models."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from msbuddy import Adduct, Formula, MetaFeature, Msbuddy, MsbuddyConfig, Spectrum
from msbuddy.load import load_mgf, load_usi
from msbuddy.utils import read_formula


class RecordingModel:
    def __init__(self, score):
        self.score = score
        self.shapes = []

    def predict(self, matrix):
        self.shapes.append(matrix.shape)
        return np.full(len(matrix), self.score)


class WorkflowIntegrationTests(unittest.TestCase):
    def test_mgf_usi_and_custom_inputs_cover_all_spectrum_modes(self):
        formula = read_formula('C6H12O6')
        neutral_mass = Formula(formula, 0).mass
        masses = np.array([neutral_mass])
        common = np.empty((0, 12), dtype=np.int16)
        models = [RecordingModel(i) for i in range(4)]
        gd = {
            'basic_db_mass': masses,
            'basic_db_formula': np.array([Formula(formula).array]),
            'basic_db_idx': np.searchsorted(masses, np.arange(15000) / 10),
            'halogen_db_mass': np.array([]),
            'halogen_db_formula': np.empty((0, 12), dtype=np.int16),
            'halogen_db_idx': np.zeros(15000, dtype=int),
            'common_frag_db': common,
            'common_loss_db': common,
            **dict(zip(('model_ms1_ms2', 'model_ms1_noms2',
                        'model_noms1_ms2', 'model_noms1_noms2'), models)),
        }
        for group in range(4):
            gd[f'platt_a_{group}'] = 1.0
            gd[f'platt_b_{group}'] = 0.0

        ion = Adduct('[M+H]+', True)
        mz = neutral_mass + ion.net_formula.mass - 0.00054858
        fragment_mz = Formula(read_formula('C3H7O'), 1).mass
        mgf = f'''\
BEGIN IONS
TITLE=paired
PEPMASS={mz}
CHARGE=1+
MSLEVEL=1
{mz} 100
{mz + 1.003355} 7
END IONS
BEGIN IONS
TITLE=paired
PEPMASS={mz}
CHARGE=1+
MSLEVEL=2
{fragment_mz} 100
END IONS
BEGIN IONS
TITLE=empty
PEPMASS={mz}
CHARGE=1+
END IONS
'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'inputs.mgf'
            path.write_text(mgf, encoding='utf-8')
            mgf_features = load_mgf(path)
        response = SimpleNamespace(text=json.dumps({
            'precursor_mz': mz, 'precursor_charge': 1, 'peaks': [[fragment_mz, 100]]}))
        with patch('msbuddy.load.get', return_value=response):
            usi_features = load_usi('mzspec:example', '[M+H]+')

        ms1 = Spectrum([mz, mz + 1.003355], [100, 7])
        custom = [MetaFeature('ms1_only', mz, 1, ms1=ms1),
                  MetaFeature('precursor_only', mz, 1, ms2=Spectrum([mz], [100])),
                  MetaFeature('zero_intensity', mz, 1, ms2=Spectrum([40], [0])),
                  MetaFeature('nohit', 1.1, 1)]
        for charge, adduct, fragment in ((2, '[M+2H]2+', 'C5H9O4'),
                                         (-2, '[M-2H]2-', 'C5H7O4')):
            double_ion = Adduct(adduct, charge > 0)
            double_mz = (neutral_mass + double_ion.net_formula.mass - charge * 0.00054858) / 2
            high_fragment = Formula(read_formula(fragment), 1 if charge > 0 else -1).mass
            self.assertGreater(high_fragment, double_mz)
            custom.append(MetaFeature(f'double_{charge}', double_mz, charge, adduct=adduct,
                                      ms2=Spectrum([double_mz, high_fragment], [50, 100])))
        engine = object.__new__(Msbuddy)
        engine.config = MsbuddyConfig(ms_instr='orbitrap')
        engine._dependencies = gd
        engine.add_data(mgf_features + usi_features + custom)
        with patch('msbuddy.main.init_ml_models', side_effect=lambda dependencies, **kwargs: dependencies), \
             contextlib.redirect_stdout(io.StringIO()):
            engine.annotate_formula()

        results = {feature.identifier: feature for feature in engine.data}
        expected_groups = {'paired': 0, 'ms1_only': 1, 'mzspec:example': 2,
                           'empty': 3, 'precursor_only': 3, 'zero_intensity': 3}
        for name, group in expected_groups.items():
            feature = results[name]
            self.assertEqual([str(candidate.formula) for candidate in feature.candidate_formula_list], ['C6H12O6'])
            expected_probability = 1 / (1 + np.exp(-group))
            self.assertAlmostEqual(feature.candidate_formula_list[0].estimated_prob, expected_probability)
            self.assertEqual(feature.candidate_formula_list[0].estimated_fdr, 0.0)
        self.assertTrue(results['paired'].candidate_formula_list[0].ms2_raw_explanation)
        self.assertTrue(results['mzspec:example'].candidate_formula_list[0].ms2_raw_explanation)
        for name in ('double_2', 'double_-2'):
            self.assertTrue(results[name].candidate_formula_list[0].ms2_raw_explanation, name)
        self.assertEqual(results['nohit'].candidate_formula_list, [])
        self.assertEqual([model.shapes[0][1] for model in models], [53, 29, 52, 28])

        for bounds, expect_match in (((0, 5), False), ((6, 6), True)):
            with self.subTest(carbon_range=bounds):
                limited = object.__new__(Msbuddy)
                limited.config = MsbuddyConfig(ms_instr='orbitrap', element_ranges={'C': bounds})
                limited._dependencies = gd
                limited.add_data(MetaFeature('limited', mz, 1))
                with patch('msbuddy.main.init_ml_models', side_effect=lambda dependencies, **kwargs: dependencies), \
                     contextlib.redirect_stdout(io.StringIO()):
                    limited.annotate_formula()
                self.assertEqual(bool(limited.data[0].candidate_formula_list), expect_match)


if __name__ == '__main__':
    unittest.main()
