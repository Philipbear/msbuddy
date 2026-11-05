"""Regression tests for candidate matching and model feature assembly."""

import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from msbuddy.base import Adduct, CandidateFormula, Formula, MetaFeature, Spectrum
from msbuddy.cand import (_calc_subform_mass, _fill_in_db_existence, assign_subformula,
                          _gen_candidate_formula_from_ms2, _merge_cand_form_list,
                          assign_subformula_cand_form, gen_candidate_formula)
from msbuddy.ml import (_calc_log_p_norm, _calc_log_p_norm_many,
                        _gen_arr_from_buddy_data, gen_ml_feature, gen_ml_feature_single,
                        predict_formula_probability)
from msbuddy.utils import read_formula


class CandidateMatchingTests(unittest.TestCase):
    def test_merge_preserves_order_and_ms1_candidates(self):
        ms1 = [object(), object()]
        ms2 = [object(), object(), object()]
        merged = _merge_cand_form_list(ms1, ms2, ['A', 'B'], ['B', 'C', 'C'])
        self.assertEqual(merged, [*ms1, ms2[1], ms2[2]])

    def test_database_existence_marks_only_matching_candidates(self):
        ms2 = [SimpleNamespace(db_existed=False) for _ in range(3)]
        result = _fill_in_db_existence([object(), object()], ms2, ['A', 'B'], ['B', 'C', 'A'])
        self.assertIs(result, ms2)
        self.assertEqual([candidate.db_existed for candidate in result], [True, False, True])

    def test_repeated_precursor_counts_each_explained_peak_once(self):
        def formula(text, charge, mass):
            return Formula(read_formula(text), charge, mass=mass)

        frag1 = formula('C3H7O3', 1, 50.0)
        loss1 = formula('C3H6O3', 0, 150.0)
        frag2 = formula('C2H5O2', 1, 75.0)
        loss2 = formula('C4H8O4', 0, 125.0)
        spectrum = SimpleNamespace(mz_array=np.array([50.0, 75.0]),
                                   int_array=np.array([0.25, 0.75]),
                                   normalize_intensity=lambda method: None)
        adduct = SimpleNamespace(net_formula=formula('H', 0, 1.007825),
                                 pos_mode=True, m=1, charge=1, loss_formula=None)
        feature = SimpleNamespace(mz=200.0, adduct=adduct, ms2_processed=spectrum)
        lower = np.zeros(12, dtype=np.int16)
        upper = np.full(12, 100, dtype=np.int16)
        with patch('msbuddy.cand._query_frag_nl_pair',
                   side_effect=[([frag1, frag1], [loss1]), ([frag2], [loss2])]):
            candidates, names = _gen_candidate_formula_from_ms2(
                feature, False, 0.01, 0.01, lower, upper, 0, {})
        self.assertEqual(names, ['C6H12O6'])
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].exp_ms2_sum_int, 1.0)

    def test_invalid_candidates_are_skipped_without_discarding_carbon_free_formulas(self):
        def formula(text, charge, mass=None):
            return Formula(read_formula(text), charge, mass=mass)

        empty = CandidateFormula(Formula({}, 0, mass=0),
                                 formula('H', 1), exp_ms2_sum_int=0.0)
        zero_ion = CandidateFormula(formula('H2O', 0),
                                    formula('H', 1, mass=0), exp_ms2_sum_int=0.0)
        nan_mass = CandidateFormula(formula('H2O', 0, mass=float('nan')),
                                    formula('H3O', 1), exp_ms2_sum_int=0.0)
        negative_atoms = CandidateFormula(
            Formula._from_array(np.array([2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int16), 0),
            formula('CH', 1), exp_ms2_sum_int=0.0)
        water = CandidateFormula(formula('H2O', 0), formula('H3O', 1), exp_ms2_sum_int=0.0)
        glucose = CandidateFormula(formula('C6H12O6', 0), formula('C6H13O6', 1), exp_ms2_sum_int=0.0)
        candidates = [empty, water, zero_ion, nan_mass, negative_atoms, glucose]
        feature = MetaFeature(identifier='mixed', mz=100, charge=1)
        lower = np.zeros(12, dtype=np.int16)
        upper = np.full(12, 100, dtype=np.int16)
        with patch('msbuddy.cand._gen_candidate_formula_from_mz', return_value=(candidates, [])), \
             self.assertLogs(level='WARNING'):
            gen_candidate_formula(feature, True, 5, 10, 0, lower, upper, 4, {})
        self.assertEqual(feature.candidate_formula_list, [water, glucose])

        class Model:
            def predict(self, matrix):
                return np.ones(len(matrix))

        config = SimpleNamespace(ppm=True, ms1_tol=5, ms2_tol=10)
        gd = {'model_noms1_noms2': Model(), 'platt_a_3': 1.0, 'platt_b_3': 0.0}
        predict_formula_probability([feature], 0, 1, config, gd)
        self.assertTrue(all(candidate.estimated_prob is not None for candidate in feature.candidate_formula_list))

        no_valid_candidates = MetaFeature(identifier='empty', mz=100, charge=1)
        with patch('msbuddy.cand._gen_candidate_formula_from_mz', return_value=([empty, zero_ion], [])), \
             self.assertLogs(level='WARNING'):
            gen_candidate_formula(no_valid_candidates, True, 5, 10, 0, lower, upper, 4, {})
        predict_formula_probability([no_valid_candidates], 0, 1, config, gd)
        self.assertEqual(no_valid_candidates.candidate_formula_list, [])


class AdductValidationTests(unittest.TestCase):
    def test_formula_dictionary_preserves_internal_mass_and_dbe(self):
        values = [6, 12, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0]
        from_mapping = Formula({'C': 6, 'H': 12, 'O': 6}, 0)
        from_array = Formula._from_array(np.array(values, dtype=np.int16), 0)
        self.assertEqual(from_mapping.dbe, from_array.dbe)
        self.assertEqual(from_mapping.mass, from_array.mass)
        with self.assertRaises(TypeError):
            Formula(values, 0)

    def test_invalid_charge_and_multiplier_follow_documented_fallback(self):
        for positive, expected, examples in (
            (True, '[M+H]+', ('[M+H]0+', '[0M+H]+', '[M+H]oops+', '[M+2+1]+', '[M+H++]+')),
            (False, '[M-H]-', ('[M-H]0-', '[0M-H]-', '[M-H]oops-', '[M-2-1]-', '[M-H--]-')),
        ):
            for text in examples:
                with self.subTest(adduct=text):
                    with self.assertLogs(level='WARNING'):
                        adduct = Adduct(text, positive)
                    charge = 1 if positive else -1
                    self.assertEqual((adduct.string, adduct.m, adduct.charge), (expected, 1, charge))
                    with self.assertRaises(ValueError):
                        Adduct(text, positive, report_invalid=True)

    def test_multi_part_adducts_apply_each_multiplier(self):
        examples = [
            ('[M+2H+2O]+', True, 1, 1, {'H': 2, 'O': 2}, {}),
            ('[M+2H2O+2H]+', True, 1, 1, {'H': 6, 'O': 2}, {}),
            ('[M-2H-1H]-', False, 1, -1, {'H': -3}, {'H': 3}),
            ('[2M+ACN-H2O+2H]+', True, 2, 1,
             {'C': 2, 'H': 3, 'N': 1, 'O': -1}, {'H': 2, 'O': 1}),
            ('[10M+2H]2+', True, 10, 2, {'H': 2}, {}),
        ]
        for text, positive, multiplier, charge, net, loss in examples:
            with self.subTest(adduct=text):
                adduct = Adduct(text, positive, report_invalid=True)
                self.assertEqual((adduct.m, adduct.charge), (multiplier, charge))
                self.assertEqual(dict(zip(Formula.alphabet, adduct.net_formula.array)),
                                 {element: net.get(element, 0) for element in Formula.alphabet})
                self.assertEqual(dict(zip(Formula.alphabet, adduct.loss_formula.array)),
                                 {element: loss.get(element, 0) for element in Formula.alphabet})

    def test_adduct_aliases_are_expanded_before_element_validation(self):
        tfa = Adduct('[M+TFA-H]-', False, report_invalid=True)
        hfa = Adduct('[M+HFA-H]-', False, report_invalid=True)
        self.assertEqual(tfa.net_formula.composition, {'C': 2, 'F': 3, 'O': 2})
        self.assertEqual(hfa.net_formula.composition, {'C': 1, 'H': 1, 'O': 2})

    def test_varied_adduct_terms_match_independent_atom_counts(self):
        rng = random.Random(73)
        terms = [('H2O', {'H': 2, 'O': 1}), ('Na', {'Na': 1}),
                 ('C2H3N', {'C': 2, 'H': 3, 'N': 1}), ('Cl', {'Cl': 1})]
        for _ in range(100):
            multiplier = rng.randint(2, 12)
            charge = rng.choice((-3, -2, -1, 1, 2, 3))
            expected_net = {element: 0 for element in Formula.alphabet}
            expected_loss = expected_net.copy()
            parts = []
            for _ in range(rng.randint(2, 5)):
                formula, atoms = rng.choice(terms)
                coefficient = rng.randint(1, 4)
                sign = rng.choice(('+', '-'))
                parts.append(f'{sign}{coefficient}{formula}')
                for element, count in atoms.items():
                    expected_net[element] += count * coefficient * (1 if sign == '+' else -1)
                    if sign == '-':
                        expected_loss[element] += count * coefficient
            suffix = (str(abs(charge)) if abs(charge) > 1 else '') + ('+' if charge > 0 else '-')
            text = f'[{multiplier}M{"".join(parts)}]{suffix}'
            adduct = Adduct(text, charge > 0, report_invalid=True)
            self.assertEqual((adduct.m, adduct.charge), (multiplier, charge), text)
            self.assertEqual(dict(zip(Formula.alphabet, adduct.net_formula.array)), expected_net, text)
            self.assertEqual(dict(zip(Formula.alphabet, adduct.loss_formula.array)), expected_loss, text)


class FeatureAssemblyTests(unittest.TestCase):
    def setUp(self):
        first = SimpleNamespace(array=np.arange(12, dtype=np.int16), dbe=2.5, mass=123.4)
        second = SimpleNamespace(array=np.arange(12, dtype=np.int16) + 3, dbe=4.0, mass=222.2)
        self.candidates = [SimpleNamespace(formula=first, value=1),
                           SimpleNamespace(formula=second, value=2)]
        self.features = [SimpleNamespace(candidate_formula_list=None),
                         SimpleNamespace(candidate_formula_list=self.candidates[:1]),
                         SimpleNamespace(candidate_formula_list=[]),
                         SimpleNamespace(candidate_formula_list=self.candidates[1:])]

    def test_formula_arrays_preserve_order_dtype_and_empty_shapes(self):
        formulas, dbes, masses = _gen_arr_from_buddy_data(self.features)
        np.testing.assert_array_equal(formulas, [candidate.formula.array for candidate in self.candidates])
        np.testing.assert_array_equal(dbes, [2.5, 4.0])
        np.testing.assert_array_equal(masses, [123.4, 222.2])
        self.assertEqual(formulas.dtype, np.float64)
        empty = _gen_arr_from_buddy_data([])
        self.assertEqual([arr.shape for arr in empty], [(0, 12), (0,), (0,)])

    def test_feature_matrix_preserves_order_and_empty_shape(self):
        with patch('msbuddy.ml.gen_ml_feature_single',
                   side_effect=lambda mf, candidate, *args: np.array([candidate.value, candidate.value + 10])):
            matrix = gen_ml_feature(self.features, True, 5, 10, {})
            one = gen_ml_feature(self.features[1:2], True, 5, 10, {})
            empty = gen_ml_feature([], True, 5, 10, {})
        np.testing.assert_array_equal(matrix, [[1, 11], [2, 12]])
        np.testing.assert_array_equal(one, [[1, 11]])
        self.assertEqual(empty.shape, (1, 0))

    def test_batched_scores_match_scalar_scores_for_float32_and_float64(self):
        for dtype in (np.float32, np.float64):
            values = np.array([-20.0, -3.0, -0.1234567, 0.0, 0.1234567, 3.0, 20.0], dtype=dtype)
            expected = np.array([_calc_log_p_norm(value, 3.3) for value in values])
            np.testing.assert_array_equal(_calc_log_p_norm_many(values, 3.3), expected)

    def test_multicharged_formula_uses_its_already_charge_adjusted_mass(self):
        neutral = Formula(read_formula('C6H12O6'), 0)
        for formula, charge, adduct in (
            ('C6H14O6', 2, '[M+2H]2+'),
            ('C6H10O6', -2, '[M-2H]2-'),
        ):
            with self.subTest(charge=charge):
                charged = Formula(read_formula(formula), charge)
                candidate = CandidateFormula(neutral, charged)
                candidate.formula_feature_array = np.zeros(26)
                feature = MetaFeature('double', charged.mass, charge, adduct=adduct)
                values = gen_ml_feature_single(feature, candidate, True, 5, 10, {})
                self.assertAlmostEqual(values[1], 0.0, places=12)


class SubformulaMassTests(unittest.TestCase):
    def test_empty_msms_has_no_subformula_results(self):
        for peaks in ([], np.array([], dtype=float)):
            self.assertEqual(assign_subformula(peaks, 'C6H12O6', '[M+H]+'), [])

    def test_masses_keep_float32_precision_for_both_polarities(self):
        formulas = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [6, 12, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0],
            [17, 24, 1, 2, 3, 1, 0, 2, 0, 5, 1, 1],
            [48, 99, 0, 0, 0, 0, 0, 5, 0, 12, 1, 2],
        ], dtype=np.int16)
        expected = {
            -1: [0.000548599986359477, 180.06393432617188, 731.8706665039062, 1032.6473388671875],
            1: [-0.000548599986359477, 180.06283569335938, 731.8695678710938, 1032.6463623046875],
            2: [-0.001097199972718954, 180.06228637695312, 731.8690185546875, 1032.645751953125],
        }
        for charge, masses in expected.items():
            np.testing.assert_array_equal(_calc_subform_mass(formulas, charge), np.array(masses, dtype=np.float32))

    def test_public_subformula_assignment_keeps_match_order(self):
        results = assign_subformula([107.05, 149.02, 209.04, 221.04, 230.96],
                                    'C15H16O5', '[M+H]+',
                                    ms2_tol=0.02, ppm=False, dbe_cutoff=-1)
        self.assertEqual([result.idx for result in results], list(range(5)))
        self.assertEqual([[formula.formula for formula in result.subform_list] for result in results], [
            ['C7H7O', 'C3H7O4'],
            ['C12H5', 'C11HO', 'C8H5O3'],
            ['C14H9O2', 'C13H5O3', 'C10H9O5'],
            ['C15H9O2', 'C14H5O3', 'C11H9O5'],
            [],
        ])

    def test_duplicate_peaks_and_both_ion_modes(self):
        for adduct in ('[M+H]+', '[M-H]-'):
            for ppm, tolerance in ((False, 0.02), (True, 20)):
                with self.subTest(adduct=adduct, ppm=ppm):
                    results = assign_subformula([107.05, 107.05, 500.0],
                                                'C15H16O5', adduct,
                                                ms2_tol=tolerance, ppm=ppm, dbe_cutoff=-1)
                    first = [(f.formula, f.mass_error) for f in results[0].subform_list]
                    second = [(f.formula, f.mass_error) for f in results[1].subform_list]
                    self.assertEqual(first, second)
                    self.assertEqual(results[2].subform_list, [])
                    for match in results[0].subform_list:
                        allowed = tolerance * 107.05 / 1e6 if ppm else tolerance
                        self.assertLessEqual(abs(match.mass_error), allowed)

    def test_singly_charged_fragments_with_doubly_charged_precursor(self):
        fragment_mz = 107.04914855957031  # C7H7O+, using the library's float32 masses
        for mz, adduct in ((fragment_mz, '[M+2H]2+'), (107.05024719238281, '[M-2H]2-')):
            results = assign_subformula([mz], 'C15H16O5', adduct,
                                        ms2_tol=1e-5, ppm=False, dbe_cutoff=-1)
            self.assertIn('C7H7O', [formula.formula for formula in results[0].subform_list])

        neutral = Formula(read_formula('C15H16O5'), 0)
        charged = Formula(read_formula('C15H18O5'), 2)
        feature = MetaFeature('double', charged.mass, 2, adduct='[M+2H]2+',
                              ms2=Spectrum([fragment_mz], [100]))
        feature.data_preprocess(False, 0.001, 1e-5, 0.02, 4, 0.01, 6)
        feature.candidate_formula_list = [CandidateFormula(neutral, charged)]
        assign_subformula_cand_form(feature, False, 1e-5)
        explanation = feature.candidate_formula_list[0].ms2_raw_explanation
        self.assertTrue(explanation)
        self.assertEqual(str(explanation.explanation_list[0]), 'C7H7O')


if __name__ == '__main__':
    unittest.main()
