"""Independent exhaustive-reference checks for the bounded matcher."""
import unittest
from types import SimpleNamespace

import numpy as np

from msbuddy._subformula import closest_subformulas
from msbuddy.cand import (_calc_subform_mass, _dbe_subform_filter,
                          _senior_subform_filter, _valid_subform_check)
from msbuddy import Formula
from msbuddy.base import Spectrum
from msbuddy.cand import CandidateSpace, FragExplanation
from msbuddy.utils import enumerate_subformula, read_formula


class TargetedSubformulaTests(unittest.TestCase):
    def test_matches_exhaustive_search_including_halogen_and_adduct_atoms(self):
        rng = np.random.default_rng(924)
        for name in ['C8H16O5', 'C12H20Cl2N3O4', 'C10H18BrNO3',
                     'C14H22F3N4O5', 'C10H22NaO6', 'C10H22KO6',
                     'C12H24NO6PS', 'H4O2', 'C6O6']:
            precursor = Formula(read_formula(name)).array
            forms = enumerate_subformula(precursor)
            for charge in [-1, 1]:
                masses = _calc_subform_mass(forms, charge)
                selected = rng.integers(0, len(forms), size=25)
                targets = np.maximum(1, masses[selected] * (1 + rng.normal(0, 4e-6, len(selected)))).astype(np.float32)
                tolerances = targets.astype(float) * 10e-6
                found, matched, calculated = closest_subformulas(precursor, targets, tolerances, targets, charge)
                for i, target in enumerate(targets):
                    indices = np.flatnonzero(np.abs(target - masses) <= tolerances[i])
                    subset = forms[indices]
                    valid = (_dbe_subform_filter(subset, 0) & _senior_subform_filter(subset)
                             & _valid_subform_check(subset, precursor))
                    indices = indices[valid]
                    self.assertEqual(bool(found[i]), bool(len(indices)), (name, charge, i))
                    if len(indices):
                        closest = indices[np.argmin(np.abs(target - masses[indices]))]
                        np.testing.assert_array_equal(matched[i], forms[closest])
                        self.assertEqual(calculated[i], masses[closest])

    def test_unreachable_targets_are_not_reported_as_matches(self):
        precursor = Formula(read_formula('C2H6O')).array
        found, _, _ = closest_subformulas(precursor, np.array([300., 400.], dtype=np.float32),
                                        np.array([.001, .001]), np.array([300., 400.], dtype=np.float32), 1)
        self.assertFalse(found.any())

    def test_nearby_noise_peak_is_not_mistaken_for_an_isotope(self):
        neutral = Formula(read_formula('C6H12O6'))
        charged = Formula(read_formula('C6H13O6'), 1)
        fragment = Formula(read_formula('C3H7O3'), 1)
        for difference, expected in [(.2, 1), (1.003355, 1), (1.1, 1)]:
            feature = SimpleNamespace(ms2_raw=Spectrum([fragment.mass, fragment.mass + difference], [100, 20]),
                                      ms2_processed=SimpleNamespace(idx_array=np.array([0])),
                                      adduct=SimpleNamespace(charge=1))
            explanation = FragExplanation(0, fragment,
                                          Formula._from_array(charged.array - fragment.array, 0))
            space = CandidateSpace(neutral.array, charged.array, frag_exp_ls=[explanation])
            result = space.refine_explanation(feature, .02)
            self.assertEqual(len(result.ms2_raw_explanation), expected, difference)

    def test_raw_peak_indices_above_int16_range_are_preserved(self):
        neutral = Formula(read_formula('C6H12O6'))
        charged = Formula(read_formula('C6H13O6'), 1)
        fragment = Formula(read_formula('C3H7O3'), 1)
        index = 40000
        feature = SimpleNamespace(ms2_raw=SimpleNamespace(mz_array=np.full(index + 1, fragment.mass)),
                                  ms2_processed=SimpleNamespace(idx_array=np.array([index])),
                                  adduct=SimpleNamespace(charge=1))
        explanation = FragExplanation(index, fragment, Formula._from_array(charged.array - fragment.array, 0))
        result = CandidateSpace(neutral.array, charged.array, frag_exp_ls=[explanation]).refine_explanation(feature, .02)
        self.assertEqual(result.ms2_raw_explanation.idx_array.tolist(), [index])


if __name__ == '__main__':
    unittest.main()
