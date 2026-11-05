"""Regression tests for formula database boundaries and adduct tolerances."""

import unittest

import numpy as np

from msbuddy.base import Adduct, Formula
from msbuddy.main import Msbuddy
from msbuddy.query import query_neutral_mass, query_precursor_mass
from msbuddy.utils import read_formula


def database(masses):
    masses = np.array(sorted(masses), dtype=np.float64)
    forms = np.array([Formula(read_formula('C6H12O6')).array] * len(masses), dtype=np.int16).reshape(-1, 12)
    return {
        'basic_db_mass': masses,
        'basic_db_formula': forms,
        'basic_db_idx': np.searchsorted(masses, np.arange(15000) / 10),
        'halogen_db_mass': np.empty(0, dtype=np.float64),
        'halogen_db_formula': np.empty((0, 12), dtype=np.int16),
        'halogen_db_idx': np.zeros(15000, dtype=int),
    }


class MassQueryTests(unittest.TestCase):
    def test_exact_and_inclusive_bin_boundaries(self):
        gd = database([99.9, 100.0, 100.1])
        self.assertEqual([f.mass for f in query_neutral_mass(100, 0, False, False, gd)], [100.0])
        self.assertEqual([f.mass for f in query_neutral_mass(100, 0.1, False, False, gd)],
                         [99.9, 100.0, 100.1])

    def test_upper_and_lower_index_boundaries(self):
        gd = database([*np.linspace(1, 1498, 20000), 1499.92, 1499.99, 1500.02, 1500.2])
        self.assertEqual([f.mass for f in query_neutral_mass(1499.99, 0.05, False, False, gd)],
                         [1499.99, 1500.02])
        small_gd = database([0.05, 1.0])
        self.assertEqual([f.mass for f in query_neutral_mass(0.05, 0.1, False, False, small_gd)], [0.05])
        self.assertEqual(query_neutral_mass(-1.0, 0.1, False, False, small_gd), [])

    def test_adduct_charge_and_multimer_scale_tolerance_in_mz(self):
        neutral_mass = 180.0633881
        gd = database([neutral_mass])
        cases = [('[M+2H]2+', 0.000075, True), ('[M-2H]2-', 0.000075, True),
                 ('[2M+H]+', 0.000075, True), ('[2M+H]+', 0.00015, False)]
        for text, offset, should_match in cases:
            with self.subTest(adduct=text):
                ion = Adduct(text, text.endswith('+'))
                exact_mz = (neutral_mass * ion.m + ion.net_formula.mass -
                            ion.charge * 0.00054858) / abs(ion.charge)
                found, _ = query_precursor_mass(exact_mz + offset, ion, 0.0001, False, 0, gd)
                self.assertEqual(bool(found), should_match)

    def test_ppm_tolerance_scales_for_multiply_charged_ions(self):
        neutral_mass = 180.0633881
        gd = database([neutral_mass])
        ion = Adduct('[M+2H]2+', True)
        exact_mz = (neutral_mass + ion.net_formula.mass - ion.charge * 0.00054858) / 2
        tolerance = exact_mz * 2 / 1e6
        for factor, expected in ((0.75, True), (1.25, False)):
            found, _ = query_precursor_mass(exact_mz + tolerance * factor, ion, 2, True, 0, gd)
            self.assertEqual(bool(found), expected)

    def test_halogen_database_is_only_searched_when_requested(self):
        gd = database([100.0])
        gd['halogen_db_mass'] = np.array([100.0])
        gd['halogen_db_formula'] = np.array([Formula(read_formula('C5H5Cl')).array], dtype=np.int16)
        gd['halogen_db_idx'] = np.searchsorted(gd['halogen_db_mass'], np.arange(15000) / 10)
        self.assertEqual(len(query_neutral_mass(100, 0.01, False, False, gd)), 1)
        self.assertEqual(len(query_neutral_mass(100, 0.01, False, True, gd)), 2)

    def test_mz_to_formula_reports_the_multicharge_mass_error(self):
        neutral_mass = 180.0633881
        gd = database([neutral_mass])
        ion = Adduct('[M+2H]2+', True)
        mz = (neutral_mass + ion.net_formula.mass - ion.charge * 0.00054858) / abs(ion.charge)
        engine = object.__new__(Msbuddy)
        engine._dependencies = gd
        matches = engine.mz_to_formula(mz, '[M+2H]2+', mz_tol=0.0001, ppm=False)
        self.assertEqual([match.formula for match in matches], ['C6H12O6'])
        self.assertAlmostEqual(matches[0].mass_error, 0.0, places=10)


if __name__ == '__main__':
    unittest.main()
