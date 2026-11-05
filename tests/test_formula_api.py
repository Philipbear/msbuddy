"""Public formula and element-range APIs use validated element mappings."""

import random
import unittest

import numpy as np

from msbuddy import (Adduct, Formula, MetaFeature, Msbuddy, MsbuddyConfig,
                     SUPPORTED_ELEMENTS, UnsupportedElementError, assign_subformula,
                     enumerate_subformulas, formula_to_string, read_formula)
from msbuddy.utils import add_formula_str, read_formula_str


class FormulaApiTests(unittest.TestCase):
    def test_formula_strings_and_mappings_round_trip(self):
        examples = {
            'C6H12O6': {'C': 6, 'H': 12, 'O': 6},
            '(CH3)2CHOH': {'C': 3, 'H': 8, 'O': 1},
            'C5H7NO2.HCl': {'C': 5, 'H': 8, 'Cl': 1, 'N': 1, 'O': 2},
            '2H2O·3NaCl': {'H': 4, 'Cl': 3, 'Na': 3, 'O': 2},
            'CHBrClFIKNNaOPS': {element: 1 for element in SUPPORTED_ELEMENTS},
        }
        for text, expected in examples.items():
            with self.subTest(formula=text):
                composition = read_formula(text)
                self.assertEqual(composition, expected)
                self.assertEqual(read_formula(formula_to_string(composition)), expected)
                self.assertEqual(read_formula_str(text), expected)
                self.assertEqual(Formula.from_string(text).composition, expected)
        self.assertEqual(add_formula_str('C6H12O5', 'H2O'), {'C': 6, 'H': 14, 'O': 6})

    def test_composition_is_a_copy_and_zero_counts_are_omitted(self):
        formula = Formula({'O': np.int64(6), 'C': 6, 'H': 12, 'S': 0})
        composition = formula.composition
        self.assertEqual(str(formula), 'C6H12O6')
        self.assertEqual(composition, {'C': 6, 'H': 12, 'O': 6})
        composition['C'] = 7
        self.assertEqual(formula.composition['C'], 6)
        self.assertEqual(formula.charge, 0)

    def test_varied_compositions_round_trip_without_element_order_assumptions(self):
        rng = random.Random(2026)
        for _ in range(100):
            counts = {element: rng.randint(0, 5) for element in SUPPORTED_ELEMENTS}
            counts = {element: count for element, count in counts.items() if count}
            if not counts:
                continue
            self.assertEqual(read_formula(formula_to_string(counts)), counts)
            self.assertEqual(Formula(counts).composition, counts)

    def test_subformula_enumeration_is_lazy_and_returns_mappings(self):
        result = enumerate_subformulas({'C': 1, 'O': 1})
        self.assertIs(iter(result), result)
        self.assertEqual(list(result), [{}, {'C': 1}, {'O': 1}, {'C': 1, 'O': 1}])
        self.assertEqual(list(enumerate_subformulas({})), [{}])

    def test_invalid_syntax_and_atom_counts_are_rejected(self):
        for text in ('', 'H2O!', 'C0H2', 'C-1H', 'C(OH', 'C()', 'C..H', '2', 'C 2'):
            with self.subTest(formula=text), self.assertRaises(ValueError):
                read_formula(text)
        for count in (-1, 1.5, True, np.bool_(True), 32768, float('nan')):
            with self.subTest(count=count), self.assertRaises(ValueError):
                Formula({'C': count})
        with self.assertRaises(TypeError):
            Formula([6, 12, 6])
        with self.assertRaises(TypeError):
            read_formula({'C': 6})

    def test_unsupported_elements_raise_at_every_public_formula_boundary(self):
        engine = object.__new__(Msbuddy)
        operations = [
            lambda: read_formula('C6H12O6Fe'),
            lambda: read_formula_str('H2O.Xe'),
            lambda: add_formula_str('H2O', 'SiO2'),
            lambda: formula_to_string({'Fe': 0}),
            lambda: enumerate_subformulas({'Xe': 1}),
            lambda: Formula({'Mg': 1}),
            lambda: MsbuddyConfig(element_ranges={'Fe': (0, 1)}),
            lambda: assign_subformula([], {'Ca': 1}, '[M+H]+'),
            lambda: assign_subformula([], 'H2O', '[M+Fe]+'),
            lambda: Adduct('[M+Fe]+', True),
            lambda: Adduct('[Fe]+', True),
            lambda: Adduct('[M+Fe+', True),
            lambda: MetaFeature('bad', 100, 1, adduct='[M+Fe]+'),
            lambda: engine.mz_to_formula(100, '[M+Fe]+'),
        ]
        for operation in operations:
            with self.subTest(operation=operation), self.assertRaisesRegex(UnsupportedElementError, 'Unsupported element'):
                operation()

    def test_subformula_assignment_accepts_strings_and_mappings(self):
        args = ([107.05, 149.02], '[M+H]+')
        by_string = assign_subformula(args[0], 'C15H16O5', args[1], ms2_tol=0.02, ppm=False)
        by_mapping = assign_subformula(args[0], {'C': 15, 'H': 16, 'O': 5}, args[1], ms2_tol=0.02, ppm=False)
        self.assertEqual([[match.formula for match in result.formulas] for result in by_mapping],
                         [[match.formula for match in result.formulas] for result in by_string])
        self.assertEqual([result.peak_index for result in by_mapping], [0, 1])
        for result in by_mapping:
            for match in result.formulas:
                self.assertAlmostEqual(match.mass - match.target_mass, match.mass_error)
        with self.assertRaisesRegex(ValueError, 'adduct produces invalid'):
            assign_subformula([], {'C': 1}, '[M-H]-')


class ElementRangeTests(unittest.TestCase):
    def test_defaults_and_explicit_ranges(self):
        config = MsbuddyConfig(element_ranges={'C': (2, 40), 'O': [1, 8]})
        self.assertEqual(config.element_ranges['C'], (2, 40))
        self.assertEqual(config.element_ranges['O'], (1, 8))
        self.assertEqual(config.element_ranges['H'], (0, 150))
        self.assertEqual(config.element_ranges['Cl'], (0, 0))
        self.assertEqual(config.ele_lower[SUPPORTED_ELEMENTS.index('O')], 1)
        self.assertEqual(config.ele_upper[SUPPORTED_ELEMENTS.index('C')], 40)
        with self.assertRaises(TypeError):
            config.element_ranges['C'] = (0, 1)
        halogen = MsbuddyConfig(halogen=True, element_ranges={'Cl': (1, 3)})
        self.assertEqual(halogen.element_ranges['Cl'], (1, 3))
        self.assertEqual(halogen.element_ranges['Br'], (0, 10))

    def test_invalid_ranges_are_rejected(self):
        for bounds in ((1,), (2, 1), (-1, 2), (0, 32768), (0, 1.5), (False, 1), '0,1'):
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                MsbuddyConfig(element_ranges={'C': bounds})
        with self.assertRaises(TypeError):
            MsbuddyConfig(element_ranges=[('C', (0, 1))])
        with self.assertRaisesRegex(ValueError, 'halogen=True'):
            MsbuddyConfig(element_ranges={'Cl': (0, 1)})

    def test_add_data_appends_and_validates_before_modifying(self):
        engine = object.__new__(Msbuddy)
        first, second, third = [MetaFeature(name, 100, 1) for name in ('first', 'second', 'third')]
        engine.add_data(first)
        engine.add_data([second])
        engine.add_data(iter([third]))
        self.assertEqual([item.identifier for item in engine.data], ['first', 'second', 'third'])
        with self.assertRaises(TypeError):
            engine.add_data([first, object()])
        self.assertEqual(len(engine.data), 3)
        engine.clear_data()
        self.assertIsNone(engine.data)


if __name__ == '__main__':
    unittest.main()
