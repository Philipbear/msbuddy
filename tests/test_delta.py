"""Physics and completeness checks for signed molecular-network differences."""

from dataclasses import FrozenInstanceError
from itertools import product, repeat
import json
import math
import random
import unittest

import numpy as np

from msbuddy.delta import (CARBON_ISOTOPE_DIFFERENCE, DEFAULT_DELTA_RANGES,
                          BROAD_DELTA_RANGES, ELECTRON_MASS, MONOISOTOPIC_MASSES, DeltaFormula,
                          DeltaSearchSpaceError, DeltaTransformation,
                          annotate_delta_mass, annotate_network_delta)
from msbuddy.utils import SUPPORTED_ELEMENTS, UnsupportedElementError, read_formula


def ion_mz(formula, adduct_counts, charge=1, multiplier=1):
    """Independent explicit ion mass; no production adduct parser is called."""
    composition = read_formula(formula) if isinstance(formula, str) else formula
    atoms = {e: multiplier * composition.get(e, 0) + adduct_counts.get(e, 0)
             for e in SUPPORTED_ELEMENTS}
    return (math.fsum(atoms[e] * MONOISOTOPIC_MASSES[e] for e in SUPPORTED_ELEMENTS)
            - charge * ELECTRON_MASS) / abs(charge)


class DeltaFormulaTests(unittest.TestCase):
    def test_signed_formula_round_trip_arithmetic_and_precise_mass(self):
        delta = DeltaFormula({'H': -1, 'Na': np.int64(1), 'O': 0})
        self.assertEqual(str(delta), '-H+Na')
        self.assertEqual(delta.composition, {'H': -1, 'Na': 1})
        self.assertAlmostEqual(delta.mass, 21.98194424977, places=10)
        self.assertEqual(delta.changed_atoms, 2)
        self.assertEqual(DeltaFormula.from_string('-H+Na'), delta)
        self.assertEqual(DeltaFormula.from_string('+Na-H'), delta)
        self.assertEqual(str(-delta), '+H-Na')
        self.assertFalse(delta + (-delta))
        self.assertEqual(str(delta - delta), '0')
        self.assertEqual(DeltaFormula.from_string('0'), delta - delta)
        self.assertEqual(DeltaFormula.from_string('-H2O').composition, {'H': -2, 'O': -1})
        self.assertEqual(DeltaFormula.from_string('+O-H2').composition, {'H': -2, 'O': 1})
        self.assertEqual(DeltaFormula.from_string('+2H2O-H2O'), DeltaFormula.from_string('H2O'))
        self.assertEqual(DeltaFormula.from_string('C2H4O'), DeltaFormula({'C': 2, 'H': 4, 'O': 1}))
        for element in SUPPORTED_ELEMENTS:
            delta = DeltaFormula({element: -3})
            self.assertEqual(DeltaFormula.from_string(str(delta)), delta)
            self.assertAlmostEqual(delta.mass, -3 * MONOISOTOPIC_MASSES[element], places=10)

    def test_immutable_counts_and_detached_composition(self):
        delta = DeltaFormula({'O': 1})
        copied = delta.composition
        copied['O'] = 9
        self.assertEqual(delta.composition, {'O': 1})
        with self.assertRaises(FrozenInstanceError):
            delta.counts = (0,) * 12
        self.assertEqual({delta: 'value'}[DeltaFormula({'O': 1})], 'value')

    def test_source_propagation_is_not_a_signed_molecular_formula(self):
        change = DeltaFormula.from_string('O-H2')
        self.assertEqual(change.apply_to('C2H6O'), {'C': 2, 'H': 4, 'O': 2})
        with self.assertRaisesRegex(ValueError, 'more atoms'):
            DeltaFormula({'H': -7}).apply_to('C2H6O')
        with self.assertRaises(ValueError):
            DeltaFormula({'C': 1}).apply_to({'C': 32767})
        from msbuddy import Formula
        with self.assertRaises(ValueError):
            Formula({'H': -1, 'Na': 1})
        with self.assertRaisesRegex(ValueError, 'neutral'):
            change.apply_to(Formula({'C': 2, 'H': 6}, charge=1))

    def test_rejects_invalid_signed_input(self):
        for value in (True, np.bool_(False), 1.2, float('nan'), -32768, 32768):
            with self.subTest(value=value), self.assertRaises(ValueError):
                DeltaFormula({'C': value})
        for text in ('', '+', '++H', 'H-', 'H+-O', 'C0', '[13C]', 'Fe', 'H2+', 'H 2'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                DeltaFormula.from_string(text)
        with self.assertRaises(UnsupportedElementError):
            DeltaFormula({'Xe': 0})
        with self.assertRaises(TypeError):
            DeltaFormula([0] * 12)
        with self.assertRaises(TypeError):
            DeltaFormula.from_string({'O': 1})


class SignedSearchTests(unittest.TestCase):
    def test_complete_matches_against_small_brute_force_oracle(self):
        rng = random.Random(613)
        for trial in range(24):
            elements = rng.sample(list(SUPPORTED_ELEMENTS), rng.randint(2, 6))
            bounds = {e: (rng.randint(-2, 0), rng.randint(0, 2)) for e in elements}
            target_counts = {e: rng.randint(*bounds[e]) for e in elements}
            target_mass = DeltaFormula(target_counts).mass + rng.uniform(-.01, .01)
            tolerance = rng.choice((.01, .1, 2.))
            max_atoms, max_elements = rng.randint(0, 10), rng.randint(0, len(elements))
            source = {e: rng.randint(0, 3) for e in SUPPORTED_ELEMENTS} if trial % 2 else None
            expected = set()
            for counts in product(*(range(bounds[e][0], bounds[e][1] + 1) for e in elements)):
                delta = DeltaFormula(dict(zip(elements, counts)))
                if sum(abs(n) for n in counts) > max_atoms or sum(n != 0 for n in counts) > max_elements:
                    continue
                if source is not None and any(source[e] + n < 0 for e, n in zip(elements, counts)):
                    continue
                if abs(delta.mass - target_mass) <= tolerance:
                    expected.add(delta)
            result = annotate_delta_mass(target_mass, tolerance=tolerance, element_ranges=bounds,
                                         max_changed_atoms=max_atoms, max_changed_elements=max_elements,
                                         source_formula=source, max_steps=0)
            self.assertEqual({candidate.delta for candidate in result}, expected, 'trial %d' % trial)
            self.assertEqual(len(result), len(expected))
            self.assertTrue(result.complete)

    def test_tolerance_boundary_and_no_mass_rounding(self):
        # Broad cancellation across heavy atoms, plus precise Na/H exchange.
        for counts in ({'H': -1, 'Na': 1}, {'C': -6, 'I': 1, 'F': -2}):
            delta = DeltaFormula(counts)
            bounds = {e: (n, n) for e, n in counts.items()}
            exact = annotate_delta_mass(delta.mass, tolerance=0., element_ranges=bounds)
            self.assertEqual([row.delta for row in exact], [delta])
            observed = delta.mass + .005
            boundary = abs(delta.mass - observed)
            self.assertEqual(len(annotate_delta_mass(observed, tolerance=boundary, element_ranges=bounds)), 1)
            self.assertEqual(len(annotate_delta_mass(observed, tolerance=np.nextafter(boundary, 0.),
                                                     element_ranges=bounds)), 0)

    def test_sign_reversal_returns_negated_candidates_and_errors(self):
        bounds = {'C': (-3, 3), 'H': (-8, 8), 'O': (-3, 3), 'Na': (-1, 1)}
        forward = annotate_delta_mass(21.982, tolerance=.01, element_ranges=bounds)
        reverse = annotate_delta_mass(-21.982, tolerance=.01, element_ranges=bounds)
        expected = {-row.delta: -row.mass_error for row in forward}
        self.assertEqual(set(expected), {row.delta for row in reverse})
        for row in reverse:
            self.assertAlmostEqual(row.mass_error, expected[row.delta], places=12)

    def test_zero_is_a_real_candidate_and_custom_bounds_replace_defaults(self):
        result = annotate_delta_mass(0., tolerance=0., element_ranges={})
        self.assertEqual(len(result), 1)
        self.assertFalse(result[0].delta)
        self.assertEqual(result[0].formula, '0')
        self.assertEqual(result[0].hypotheses, ())
        self.assertEqual(set(result.element_ranges.values()), {(0, 0)})
        self.assertFalse(annotate_delta_mass(0., tolerance=0., element_ranges={'C': (1, 1)}))

    def test_source_constraints_and_explicit_empty_effective_domain(self):
        result = annotate_delta_mass(DeltaFormula({'H': -2}).mass, tolerance=.001,
                                     source_formula='CH4', element_ranges={'H': (-6, 6)})
        self.assertEqual(result[0].target_formula, 'CH2')
        self.assertEqual(result[0].target_composition, {'C': 1, 'H': 2})
        impossible = annotate_delta_mass(-6.047, tolerance=.01, source_formula='CH4',
                                         element_ranges={'H': (-6, -5)})
        self.assertFalse(impossible)
        self.assertTrue(impossible.complete)

    def test_default_all_elements_and_common_rule_coverage(self):
        self.assertEqual(set(DEFAULT_DELTA_RANGES), set(SUPPORTED_ELEMENTS))
        for element in ('F', 'Cl', 'Br', 'I', 'Na', 'K'):
            delta = DeltaFormula({element: 1, 'H': -1})
            result = annotate_delta_mass(delta.mass, tolerance=.00001)
            self.assertIn(delta, [row.delta for row in result])
        for text in ('C6H10O5', 'C6H8O6', 'HPO3', 'SO3'):
            delta = DeltaFormula.from_string(text)
            result = annotate_delta_mass(delta.mass)
            self.assertEqual(result[0].delta, delta)
            self.assertTrue(result[0].hypotheses)

    def test_broad_preset_matches_explicit_settings_and_permits_overrides(self):
        delta = DeltaFormula.from_string('C8H12O4')
        source = 'C6H12O6'
        default = annotate_delta_mass(delta.mass, source_formula=source)
        broad = annotate_delta_mass(delta.mass, source_formula=source, preset='broad')
        explicit = annotate_delta_mass(delta.mass, source_formula=source,
                                        element_ranges=BROAD_DELTA_RANGES,
                                        max_changed_atoms=40, max_changed_elements=6)
        self.assertNotIn(delta, [row.delta for row in default])
        self.assertIn(delta, [row.delta for row in broad])
        self.assertEqual([row.delta for row in broad], [row.delta for row in explicit])
        self.assertEqual(broad.preset, 'broad')
        self.assertEqual(broad.max_changed_atoms, 40)
        self.assertEqual(broad.max_changed_elements, 6)
        custom = annotate_delta_mass(0., preset='broad', element_ranges={},
                                      max_changed_atoms=0, max_changed_elements=0)
        self.assertEqual(len(custom), 1)
        self.assertEqual(custom.max_changed_atoms, 0)
        self.assertEqual(custom.max_changed_elements, 0)
        self.assertEqual(custom.element_ranges, {e: (0, 0) for e in SUPPORTED_ELEMENTS})

    def test_multistep_net_hypotheses_dont_restrict_enumeration(self):
        delta = DeltaFormula.from_string('O2')
        with_steps = annotate_delta_mass(delta.mass, element_ranges={'O': (0, 3)})
        without_steps = annotate_delta_mass(delta.mass, element_ranges={'O': (0, 3)}, max_steps=0)
        self.assertEqual([row.delta for row in with_steps], [row.delta for row in without_steps])
        self.assertEqual(len(with_steps[0].hypotheses[0].steps), 2)
        self.assertEqual(without_steps[0].hypotheses, ())
        custom = DeltaTransformation('synthetic oxygen rule', DeltaFormula({'O': 1}))
        result = annotate_delta_mass(delta.mass, element_ranges={'O': (0, 3)}, transformations=[custom])
        self.assertEqual(result[0].hypotheses[0].steps, (custom.name, custom.name))

    def test_many_named_explanations_preserve_order_and_deduplicate(self):
        # A finite, permitted custom list may contain many names for one net
        # change. It must retain each explanation without quadratic list scans.
        delta = DeltaFormula({'O': 1})
        rules = [DeltaTransformation('oxygen route %d' % i, delta) for i in range(5000)]
        result = annotate_delta_mass(delta.mass, element_ranges={'O': (1, 1)},
                                     transformations=rules + rules[:10], max_steps=1)
        self.assertEqual([row.steps for row in result[0].hypotheses],
                         [(rule.name,) for rule in rules])

    def test_isotope_hypotheses_separate_from_signed_element_formulas(self):
        result = annotate_delta_mass(CARBON_ISOTOPE_DIFFERENCE, tolerance=.0001, element_ranges={})
        self.assertFalse(result)
        self.assertEqual(len(result.phenomena), 1)
        self.assertIn('13C-like', result.phenomena[0].name)
        self.assertEqual(result.phenomena[0].mass_error, 0.)

    def test_ppm_reference_required_and_undefined_delta_ppm_is_none(self):
        with self.assertRaisesRegex(ValueError, 'reference_mass'):
            annotate_delta_mass(0., ppm=True)
        result = annotate_delta_mass(0.001, tolerance=5, ppm=True, reference_mass=500, element_ranges={})
        self.assertAlmostEqual(result.tolerance_da, .0025)
        self.assertAlmostEqual(result[0].mass_error_ppm, -2.)
        self.assertIsNone(annotate_delta_mass(0., element_ranges={})[0].mass_error_ppm)

    def test_resource_limits_raise_instead_of_returning_partial_lists(self):
        with self.assertRaisesRegex(DeltaSearchSpaceError, 'Half-space'):
            annotate_delta_mass(0., max_half_states=1)
        with self.assertRaisesRegex(DeltaSearchSpaceError, 'max_candidates'):
            annotate_delta_mass(0., tolerance=100, element_ranges={'H': (-2, 2)}, max_candidates=1)
        with self.assertRaisesRegex(DeltaSearchSpaceError, 'Too many transformation rules'):
            annotate_delta_mass(0., transformations=repeat(DeltaTransformation('O', DeltaFormula({'O': 1}))))
        # The L0 limit is reported and can explicitly be removed.
        delta = DeltaFormula({'C': 1, 'H': 1, 'N': 1, 'O': 1, 'S': 1})
        ranges = {e: (n, n) for e, n in delta.composition.items()}
        self.assertFalse(annotate_delta_mass(delta.mass, element_ranges=ranges))
        result = annotate_delta_mass(delta.mass, element_ranges=ranges, max_changed_elements=12)
        self.assertEqual(result[0].delta, delta)
        self.assertEqual(result.max_changed_elements, 12)

    def test_invalid_search_parameters(self):
        for kwargs in ({'delta_mass': float('nan')}, {'tolerance': -1}, {'tolerance': True},
                       {'ppm': 'yes'}, {'include_ion': 1}, {'reference_mass': 0},
                       {'max_changed_atoms': True}, {'max_changed_elements': 13},
                       {'max_steps': 4}, {'max_steps': 1.5}, {'max_candidates': 0},
                       {'preset': 'unknown'}, {'preset': None}, {'preset': {}},
                       {'element_ranges': {'C': (2, 1)}}, {'element_ranges': {'C': (0, .5)}},
                       {'element_ranges': {'C': (0, True)}}, {'element_ranges': {'C': '12'}},
                       {'element_ranges': {'C': (0,)}}, {'element_ranges': {'Xe': (0, 0)}}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                annotate_delta_mass(**dict({'delta_mass': 0.}, **kwargs))
        with self.assertRaises(TypeError):
            annotate_delta_mass(0., element_ranges=[])
        with self.assertRaises(TypeError):
            annotate_delta_mass(0., transformations=['oxidation'])

    def test_no_input_mutation_and_detached_json_export(self):
        source = {'C': 6, 'H': 12, 'O': 6}
        ranges = {'C': [-2, 2], 'H': [-4, 4], 'O': [-2, 2]}
        rules = [DeltaTransformation('oxygen', DeltaFormula({'O': 1}))]
        before = (json.dumps(source, sort_keys=True), json.dumps(ranges, sort_keys=True), list(rules))
        result = annotate_delta_mass(MONOISOTOPIC_MASSES['O'], element_ranges=ranges,
                                     source_formula=source, transformations=rules)
        after = (json.dumps(source, sort_keys=True), json.dumps(ranges, sort_keys=True), list(rules))
        self.assertEqual(before, after)
        record = result.to_dict()
        record['search']['element_ranges']['O'][0] = -99
        record['search']['transformations'][0]['delta']['O'] = 999
        self.assertEqual(result.element_ranges['O'], (-2, 2))
        self.assertEqual(result.transformations[0].delta.composition, {'O': 1})

    def test_nonfinite_and_overflowing_numeric_inputs(self):
        for value in (float('inf'), -float('inf'), float('nan'), 10 ** 1000):
            for parameter in ('delta_mass', 'tolerance', 'reference_mass'):
                with self.subTest(value=value, parameter=parameter), self.assertRaises(ValueError):
                    annotate_delta_mass(**dict({'delta_mass': 0.}, **{parameter: value}))
        with self.assertRaisesRegex(ValueError, 'Converted tolerance'):
            annotate_delta_mass(0., tolerance=1e308, ppm=True, reference_mass=1e308)
        with self.assertRaises(DeltaSearchSpaceError):
            annotate_delta_mass(0., element_ranges={e: (-32767, 32767) for e in SUPPORTED_ELEMENTS},
                                max_changed_atoms=32767, max_changed_elements=12)


class NetworkDeltaTests(unittest.TestCase):
    def test_network_small_arguments_reject_unbounded_or_ambiguous_inputs(self):
        for value in (repeat(5.), iter([5., 5., 5.]), b'55', '55'):
            with self.subTest(value=type(value).__name__), self.assertRaisesRegex(ValueError, 'mz_tolerance'):
                annotate_network_delta(500., 500., '[M+H]+', '[M+H]+', mz_tolerance=value,
                                       element_ranges={})
        result = annotate_network_delta(500., 500., '[M+H]+', '[M+H]+',
                                         mz_tolerance=iter([5., 10.]), element_ranges={})
        self.assertAlmostEqual(result.tolerance_da, .0075)
        for value in (np.array(['rules']), np.array(['rules', 'balanced']), ['rules']):
            with self.subTest(value=repr(value)), self.assertRaisesRegex(ValueError, 'ranking'):
                annotate_network_delta(500., 500., '[M+H]+', '[M+H]+', ranking=value,
                                       element_ranges={})

    def test_proton_sodium_edge_distinguishes_neutral_and_ion_changes(self):
        formula = 'C6H12O6'
        src = ion_mz(formula, {'H': 1})
        dst = ion_mz(formula, {'Na': 1})
        result = annotate_network_delta(src, dst, '[M+H]+', '[M+Na]+', source_formula=formula)
        self.assertEqual(result.mass_space, 'neutral')
        self.assertAlmostEqual(result.delta_mass, 0., places=12)
        self.assertFalse(result[0].delta)
        self.assertEqual(result[0].target_formula, formula)
        self.assertEqual(result.adduct_delta, DeltaFormula({'H': -1, 'Na': 1}))
        self.assertEqual(result[0].ion_delta, result.adduct_delta)
        self.assertAlmostEqual(result.observed_mz_difference, result.adduct_delta.mass, places=10)
        self.assertAlmostEqual(result.tolerance_da, (src + dst) * 5e-6, places=12)
        self.assertAlmostEqual(result.source_mass_error, 0., places=10)
        self.assertAlmostEqual(result[0].target_mass_error, 0., places=10)
        raw = annotate_delta_mass(dst - src, tolerance=.0001)
        self.assertEqual(raw[0].delta, result.adduct_delta)
        self.assertTrue(any('ion' in hypothesis.kinds for hypothesis in raw[0].hypotheses))

    def test_unequal_charges_multimers_and_polarities(self):
        formula = 'C6H12O6'
        source = ion_mz(formula, {'H': 1})
        controls = [('[M+2H]2+', {'H': 2}, 2, 1),
                    ('[2M+H]+', {'H': 1}, 1, 2),
                    ('[M-H]-', {'H': -1}, -1, 1),
                    ('[2M-2H]2-', {'H': -2}, -2, 2),
                    ('[M]+', {}, 1, 1), ('[M]2+', {}, 2, 1)]
        for adduct, counts, charge, multiplier in controls:
            target = ion_mz(formula, counts, charge, multiplier)
            with self.subTest(adduct=adduct):
                result = annotate_network_delta(source, target, '[M+H]+', adduct,
                                                 source_formula=formula, element_ranges={})
                self.assertEqual(len(result), 1)
                self.assertFalse(result[0].delta)
                expected = {e: (multiplier - 1) * read_formula(formula).get(e, 0)
                               + counts.get(e, 0) - (1 if e == 'H' else 0) for e in SUPPORTED_ELEMENTS}
                self.assertEqual(result[0].ion_delta, DeltaFormula(expected))
                atomic_mass_difference = result[0].ion_delta.mass
                ion_mass_difference = abs(charge) * target - source
                self.assertAlmostEqual(ion_mass_difference,
                                       atomic_mass_difference - (charge - 1) * ELECTRON_MASS, places=10)
                self.assertAlmostEqual(result.tolerance_da,
                    5e-6 * (source + abs(charge) * target / multiplier), places=12)
                without_formula = annotate_network_delta(source, target, '[M+H]+', adduct, element_ranges={})
                self.assertEqual(without_formula[0].ion_delta is None, multiplier != 1)

    def test_adduct_alias_and_explicit_adduct_object(self):
        from msbuddy import Adduct
        source = ion_mz('C6H12O6', {'H': 1})
        target = ion_mz('C6H12O6', {'C': 2, 'H': 4, 'N': 1})
        result = annotate_network_delta(source, target, Adduct('[M+H]+', True), '[M+ACN+H]+',
                                         source_formula='C6H12O6', element_ranges={})
        self.assertEqual(len(result), 1)
        self.assertEqual(result.adduct_delta, DeltaFormula({'C': 2, 'H': 3, 'N': 1}))

    def test_biotransformation_plus_adduct_exchange_and_reverse_edge(self):
        source_formula, target_formula = 'C10H16O2', 'C10H16O3'
        source = ion_mz(source_formula, {'H': 1})
        target = ion_mz(target_formula, {'Na': 1})
        forward = annotate_network_delta(source, target, '[M+H]+', '[M+Na]+',
                                         source_formula=source_formula, element_ranges={'O': (-2, 2)})
        reverse = annotate_network_delta(target, source, '[M+Na]+', '[M+H]+',
                                         source_formula=target_formula, element_ranges={'O': (-2, 2)})
        self.assertEqual(forward[0].delta, DeltaFormula({'O': 1}))
        self.assertEqual(reverse[0].delta, -forward[0].delta)
        self.assertEqual(forward[0].ion_delta, DeltaFormula({'O': 1, 'H': -1, 'Na': 1}))
        self.assertEqual(reverse[0].ion_delta, -forward[0].ion_delta)
        self.assertEqual(forward[0].target_formula, target_formula)

    def test_individual_anchor_and_target_tolerances_are_checked(self):
        exact = ion_mz('C6H12O6', {'H': 1})
        with self.assertRaisesRegex(ValueError, 'source_formula does not match'):
            annotate_network_delta(exact + .003, exact, '[M+H]+', '[M+H]+',
                                   source_formula='C6H12O6', ppm=False, mz_tolerance=.001)
        # Observed delta=0 alone would pass; the known exact anchor means the
        # target's much smaller individual tolerance rules out this target.
        result = annotate_network_delta(exact + .0008, exact + .0008, '[M+H]+', '[M+H]+',
                                        source_formula='C6H12O6', ppm=False,
                                        mz_tolerance=(.001, .0001), element_ranges={})
        self.assertFalse(result)
        result = annotate_network_delta(exact + .0008, exact, '[M+H]+', '[M+H]+',
                                        source_formula='C6H12O6', ppm=False,
                                        mz_tolerance=(.001, .0001), element_ranges={})
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0].mass_error, .0008, places=10)
        self.assertAlmostEqual(result[0].target_mass_error, 0., places=10)

    def test_adduct_subtraction_must_be_possible_for_anchor_and_target(self):
        source = ion_mz('C', {'H': -1}, charge=-1)
        with self.assertRaisesRegex(ValueError, 'source_adduct removes'):
            annotate_network_delta(source, source, '[M-H]-', '[M-H]-', source_formula='C')
        source = ion_mz('C', {})
        target = ion_mz('C', {'H': -1}, charge=-1)
        result = annotate_network_delta(source, target, '[M]+', '[M-H]-', source_formula='C', element_ranges={})
        self.assertFalse(result)

    def test_ppm_is_propagated_from_precursors_not_tiny_difference(self):
        source = 500.
        target = 500.0001
        result = annotate_network_delta(source, target, '[M+H]+', '[M+H]+',
                                         mz_tolerance=(5., 10.), element_ranges={})
        self.assertAlmostEqual(result.tolerance_da, source * 5e-6 + target * 10e-6)
        self.assertEqual(len(result), 1)
        self.assertIsNone(result[0].mass_error_ppm)

    def test_json_export_preserves_direction_bounds_and_provenance(self):
        source = ion_mz('C6H12O6', {'H': 1})
        target = ion_mz('C6H12O6', {'Na': 1})
        result = annotate_network_delta(source, target, '[M+H]+', '[M+Na]+',
                                         source_formula='C6H12O6', element_ranges={}, max_steps=0)
        record = json.loads(json.dumps(result.to_dict(), allow_nan=False))
        self.assertEqual(record['schema'], 'msbuddy.signed_delta.v1')
        self.assertEqual(record['candidate_count'], 1)
        self.assertEqual(record['candidates'][0]['delta'], {})
        self.assertEqual(record['candidates'][0]['ion_delta'], {'H': -1, 'Na': 1})
        self.assertEqual(record['network']['adduct_delta'], {'H': -1, 'Na': 1})
        self.assertEqual(record['network']['source_charge'], 1)
        self.assertEqual(record['network']['target_multiplier'], 1)
        self.assertEqual(record['network']['source_mz'], source)
        self.assertEqual(record['source_composition'], {'C': 6, 'H': 12, 'O': 6})
        self.assertEqual(record['search']['max_steps'], 0)
        self.assertEqual(record['search']['element_ranges']['F'], [0, 0])
        self.assertTrue(record['complete_within_bounds'])
        self.assertFalse(record['truncated'])
        record['candidates'][0]['target_composition']['C'] = 99
        self.assertEqual(result[0].target_composition['C'], 6)

    def test_balanced_ranking_is_explicit_and_changes_no_candidates(self):
        source = ion_mz('C10H16O2', {'H': 1})
        target = ion_mz('C10H16O3', {'Na': 1})
        rules = annotate_network_delta(source, target, '[M+H]+', '[M+Na]+',
                                         source_formula='C10H16O2')
        balanced = annotate_network_delta(source, target, '[M+H]+', '[M+Na]+',
                                            source_formula='C10H16O2', ranking='balanced')
        self.assertEqual(rules.ranking_method, 'rules')
        self.assertTrue(all(row.heuristic_score is None for row in rules))
        self.assertEqual(balanced.ranking_method, 'balanced')
        self.assertEqual({row.delta for row in rules}, {row.delta for row in balanced})
        self.assertTrue(all(np.isfinite(row.heuristic_score) for row in balanced))
        values = [row.heuristic_score for row in balanced]
        self.assertEqual(values, sorted(values))
        self.assertEqual(balanced.to_dict()['ranking'], 'balanced')
        self.assertEqual(balanced.to_dict()['candidates'][0]['heuristic_score'], values[0])
        for row in balanced:
            counts = row.target_composition
            dbe = counts.get('C', 0) + 1 + (counts.get('N', 0) + counts.get('P', 0)) / 2
            dbe -= sum(counts.get(e, 0) for e in ('H', 'F', 'Cl', 'Br', 'I', 'Na', 'K')) / 2
            prior = 1 if not row.delta else 1 / len(row.hypotheses[0].steps) if row.hypotheses else 0
            expected = .8 * math.log1p(row.delta.changed_atoms) + .2 * len(row.delta.composition)
            expected += -1.5 * prior + .5 * min(10, max(0, -dbe))
            self.assertAlmostEqual(row.heuristic_score, expected, places=12)

    def test_balanced_requires_anchor_and_supports_zero_tolerance(self):
        exact = ion_mz('C6H12O6', {'H': 1})
        with self.assertRaisesRegex(ValueError, 'requires a known source_formula'):
            annotate_network_delta(exact, exact, '[M+H]+', '[M+H]+', ranking='balanced')
        with self.assertRaisesRegex(ValueError, 'ranking must'):
            annotate_network_delta(exact, exact, '[M+H]+', '[M+H]+', ranking='magic')
        result = annotate_network_delta(exact, exact, '[M+H]+', '[M+H]+', source_formula='C6H12O6',
                                         ranking='balanced', element_ranges={}, mz_tolerance=0)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].heuristic_score, -1.5)

    def test_isotope_effect_is_not_forced_into_a_neutral_delta(self):
        source = ion_mz('C10H20O2', {'H': 1}, multiplier=2)
        target = source + CARBON_ISOTOPE_DIFFERENCE
        result = annotate_network_delta(source, target, '[2M+H]+', '[2M+H]+',
                                         element_ranges={}, ppm=False, mz_tolerance=.0001)
        self.assertFalse(result)
        self.assertTrue(any(abs(item.mass_error) < 1e-10 for item in result.phenomena))

    def test_invalid_adducts_inputs_and_ambiguous_options_fail_explicitly(self):
        for adduct in (None, 'Na', '[M+H]', '[M+H]0+', '[0M+H]+', '[M+Fe]+', '[M++H]+',
                       '[M+H]++', '[M+H]+junk', '[M+H]2.0+', '[M-H]0-'):
            with self.subTest(adduct=adduct), self.assertRaises((ValueError, TypeError)):
                annotate_network_delta(100., 110., '[M+H]+', adduct)
        for kwargs in ({'source_mz': float('nan')}, {'target_mz': 0.}, {'ppm': 'yes'},
                       {'mz_tolerance': -1}, {'mz_tolerance': (1,)},
                       {'mz_tolerance': (True, 1)}, {'mz_tolerance': object()}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                annotate_network_delta(**dict(dict(source_mz=100., target_mz=110.,
                                                    source_adduct='[M+H]+', target_adduct='[M+H]+'), **kwargs))
        with self.assertRaisesRegex(TypeError, 'tolerance'):
            annotate_network_delta(100., 110., '[M+H]+', '[M+H]+', tolerance=.01)
        with self.assertRaisesRegex(ValueError, 'invalid neutral'):
            annotate_network_delta(.1, .2, '[M+H]+', '[M+H]+')


if __name__ == '__main__':
    unittest.main()
