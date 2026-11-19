"""Check that recall diagnostics locate real losses without changing selection."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from benchmarks.candidate_audit import CandidateTrace, first_loss, summarize
from benchmarks.formula_search import MassDeskSearch
from msbuddy import Formula, MetaFeature, MsbuddyConfig, Spectrum
from msbuddy.base import CandidateFormula, mass_e
from msbuddy.cand import gen_candidate_formula
from msbuddy.utils import read_formula


def empty_database():
    result = {}
    for prefix in ('basic', 'halogen'):
        result[prefix + '_db_mass'] = np.array([], dtype=float)
        result[prefix + '_db_formula'] = np.empty((0, 12), dtype=np.int16)
        result[prefix + '_db_idx'] = np.zeros(15000, dtype=int)
    return result


def glucose():
    feature = MetaFeature('glucose', 181.070665, 1,
                          ms2=Spectrum([91.038970], [100]))
    feature.data_preprocess(True, 5, 10, .02, 4, .01, 6)
    return feature


class CandidateAuditTests(unittest.TestCase):
    def test_neutral_loss_window_propagates_precursor_uncertainty_in_both_polarities(self):
        config = MsbuddyConfig()
        for charge in (1, -1):
            precursor = Formula(read_formula('C6H13O6' if charge > 0 else 'C6H11O6'), charge)
            fragment = Formula(read_formula('C6H11O5' if charge > 0 else 'C6H9O5'), charge)
            composition = [Formula(read_formula('H2O')), Formula(read_formula('C6H10O5'))]
            masses = np.array([formula.mass for formula in composition], dtype=float)
            database = empty_database()
            database.update(basic_db_mass=masses,
                            basic_db_formula=np.array([formula.array for formula in composition]),
                            basic_db_idx=np.searchsorted(masses, np.arange(15000) / 10))
            for propagate in (False, True):
                feature = MetaFeature('water_loss', precursor.mass * (1 + 4e-6), charge,
                                      ms2=Spectrum([fragment.mass], [100]))
                feature.data_preprocess(True, 5, 10, .02, 4, .01, 6)
                gen_candidate_formula(feature, True, 5, 10, 0, config.ele_lower,
                                       config.ele_upper, 4, database,
                                       propagate_loss_tolerance=propagate)
                names = [str(candidate.formula) for candidate in feature.candidate_formula_list]
                with self.subTest(charge=charge, propagated=propagate):
                    self.assertEqual('C6H12O6' in names, propagate)

    def test_missing_database_pair_can_be_recovered_without_a_precursor_database_hit(self):
        config = MsbuddyConfig()
        database = empty_database()
        feature = glucose()
        trace = CandidateTrace('C6H12O6', feature)
        gen_candidate_formula(feature, True, 5, 10, 0, config.ele_lower,
                               config.ele_upper, 4, database, trace=trace)
        self.assertFalse(trace.stages['precursor_database']['truth_present'])
        self.assertFalse(trace.stages['ms2_stitched']['truth_present'])
        self.assertEqual(feature.candidate_formula_list, [])
        self.assertIsNone(trace.queries[0]['fragment_matches'])
        self.assertEqual(trace.queries[0]['loss_matches'], 0)

        feature = glucose()
        trace = CandidateTrace('C6H12O6', feature)

        def de_novo(mass, fragment, *unused):
            return [Formula(read_formula('C3H7O3'), 1, mass=91.038970)] if fragment else [
                Formula(read_formula('C3H6O3'), 0, mass=90.031695)]

        gen_candidate_formula(feature, True, 5, 10, 0, config.ele_lower,
                               config.ele_upper, 4, database, trace=trace, fragment_query=de_novo)
        self.assertFalse(trace.stages['precursor_database']['truth_present'])
        self.assertTrue(trace.stages['ms2_stitched']['truth_present'])
        self.assertTrue(trace.stages['retained']['truth_present'])
        self.assertEqual(trace.queries[0]['truth_compatible_pairs'], 1)
        self.assertEqual([str(candidate.formula) for candidate in feature.candidate_formula_list], ['C6H12O6'])

    def test_chemical_filter_loss_is_distinct_from_search_loss(self):
        config = MsbuddyConfig(element_ranges={'C': (0, 5)})
        feature = glucose()
        trace = CandidateTrace('C6H12O6', feature)

        def de_novo(mass, fragment, *unused):
            return [Formula(read_formula('C3H7O3'), 1, mass=91.038970)] if fragment else [
                Formula(read_formula('C3H6O3'), 0, mass=90.031695)]

        gen_candidate_formula(feature, True, 5, 10, 0, config.ele_lower,
                               config.ele_upper, 4, empty_database(), trace=trace,
                               fragment_query=de_novo)
        self.assertTrue(trace.stages['ms2_stitched']['truth_present'])
        self.assertFalse(trace.stages['ms2_filtered']['truth_present'])
        self.assertEqual(first_loss(trace.stages, in_mass_window=True, has_ms2=True), 'chemical_filters')

    def test_candidate_pruning_is_visible_to_the_observer(self):
        feature = MetaFeature('pruning', 181.070665, 1)
        config = MsbuddyConfig()
        decoy = lambda: CandidateFormula(Formula(read_formula('C2H6O')),
                                         Formula(read_formula('C2H7O'), 1, mass=181.070665))
        truth = CandidateFormula(Formula(read_formula('C6H12O6')),
                                 Formula(read_formula('C6H13O6'), 1, mass=181.0714))
        candidates = [decoy() for _ in range(350)] + [truth]
        trace = CandidateTrace('C6H12O6', feature)
        with patch('msbuddy.cand._gen_candidate_formula_from_mz', return_value=(candidates, [])):
            gen_candidate_formula(feature, True, 5, 10, 0, config.ele_lower,
                                   config.ele_upper, 4, {}, trace=trace)
        self.assertTrue(trace.stages['valid']['truth_present'])
        self.assertFalse(trace.stages['retained']['truth_present'])
        self.assertEqual(first_loss(trace.stages, in_mass_window=True, has_ms2=False), 'candidate_pruning')

    def test_timeouts_have_their_own_denominator(self):
        complete = {'backend': 'database', 'status': 'ok', 'scores': [0.9], 'truth_rank': 1,
                    'stages': {'retained': {'truth_present': True}}, 'candidate_loss': None,
                    'search': {}, 'timing_seconds': {'total': .5}}
        timeout = {'backend': 'database', 'status': 'timeout'}
        report = summarize([complete, timeout])['database']
        self.assertEqual((report['attempted'], report['completed']), (2, 1))
        self.assertEqual(report['stage_recall_completed_only']['retained'],
                         {'correct': 1, 'denominator': 1})
        self.assertEqual(report['statuses'], {'ok': 1, 'timeout': 1})


class MassDeskAdapterTests(unittest.TestCase):
    @staticmethod
    def element_range(symbol, mass, minimum, maximum):
        return SimpleNamespace(symbol=symbol, mass=mass, min_count=minimum, max_count=maximum)

    def test_charged_mass_sign_cache_and_truncation_are_preserved(self):
        calls = []

        def decompose(target, ranges, **kwargs):
            calls.append((target, ranges, kwargs))
            return SimpleNamespace(results=[SimpleNamespace(composition={'C': 3, 'H': 7, 'O': 3},
                                                             mass=target)], truncated=True)

        query = MassDeskSearch(MsbuddyConfig(), decompose=decompose,
                               element_range=self.element_range)
        feature = glucose()
        query.configure(feature)
        for positive in (True, False):
            values = query(91.0, True, positive, False, False, 10, True, 0, {})
            self.assertAlmostEqual(calls[-1][0], 91.0 + (mass_e if positive else -mass_e))
            self.assertAlmostEqual(calls[-1][2]['tolerance_da'], 91.0 * 10e-6)
            self.assertAlmostEqual(values[0].mass, 91.0)
            self.assertEqual(values[0].charge, 1 if positive else -1)
        query(91.0, True, False, False, False, 10, True, 0, {})
        self.assertEqual(len(calls), 2)
        self.assertEqual(query.stats['truncated_queries'], 3)
        self.assertEqual(query.cached_decomposition.cache_info().hits, 1)
        self.assertEqual({item.symbol: item.min_count for item in calls[0][1]},
                         {item.symbol: 0 for item in calls[0][1]})

    def test_fallback_only_decomposes_empty_lookups_and_union_keeps_database_hits(self):
        calls = []

        def decompose(target, ranges, **kwargs):
            calls.append(target)
            return SimpleNamespace(results=[], truncated=False)

        config = MsbuddyConfig()
        query = MassDeskSearch(config, mode='fallback', decompose=decompose,
                               element_range=self.element_range)
        query.configure(glucose())
        database_hit = Formula(read_formula('CH4'))
        with patch('benchmarks.formula_search.query_fragnl_mass', return_value=[database_hit]):
            self.assertEqual(query(50, False, True, False, False, 10, True, 0, {}), [database_hit])
        self.assertEqual(calls, [])
        with patch('benchmarks.formula_search.query_fragnl_mass', return_value=[]):
            query(50, False, True, False, False, 10, True, 0, {})
        self.assertEqual(len(calls), 1)

        union = MassDeskSearch(config, mode='union', decompose=decompose,
                               element_range=self.element_range)
        union.configure(glucose())
        with patch('benchmarks.formula_search.query_fragnl_mass', return_value=[database_hit]):
            self.assertEqual(union(50, False, True, False, False, 10, True, 0, {}), [database_hit])
        self.assertEqual(len(calls), 1)  # unselected peak is kept on the fast path


if __name__ == '__main__':
    unittest.main()
