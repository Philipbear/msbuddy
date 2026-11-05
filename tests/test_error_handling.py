"""Failures should identify their cause and never retain stale annotations."""

import importlib
import unittest
from multiprocessing import TimeoutError as PoolTimeoutError
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from msbuddy.base import MetaFeature
from msbuddy.main import Msbuddy


class CandidateErrorHandlingTests(unittest.TestCase):
    def setUp(self):
        self.main = importlib.import_module('msbuddy.main')
        self.engine = object.__new__(Msbuddy)
        self.engine._dependencies = {}

    def feature(self, name):
        feature = MetaFeature(name, 100.0, 1)
        feature.candidate_formula_list = [object()]  # prior annotation must not survive failure
        return feature

    def test_serial_error_is_reported_and_stale_candidates_are_cleared(self):
        self.engine.config = SimpleNamespace(parallel=False)
        self.engine.data = [self.feature('serial')]
        with patch.object(self.main, '_generate_candidate_formula', side_effect=ValueError('bad spectrum')), \
             self.assertLogs(level='ERROR') as records:
            self.engine._preprocess_and_generate_candidate_formula()
        self.assertIn('Failed to process spectrum serial', records.output[0])
        self.assertIn('ValueError: bad spectrum', records.output[0])
        self.assertEqual(self.engine.data[0].candidate_formula_list, [])

    def test_keyboard_interrupt_is_not_swallowed(self):
        self.engine.config = SimpleNamespace(parallel=False)
        self.engine.data = [self.feature('interrupted')]
        with patch.object(self.main, '_generate_candidate_formula', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.engine._preprocess_and_generate_candidate_formula()

    def test_parallel_timeout_and_error_are_distinguished(self):
        self.engine.config = SimpleNamespace(parallel=True, n_cpu=2, timeout_secs=1)
        self.engine.data = [self.feature('timeout'), self.feature('error')]
        pool = MagicMock()
        pool.apply_async.side_effect = [
            SimpleNamespace(get=MagicMock(side_effect=PoolTimeoutError())),
            SimpleNamespace(get=MagicMock(side_effect=ValueError('bad spectrum'))),
        ]
        context = MagicMock()
        context.__enter__.return_value = pool
        with patch.object(self.main, 'Pool', return_value=context), \
             self.assertLogs(level='WARNING') as records:
            self.engine._preprocess_and_generate_candidate_formula()
        self.assertIn('Timed out processing spectrum timeout', records.output[0])
        self.assertIn('Failed to process spectrum error', records.output[1])
        self.assertTrue(all(feature.candidate_formula_list == [] for feature in self.engine.data))


if __name__ == '__main__':
    unittest.main()
