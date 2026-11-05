"""Independent engines must keep their own input and database configuration."""

import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np

from msbuddy import Formula, MetaFeature, Msbuddy, MsbuddyConfig
from msbuddy.utils import read_formula


def formula_database(formula, mass):
    masses = np.array([mass], dtype=float)
    return {
        'basic_db_mass': masses,
        'basic_db_formula': np.array([Formula(read_formula(formula)).array], dtype=np.int16),
        'basic_db_idx': np.searchsorted(masses, np.arange(15000) / 10),
        'halogen_db_mass': np.empty(0),
        'halogen_db_formula': np.empty((0, 12), dtype=np.int16),
        'halogen_db_idx': np.zeros(15000, dtype=int),
    }


class EngineIsolationTests(unittest.TestCase):
    def test_engines_keep_separate_data_and_databases(self):
        glucose = formula_database('C6H12O6', 180.0633881)
        ethanol = formula_database('C2H6O', 46.0418648)
        with patch('msbuddy.main.init_db', side_effect=[glucose, ethanol]), \
             contextlib.redirect_stdout(io.StringIO()):
            first = Msbuddy(MsbuddyConfig(data_dir='/first'))
            first.add_data([MetaFeature('first', 181.0707, 1)])
            second = Msbuddy(MsbuddyConfig(data_dir='/second'))
            second.add_data([MetaFeature('second', 47.0491, 1)])
        self.assertIsNot(first, second)
        self.assertEqual(first.data[0].identifier, 'first')
        self.assertEqual(second.data[0].identifier, 'second')
        self.assertEqual([item.formula for item in first.mass_to_formula(180.0633881, 0.0001, False)], ['C6H12O6'])
        self.assertEqual([item.formula for item in second.mass_to_formula(46.0418648, 0.0001, False)], ['C2H6O'])
        first.clear_data()
        self.assertEqual(second.data[0].identifier, 'second')

    def test_invalid_timeout_uses_the_documented_default(self):
        with self.assertLogs(level='WARNING'):
            serial = MsbuddyConfig(timeout_secs=0)
            parallel = MsbuddyConfig(parallel=True, n_cpu=2, timeout_secs=-5)
        self.assertEqual(serial.timeout_secs, 300)
        self.assertEqual(parallel.timeout_secs, 330)


if __name__ == '__main__':
    unittest.main()
