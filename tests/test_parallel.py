"""Exercise candidate workers with a small, local formula database."""

import unittest

import numpy as np

from msbuddy.base import Formula, MetaFeature
from msbuddy.main import Msbuddy, MsbuddyConfig
from msbuddy.utils import form_arr_to_str, read_formula


class ParallelCandidateTests(unittest.TestCase):
    def test_workers_match_serial_results_without_receiving_models(self):
        formula = Formula(read_formula('C6H12O6'), 0)
        formula_array = formula.array
        neutral_mass = formula.mass
        masses = np.array([neutral_mass], dtype=np.float64)
        indices = np.searchsorted(masses, np.arange(15000) / 10)
        empty_masses = np.array([], dtype=np.float64)
        data = {
            'basic_db_mass': masses,
            'basic_db_formula': np.array([formula_array]),
            'basic_db_idx': indices,
            'halogen_db_mass': empty_masses,
            'halogen_db_formula': np.empty((0, 12), dtype=np.int16),
            'halogen_db_idx': np.zeros(15000, dtype=int),
            'common_frag_db': np.empty((0, 12), dtype=np.int16),
            'common_loss_db': np.empty((0, 12), dtype=np.int16),
            # This deliberately cannot be pickled by a spawned worker.
            'model_ms1_ms2': lambda x: x,
            'platt_a_0': 1.0,
        }
        engine = object.__new__(Msbuddy)
        engine._dependencies = data

        def run(parallel):
            engine.config = MsbuddyConfig(parallel=parallel, n_cpu=2)
            engine.data = [MetaFeature(identifier=i, mz=181.0707, charge=1) for i in range(2)]
            engine._preprocess_and_generate_candidate_formula()
            return [[form_arr_to_str(candidate.formula.array) for candidate in feature.candidate_formula_list]
                    for feature in engine.data]

        serial = run(False)
        parallel = run(True)
        self.assertEqual(serial, [['C6H12O6'], ['C6H12O6']])
        self.assertEqual(parallel, serial)


if __name__ == '__main__':
    unittest.main()
