"""Regression tests for the MGF input workflow."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from msbuddy.load import load_mgf


class LoadMgfTests(unittest.TestCase):
    def load_text(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.mgf'
            path.write_text(text, encoding='utf-8')
            return load_mgf(path)

    def test_spectra_are_merged_by_identifier_in_first_seen_order(self):
        features = self.load_text('''\
BEGIN IONS
TITLE=shared
PEPMASS=200
CHARGE=1+
MSLEVEL=2
80 8
40 4
END IONS
BEGIN IONS
FEATURE_ID=shared
PEPMASS=200
CHARGE=1+
MSLEVEL=1
201 2
200 10
END IONS
BEGIN IONS
TITLE=shared
PEPMASS=200
CHARGE=1+
MSLEVEL=2
99 999
END IONS
BEGIN IONS
PEPMASS=300
IONMODE=negative
RTINMINUTES=2
END IONS
BEGIN IONS
TITLE=0
PEPMASS=400
CHARGE=1-
END IONS
''')
        self.assertEqual([feature.identifier for feature in features], ['shared', 1, '0'])
        shared, anonymous, string_id = features
        np.testing.assert_array_equal(shared.ms2_raw.mz_array, [40, 80])
        np.testing.assert_array_equal(shared.ms2_raw.int_array, [4, 8])
        np.testing.assert_array_equal(shared.ms1_raw.mz_array, [200, 201])
        np.testing.assert_array_equal(shared.ms1_raw.int_array, [10, 2])
        self.assertEqual(anonymous.rt, 120)
        self.assertIsNone(anonymous.ms2_raw)
        self.assertFalse(anonymous.adduct.pos_mode)
        self.assertFalse(string_id.adduct.pos_mode)

    def test_empty_spectrum_can_be_filled_by_a_later_entry(self):
        features = self.load_text('''\
BEGIN IONS
TITLE=a
PEPMASS=100
CHARGE=1+
END IONS
BEGIN IONS
TITLE=a
PEPMASS=100
CHARGE=1+
20 5
END IONS
''')
        self.assertEqual(len(features), 1)
        np.testing.assert_array_equal(features[0].ms2_raw.mz_array, [20])

    def test_missing_precursor_is_reported(self):
        with self.assertRaisesRegex(ValueError, 'No precursor mz found'):
            self.load_text('BEGIN IONS\nCHARGE=1+\nEND IONS\n')

    def test_malformed_adduct_does_not_abort_loading_other_spectra(self):
        text = ''.join(
            f'BEGIN IONS\nTITLE={index}\nPEPMASS=100\nCHARGE=1+\n'
            f'ADDUCT={adduct}\n20 5\nEND IONS\n'
            for index, adduct in enumerate(('[M+H]0+', '[0M+H]+', '[M+H]oops+', '[M+2+1]+', '[M+Na]+'))
        )
        with self.assertLogs(level='WARNING') as messages:
            features = self.load_text(text)
        self.assertEqual([feature.identifier for feature in features], ['0', '1', '2', '3', '4'])
        self.assertEqual([feature.adduct.string for feature in features],
                         ['[M+H]+', '[M+H]+', '[M+H]+', '[M+H]+', '[M+Na]+'])
        self.assertEqual([(feature.adduct.m, feature.adduct.charge) for feature in features], [(1, 1)] * 5)
        self.assertEqual(len(messages.output), 4)


if __name__ == '__main__':
    unittest.main()
