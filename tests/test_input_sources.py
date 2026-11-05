"""Exercise MGF, USI, and custom input edge cases without network access."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from msbuddy.load import _load_usi, load_mgf, load_usi
from msbuddy import UnsupportedElementError


class MgfInputTests(unittest.TestCase):
    def test_unsupported_adduct_element_is_not_silently_replaced(self):
        text = 'BEGIN IONS\nTITLE=bad\nPEPMASS=100\nCHARGE=1+\nADDUCT=[M+Fe]+\nEND IONS\n'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.mgf'
            path.write_text(text, encoding='utf-8')
            with self.assertRaisesRegex(UnsupportedElementError, 'Fe'):
                load_mgf(path)

    def test_unsupported_element_in_a_second_entry_is_rejected(self):
        text = ('BEGIN IONS\nTITLE=one\nPEPMASS=100\nCHARGE=1+\nEND IONS\n'
                'BEGIN IONS\nTITLE=one\nPEPMASS=100\nCHARGE=1+\nADDUCT=[M+Fe]+\nEND IONS\n')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.mgf'
            path.write_text(text, encoding='utf-8')
            with self.assertRaisesRegex(UnsupportedElementError, 'Fe'):
                load_mgf(path)

    def test_unsigned_charge_uses_adduct_polarity_and_missing_mode_warns(self):
        text = ('BEGIN IONS\nTITLE=negative\nPEPMASS=89.02\nCHARGE=2\nADDUCT=[M-2H]2-\nEND IONS\n'
                'BEGIN IONS\nTITLE=unknown\nPEPMASS=181.07\nEND IONS\n')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.mgf'
            path.write_text(text, encoding='utf-8')
            with self.assertLogs(level='WARNING') as records:
                negative, unknown = load_mgf(path)
        self.assertEqual((negative.adduct.string, negative.adduct.charge), ('[M-2H]2-', -2))
        self.assertEqual((unknown.adduct.string, unknown.adduct.charge), ('[M+H]+', 1))
        self.assertIn('assuming positive mode', records.output[0])

    def test_charge_without_adduct_preserves_magnitude(self):
        text = ('BEGIN IONS\nTITLE=double\nPEPMASS=91.04\nCHARGE=2+\nEND IONS\n'
                'BEGIN IONS\nTITLE=negative-double\nPEPMASS=89.02\nCHARGE=2-\nEND IONS\n')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.mgf'
            path.write_text(text, encoding='utf-8')
            positive, negative = load_mgf(path)
        self.assertEqual((positive.adduct.string, positive.adduct.charge), ('[M+2H]2+', 2))
        self.assertEqual((negative.adduct.string, negative.adduct.charge), ('[M-2H]2-', -2))

    def test_common_metadata_variations_and_empty_spectrum(self):
        text = '''\
# An optional MGF comment
  BEGIN IONS
TITLE=negative
IONMODE=Negative
CHARGE=1
ADDUCT=[M-H]-
PEPMASS=181.0707 5000
MSLEVEL=1
181.0707 100 1+
182.0741 7
  END IONS
BEGIN IONS
TITLE=positive-from-adduct
ADDUCT=[M+Na]+
PEPMASS=203.0526 2500
END IONS
'''
        text = text.replace('  BEGIN IONS\n', '  BEGIN IONS  \n', 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.mgf'
            path.write_text(text, encoding='utf-8')
            negative, positive = load_mgf(path)
        self.assertFalse(negative.adduct.pos_mode)
        self.assertEqual(negative.adduct.string, '[M-H]-')
        self.assertIsNone(negative.ms2_raw)
        np.testing.assert_array_equal(negative.ms1_raw.int_array, [100, 7])
        self.assertTrue(positive.adduct.pos_mode)
        self.assertEqual(positive.adduct.string, '[M+Na]+')
        self.assertIsNone(positive.ms2_raw)


class UsiInputTests(unittest.TestCase):
    def setUp(self):
        self.payloads = {
            'a': {'precursor_charge': 1, 'precursor_mz': 181.0707,
                  'peaks': [[100, 5], [50, 10]]},
            'b': {'precursor_charge': 1, 'precursor_mz': 181.0707, 'peaks': []},
            'bad': {'error': 'spectrum not found'},
        }

    def get(self, url, timeout):
        return SimpleNamespace(text=json.dumps(self.payloads[url.rsplit('=', 1)[1]]))

    def test_single_usi_and_adduct_string(self):
        with patch('msbuddy.load.get', side_effect=self.get):
            result = load_usi(' a ', '[M+Na]+')
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].identifier, 'a')
        self.assertEqual(result[0].adduct.string, '[M+Na]+')
        np.testing.assert_array_equal(result[0].ms2_raw.mz_array, [50, 100])

    def test_unsupported_adduct_element_is_not_silently_skipped(self):
        with patch('msbuddy.load.get', side_effect=self.get) as get:
            with self.assertRaisesRegex(UnsupportedElementError, 'Fe'):
                load_usi('a', '[M+Fe]+')
            with self.assertRaisesRegex(UnsupportedElementError, 'Fe'):
                load_usi(['a', 'b'], ['[M+Fe]+'])
            get.assert_not_called()

    def test_adduct_string_broadcasts_and_duplicate_keeps_first(self):
        with patch('msbuddy.load.get', side_effect=self.get), self.assertLogs(level='WARNING'):
            result = load_usi(['a', 'a', 'b'], '[M+Na]+')
        self.assertEqual([item.identifier for item in result], ['a', 'b'])
        self.assertEqual([item.adduct.string for item in result], ['[M+Na]+', '[M+Na]+'])
        self.assertIsNone(result[1].ms2_raw)

    def test_mismatched_adduct_list_uses_defaults_for_all_queries(self):
        with patch('msbuddy.load.get', side_effect=self.get), self.assertLogs(level='WARNING'):
            result = load_usi(['a', 'b'], ['[M+Na]+'])
        self.assertEqual([item.identifier for item in result], ['a', 'b'])
        self.assertEqual([item.adduct.string for item in result], ['[M+H]+', '[M+H]+'])

    def test_invalid_usi_is_skipped_and_zero_charge_uses_adduct_mode(self):
        self.payloads['b']['precursor_charge'] = 0
        with patch('msbuddy.load.get', side_effect=self.get), self.assertLogs(level='WARNING'):
            result = load_usi(['bad', 'b'], [None, '[M-H]-'])
        self.assertEqual([item.identifier for item in result], ['b'])
        self.assertEqual(result[0].adduct.string, '[M-H]-')
        self.assertIsNone(result[0].ms2_raw)

    def test_empty_peak_list_is_a_valid_query(self):
        with patch('msbuddy.load.get', side_effect=self.get):
            result = _load_usi('b')
        self.assertIsNone(result.ms2_raw)
        self.assertEqual(result.mz, 181.0707)

    def test_invalid_precursor_is_skipped_without_losing_other_queries(self):
        self.payloads['b']['precursor_mz'] = None
        with patch('msbuddy.load.get', side_effect=self.get), self.assertLogs(level='WARNING'):
            result = load_usi(['b', 'a'])
        self.assertEqual([item.identifier for item in result], ['a'])

    def test_usi_uses_reported_charge_when_adduct_is_missing(self):
        self.payloads['a']['precursor_charge'] = 2
        with patch('msbuddy.load.get', side_effect=self.get):
            result = load_usi('a')
        self.assertEqual((result[0].adduct.string, result[0].adduct.charge), ('[M+2H]2+', 2))


if __name__ == '__main__':
    unittest.main()
