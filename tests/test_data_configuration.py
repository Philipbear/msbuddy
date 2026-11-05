"""Protect configurable data files and the historical command-line options."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from msbuddy import main_cmd
from msbuddy.load import DATA_FILES, get_data_files, init_db, resolve_data_dir


class DataConfigurationTests(unittest.TestCase):
    def test_data_paths_resolve_relative_and_absolute_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            existing = base / 'custom-formulas.joblib'
            existing.touch()
            model = base / 'shared' / 'model.joblib'
            files = get_data_files(base, {'formula_db': existing.name, 'ml_model': model})
            self.assertEqual(files['formula_db']['path'], existing)
            self.assertTrue(files['formula_db']['exists'])
            self.assertEqual(files['ml_model']['path'], model)
            self.assertFalse(files['ml_model']['exists'])
            self.assertEqual(files['common_db']['path'], base / DATA_FILES['common_db']['filename'])
            with self.assertRaisesRegex(ValueError, 'Unknown data_files'):
                get_data_files(base, {'unknown': 'file.joblib'})
        self.assertEqual(resolve_data_dir(), Path(main_cmd.__file__).parent / 'data')

    def test_missing_custom_database_does_not_download_a_replacement(self):
        with tempfile.TemporaryDirectory() as directory, patch('msbuddy.load.download') as download:
            with self.assertRaisesRegex(FileNotFoundError, 'configured msbuddy data file'):
                init_db(data_dir=directory, data_files={'common_db': 'missing.joblib'})
            download.assert_not_called()

    def test_legacy_and_current_cli_options_pass_the_same_configuration(self):
        def get_config_args(arguments):
            with tempfile.TemporaryDirectory() as directory, \
                 patch.object(sys, 'argv', ['msbuddy', *arguments, '--output', directory]), \
                 patch('msbuddy.main_cmd.MsbuddyConfig') as config, \
                 patch('msbuddy.main_cmd.Msbuddy', return_value=MagicMock()):
                main_cmd.main()
                return config.call_args.kwargs

        common = ['--data-dir', '~/msbuddy-data', '--formula-db', 'custom.joblib', '--use-da']
        legacy = get_config_args(['-mgf', 'input.mgf', '-ms', 'orbitrap', '-n_cpu', '2', *common])
        current = get_config_args(['--mgf', 'input.mgf', '--ms-instr', 'orbitrap', '--n-cpu', '2', *common])
        self.assertEqual(legacy, current)
        self.assertFalse(current['ppm'])
        self.assertEqual(current['data_dir'], '~/msbuddy-data')
        self.assertEqual(current['data_files'], {'formula_db': 'custom.joblib'})
        self.assertEqual(current['element_ranges']['C'], (0, 80))
        self.assertNotIn('Cl', current['element_ranges'])
        custom = get_config_args(['--mgf', 'input.mgf', '--halogen', '--c-max', '42', '--cl-max', '3'])
        self.assertEqual(custom['element_ranges']['C'], (0, 42))
        self.assertEqual(custom['element_ranges']['Cl'], (0, 3))

    def test_nextflow_passes_element_limits_to_the_revised_configuration(self):
        import main_nextflow

        with patch.object(sys, 'argv', ['main_nextflow', '-input', 'input.mgf', '-hal', '1',
                                        '-c_max', '42', '-cl_max', '3']), \
             patch('main_nextflow.MsbuddyConfig') as config, \
             patch('main_nextflow.Msbuddy', return_value=MagicMock()), \
             patch('os.path.exists', return_value=True):
            main_nextflow.main()
        ranges = config.call_args.kwargs['element_ranges']
        self.assertEqual(ranges['C'], (0, 42))
        self.assertEqual(ranges['Cl'], (0, 3))

    def test_csv_accepts_headered_and_headerless_usi_lists(self):
        examples = [
            ('usi,adduct\nmzspec:a,[M+H]+\nmzspec:b,[M+Na]+\n',
             ['mzspec:a', 'mzspec:b'], ['[M+H]+', '[M+Na]+']),
            ('mzspec:a,[M+H]+\nmzspec:b,[M+Na]+\n',
             ['mzspec:a', 'mzspec:b'], ['[M+H]+', '[M+Na]+']),
            ('mzspec:a\nmzspec:b\n', ['mzspec:a', 'mzspec:b'], None),
        ]
        for text, expected_usi, expected_adducts in examples:
            with self.subTest(contents=text), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'input.csv'
                path.write_text(text, encoding='utf-8')
                engine = MagicMock()
                with patch.object(sys, 'argv', ['msbuddy', '--csv', str(path)]), \
                     patch('msbuddy.main_cmd.MsbuddyConfig'), \
                     patch('msbuddy.main_cmd.Msbuddy', return_value=engine):
                    main_cmd.main()
                if expected_adducts is None:
                    engine.load_usi.assert_called_once_with(expected_usi)
                else:
                    engine.load_usi.assert_called_once_with(usi_list=expected_usi, adduct_list=expected_adducts)


if __name__ == '__main__':
    unittest.main()
