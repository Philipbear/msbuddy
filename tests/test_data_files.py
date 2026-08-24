"""Tests for the configurable data-file paths and CLI options.

    python -m unittest discover -s tests -v
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import msbuddy.load as load_module
from msbuddy import main_cmd
from msbuddy.load import DATA_FILES, get_data_files, resolve_data_dir


class TestDataFiles(unittest.TestCase):

    def test_defaults_are_unchanged(self):
        """With no data options, the packaged msbuddy/data directory is used."""
        data_dir = Path(load_module.__file__).parent / 'data'

        self.assertEqual(resolve_data_dir(), data_dir)
        for name, details in get_data_files().items():
            self.assertEqual(details['path'],
                             data_dir / DATA_FILES[name]['filename'])

    def test_custom_names_and_paths_resolve(self):
        """Relative names sit under data_dir; absolute paths are used as given."""
        absolute_model = Path('/shared/msbuddy/model-custom.joblib')
        files = get_data_files('~/msbuddy-data', {'ml_model': absolute_model})

        data_dir = Path.home() / 'msbuddy-data'
        self.assertEqual(files['formula_db']['path'],
                         data_dir / DATA_FILES['formula_db']['filename'])
        self.assertEqual(files['ml_model']['path'], absolute_model)


class TestCommandLine(unittest.TestCase):

    def _config_kwargs(self, argv):
        """Run the CLI and return the arguments it passed to MsbuddyConfig."""
        with tempfile.TemporaryDirectory() as output_dir, \
                patch.object(sys, 'argv',
                             ['msbuddy'] + argv + ['--output', output_dir]), \
                patch('msbuddy.main_cmd.MsbuddyConfig') as config, \
                patch('msbuddy.main_cmd.Msbuddy', return_value=MagicMock()):
            main_cmd.main()
        return config.call_args.kwargs

    def test_data_options_reach_the_config(self):
        kwargs = self._config_kwargs(
            ['--mgf', 'input.mgf', '--data-dir', '~/msbuddy-data',
             '--formula-db', 'formula-custom.joblib'])

        self.assertEqual(kwargs['data_dir'], '~/msbuddy-data')
        self.assertEqual(kwargs['data_files'],
                         {'formula_db': 'formula-custom.joblib'})

    def test_old_and_new_option_spellings_agree(self):
        """Existing command lines keep working alongside the new spellings."""
        legacy = self._config_kwargs(
            ['-mgf', 'input.mgf', '-ms', 'orbitrap', '-n_cpu', '2'])
        modern = self._config_kwargs(
            ['--mgf', 'input.mgf', '--ms-instr', 'orbitrap', '--n-cpu', '2'])

        self.assertEqual(legacy, modern)


if __name__ == '__main__':
    unittest.main()
