"""Regression tests for multi-batch command-line output."""

import importlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from msbuddy.base import MetaFeature, Spectrum
from msbuddy.export import write_batch_results_cmd


class BatchOutputTests(unittest.TestCase):
    def test_annotating_without_data_fails_before_loading_models(self):
        main = importlib.import_module('msbuddy.main')
        engine = object.__new__(main.Msbuddy)
        engine.config = SimpleNamespace(data_dir=None, data_files=None)
        engine._dependencies = {}
        for data in (None, []):
            with self.subTest(data=data):
                engine.data = data
                with patch.object(main, 'init_ml_models') as load_models:
                    with self.assertRaisesRegex(ValueError, 'No data loaded'):
                        engine.annotate_formula()
                    load_models.assert_not_called()

    def test_details_for_no_candidates_include_spectra_and_zero_retention_time(self):
        feature = MetaFeature('nohit', 150.0, 1, rt=0.0,
                              ms1=Spectrum([150.0, 151.003355], [100, 8]),
                              ms2=Spectrum([30, 70], [5, 8]))
        feature.data_preprocess(True, 5, 10, 0.02, 4, 0.01, 6)
        feature.candidate_formula_list = []
        with tempfile.TemporaryDirectory() as directory:
            summary = write_batch_results_cmd([feature], Path(directory), True, 0, 1)
            folders = list(Path(directory).iterdir())
            self.assertEqual(len(folders), 1)
            self.assertTrue(folders[0].name.endswith('_rt_0.0'))
            formula_file = folders[0] / 'formula_results.tsv'
            self.assertTrue(formula_file.is_file())
            self.assertEqual(pd.read_csv(formula_file, sep='\t', keep_default_na=False).iloc[0]['formula'], 'NA')
            self.assertTrue((folders[0] / 'ms1_preprocessed.tsv').is_file())
            self.assertTrue((folders[0] / 'ms2_preprocessed.tsv').is_file())
        self.assertEqual(summary.iloc[0]['rt'], 0.0)

    def test_summary_keeps_batch_order_and_clears_processed_data(self):
        main = importlib.import_module('msbuddy.main')
        engine = object.__new__(main.Msbuddy)
        engine.config = SimpleNamespace(data_dir=None, data_files=None, batch_size=2, parallel=False)
        engine._dependencies = {}
        engine.data = [SimpleNamespace(identifier=f'id{i}', mz=100 + i) for i in range(5)]

        def write_batch(data, output_path, write_details, start, end):
            return pd.DataFrame([{'identifier': item.identifier, 'mz': item.mz}
                                 for item in data[start:end]])

        def annotate_batch(batch, total):
            return batch * 2, min((batch + 1) * 2, len(engine.data))

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(main, 'init_ml_models', side_effect=lambda gd, **kwargs: gd), \
                 patch.object(main, 'write_batch_results_cmd', side_effect=write_batch), \
                 patch.object(engine, '_annotate_formula_main_batch', side_effect=annotate_batch):
                engine.annotate_formula_cmd(Path(directory))
            summary = (Path(directory) / 'msbuddy_result_summary.tsv').read_text(encoding='utf-8')

        self.assertEqual(summary, 'identifier\tmz\nid0\t100\nid1\t101\nid2\t102\nid3\t103\nid4\t104\n')
        self.assertIsNone(engine.data)
        engine.add_data(MetaFeature('next', 200, 1))
        self.assertEqual([feature.identifier for feature in engine.data], ['next'])


if __name__ == '__main__':
    unittest.main()
