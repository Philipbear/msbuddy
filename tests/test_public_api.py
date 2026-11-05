"""Tests for the lightweight package entry point."""

import subprocess
import sys
import unittest
from pathlib import Path


class PublicApiTests(unittest.TestCase):
    def test_cli_version_matches_python_package(self):
        import msbuddy

        root = Path(__file__).resolve().parents[1]
        result = subprocess.run([sys.executable, '-m', 'msbuddy.main_cmd', '--version'],
                                cwd=root, check=True, text=True, capture_output=True)
        self.assertEqual(result.stdout.strip(), f'msbuddy {msbuddy.__version__}')

    def test_import_does_not_disable_caller_warnings(self):
        code = '''\
import warnings
warnings.filterwarnings("error", message="caller warning marker", category=UserWarning)
import msbuddy.ml
try:
    warnings.warn("caller warning marker", UserWarning)
except UserWarning:
    pass
else:
    raise AssertionError("importing msbuddy changed the caller's warning policy")
'''
        root = Path(__file__).resolve().parents[1]
        subprocess.run([sys.executable, '-c', code], cwd=root, check=True)

    def test_import_is_lazy_and_public_names_remain_available(self):
        code = '''\
import sys
heavy = {"numpy", "scipy", "pandas", "numba"}
already_loaded = heavy.intersection(sys.modules)
import msbuddy
assert heavy.intersection(sys.modules) == already_loaded
assert msbuddy.__version__
assert "Msbuddy" in dir(msbuddy)
from msbuddy import Msbuddy, Formula, read_formula
from msbuddy.main import Msbuddy as DirectMsbuddy
from msbuddy.base import Formula as DirectFormula
from msbuddy.utils import read_formula as direct_read_formula
assert Msbuddy is DirectMsbuddy
assert Formula is DirectFormula
assert read_formula is direct_read_formula
namespace = {}
exec("from msbuddy import *", namespace)
assert all(name in namespace for name in msbuddy.__all__)
assert msbuddy.main.Msbuddy is Msbuddy
try:
    msbuddy.missing_api_name
except AttributeError:
    pass
else:
    raise AssertionError("missing names must raise AttributeError")
'''
        root = Path(__file__).resolve().parents[1]
        subprocess.run([sys.executable, '-c', code], cwd=root, check=True)


if __name__ == '__main__':
    unittest.main()
