"""Measure scaling of input loading, candidate matching, and feature assembly.

Run from the repository root with ``python -m benchmarks.benchmark_workflows``.
The generated inputs are synthetic and do not download databases or models.
"""

import argparse
import tempfile
from pathlib import Path
from statistics import median
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from msbuddy.cand import _calc_subform_mass, _merge_cand_form_list
from msbuddy.load import load_mgf
from msbuddy.ml import _gen_arr_from_buddy_data, gen_ml_feature


def _positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return value


def _measure(function, repeats):
    function()  # Warm imports and Numba compilation outside the measurements.
    timings = []
    for _ in range(repeats):
        start = perf_counter()
        function()
        timings.append(perf_counter() - start)
    return median(timings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spectra', type=_positive_int, default=3000)
    parser.add_argument('--peaks', type=_positive_int, default=40,
                        help='number of MS2 peaks per spectrum')
    parser.add_argument('--candidates', type=_positive_int, default=6000)
    parser.add_argument('--mass-rows', type=_positive_int, default=500000)
    parser.add_argument('--repeats', type=_positive_int, default=3)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as directory:
        mgf_path = Path(directory) / 'synthetic.mgf'
        with mgf_path.open('w', encoding='utf-8') as mgf:
            peaks = ''.join(f'{50 + j / 10:.1f} {j + 1}\n' for j in range(args.peaks))
            for i in range(args.spectra):
                mgf.write(f'BEGIN IONS\nTITLE=spectrum_{i}\nPEPMASS=300\nCHARGE=1+\n')
                mgf.write(peaks)
                mgf.write('END IONS\n')
        seconds = _measure(lambda: load_mgf(mgf_path), args.repeats)
        print(f'MGF loading: {seconds:.4f} s ({args.spectra} spectra, {args.spectra * args.peaks} peaks)')

    formula = SimpleNamespace(array=np.array([10, 20, 0, 0, 0, 0, 0, 2, 0, 4, 0, 0], dtype=np.int16),
                              dbe=2.0, mass=100.0)
    candidates = [SimpleNamespace(formula=formula, value=i) for i in range(args.candidates)]
    features = [SimpleNamespace(candidate_formula_list=candidates)]
    seconds = _measure(lambda: _gen_arr_from_buddy_data(features), args.repeats)
    print(f'Formula array assembly: {seconds:.4f} s ({args.candidates} candidates)')

    # Isolate matrix assembly from the cost of computing the individual features.
    with patch('msbuddy.ml.gen_ml_feature_single',
               side_effect=lambda mf, cf, *unused: np.full(53, cf.value, dtype=float)):
        seconds = _measure(lambda: gen_ml_feature(features, True, 5, 10, {}), args.repeats)
    print(f'Model matrix assembly: {seconds:.4f} s ({args.candidates} candidates)')

    ms1 = list(range(args.candidates))
    ms2 = list(range(args.candidates, 2 * args.candidates))
    ms1_names = [str(i) for i in ms1]
    ms2_names = [str(i) for i in range(args.candidates // 2, args.candidates + args.candidates // 2)]
    seconds = _measure(lambda: _merge_cand_form_list(ms1, ms2, ms1_names, ms2_names), args.repeats)
    print(f'Candidate merge: {seconds:.4f} s ({args.candidates} candidates per list)')

    formulas = np.random.default_rng(8).integers(0, 50, size=(args.mass_rows, 12), dtype=np.int16)
    seconds = _measure(lambda: _calc_subform_mass(formulas, 1), args.repeats)
    print(f'Subformula masses: {seconds:.4f} s ({args.mass_rows} formulas)')


if __name__ == '__main__':
    main()
