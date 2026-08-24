# ==============================================================================
# Copyright (C) 2024 Shipei Xing <philipxsp@hotmail.com>
#
# Licensed under the Apache License 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at https://github.com/Philipbear/msbuddy/blob/main/LICENSE
# ==============================================================================
"""
File: main_cmd.py
Author: Shipei Xing
Email: philipxsp@hotmail.com
GitHub: Philipbear
Description: Command line interface for msbuddy.
"""

import argparse
import pathlib

import pandas as pd

from msbuddy.main import Msbuddy, MsbuddyConfig
from msbuddy.load import download_data


def main():
    parser = argparse.ArgumentParser(description="msbuddy command line interface (version 0.3.15)")
    # Keep the original single-dash spellings for backwards compatibility.
    # Documented long options use the conventional --kebab-case form.
    parser.add_argument('--mgf', '-mgf', type=str, help='Path to the MGF file.')
    parser.add_argument('--usi', '-usi', type=str, help='A single USI string.')
    parser.add_argument('--csv', '-csv', type=str,
                        help='Path to the CSV file containing USI strings in the first column (no header row).')
    parser.add_argument('-o', '--output', '-output', type=str, help='The output file path.')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Directory for the msbuddy databases and model. '
                             'Default: the original msbuddy/data directory.')
    parser.add_argument('--common-db', type=str, default=None,
                        help='Common database filename relative to --data-dir, '
                             'or an absolute file path.')
    parser.add_argument('--formula-db', type=str, default=None,
                        help='Formula database filename relative to --data-dir, '
                             'or an absolute file path.')
    parser.add_argument('--ml-model', type=str, default=None,
                        help='ML model filename relative to --data-dir, '
                             'or an absolute file path.')
    parser.add_argument('--download-data', action='store_true',
                        help='Download the databases and model, then exit.')
    parser.add_argument('-d', '--details', '-details', action='store_true',
                        help='Store true. Whether to write detailed results. Default: detailed results are not written.')
    parser.add_argument('--ms-instr', '-ms_instr', '-ms', type=str, default=None,
                        help='MS instrument type. Supported types: orbitrap, qtof, fticr.')
    parser.add_argument('--use-da', '-use_Da', dest='use_Da', action='store_true',
                        help='Store true. Whether to use Dalton for mass tolerance. Default: ppm is used.')
    parser.add_argument('--ms1-tol', '-ms1_tol', type=float, default=5, help='MS1 tolerance. Default: 5.')
    parser.add_argument('--ms2-tol', '-ms2_tol', type=float, default=10, help='MS2 tolerance. Default: 10.')
    parser.add_argument('--halogen', '-halogen', '-hal', action='store_true',
                        help='Store true. Whether to consider halogen atoms FClBrI. Default: halogen atoms are not considered.')
    parser.add_argument('-p', '--parallel', '-parallel', action='store_true',
                        help='Store true. Whether to use parallel computing. Default: parallel computing is disabled.')
    parser.add_argument('--n-cpu', '-n_cpu', type=int, default=-1, help='Number of CPUs to use. Default: -1, use all CPUs.')
    parser.add_argument('-t', '--timeout-secs', '-timeout_secs', type=int, default=300, help='Timeout in seconds. Default: 300.')
    parser.add_argument('--batch-size', '-batch_size', '-bs', type=int, default=5000,
                        help='Batch size. Default: 5000. A larger batch size needs more memory, but is faster.')
    parser.add_argument('--c-min', '-c_min', type=int, default=0, help='Minimum number of C atoms. Default: 0.')
    parser.add_argument('--c-max', '-c_max', type=int, default=80, help='Maximum number of C atoms. Default: 80.')
    parser.add_argument('--h-min', '-h_min', type=int, default=0, help='Minimum number of H atoms. Default: 0.')
    parser.add_argument('--h-max', '-h_max', type=int, default=150, help='Maximum number of H atoms. Default: 150.')
    parser.add_argument('--n-min', '-n_min', type=int, default=0, help='Minimum number of N atoms. Default: 0.')
    parser.add_argument('--n-max', '-n_max', type=int, default=20, help='Maximum number of N atoms. Default: 20.')
    parser.add_argument('--o-min', '-o_min', type=int, default=0, help='Minimum number of O atoms. Default: 0.')
    parser.add_argument('--o-max', '-o_max', type=int, default=30, help='Maximum number of O atoms. Default: 30.')
    parser.add_argument('--p-min', '-p_min', type=int, default=0, help='Minimum number of P atoms. Default: 0.')
    parser.add_argument('--p-max', '-p_max', type=int, default=10, help='Maximum number of P atoms. Default: 10.')
    parser.add_argument('--s-min', '-s_min', type=int, default=0, help='Minimum number of S atoms. Default: 0.')
    parser.add_argument('--s-max', '-s_max', type=int, default=15, help='Maximum number of S atoms. Default: 15.')
    parser.add_argument('--f-min', '-f_min', type=int, default=0, help='Minimum number of F atoms. Default: 0.')
    parser.add_argument('--f-max', '-f_max', type=int, default=20, help='Maximum number of F atoms. Default: 20.')
    parser.add_argument('--cl-min', '-cl_min', type=int, default=0, help='Minimum number of Cl atoms. Default: 0.')
    parser.add_argument('--cl-max', '-cl_max', type=int, default=15, help='Maximum number of Cl atoms. Default: 15.')
    parser.add_argument('--br-min', '-br_min', type=int, default=0, help='Minimum number of Br atoms. Default: 0.')
    parser.add_argument('--br-max', '-br_max', type=int, default=10, help='Maximum number of Br atoms. Default: 10.')
    parser.add_argument('--i-min', '-i_min', type=int, default=0, help='Minimum number of I atoms. Default: 0.')
    parser.add_argument('--i-max', '-i_max', type=int, default=10, help='Maximum number of I atoms. Default: 10.')
    parser.add_argument('--isotope-bin-mztol', '-isotope_bin_mztol', type=float, default=0.02,
                        help='m/z tolerance for isotope binning, used for MS1 isotope pattern, in Dalton. Default: 0.02.')
    parser.add_argument('--max-isotope-cnt', '-max_isotope_cnt', type=int, default=4,
                        help='Maximum isotope count, used for MS1 isotope pattern. Default: 4.')
    parser.add_argument('--rel-int-denoise-cutoff', '-rel_int_denoise_cutoff', type=float, default=0.01,
                        help='Relative intensity cutoff, used for MS2 denoise. Default: 0.01.')
    parser.add_argument('--top-n-per-50-da', '-top_n_per_50_da', type=int, default=6,
                        help='Top n peaks per 50 Da, used for MS2 denoise. Default: 6.')

    args = parser.parse_args()

    data_files = {
        'common_db': args.common_db,
        'formula_db': args.formula_db,
        'ml_model': args.ml_model,
    }
    data_files = {
        name: path for name, path in data_files.items() if path is not None
    } or None

    if args.download_data:
        downloaded = download_data(args.data_dir, data_files)
        print('Data files are available:')
        for name, file_path in downloaded.items():
            print(f'  {name}: {file_path}')
        return

    # run msbuddy
    # create a MsbuddyConfig object
    msb_config = MsbuddyConfig(
        ms_instr=args.ms_instr,
        ppm=not args.use_Da,
        ms1_tol=args.ms1_tol, ms2_tol=args.ms2_tol, halogen=args.halogen,
        parallel=args.parallel, n_cpu=args.n_cpu,
        timeout_secs=args.timeout_secs, batch_size=args.batch_size,
        c_range=(args.c_min, args.c_max), h_range=(args.h_min, args.h_max), n_range=(args.n_min, args.n_max),
        o_range=(args.o_min, args.o_max), p_range=(args.p_min, args.p_max), s_range=(args.s_min, args.s_max),
        f_range=(args.f_min, args.f_max), cl_range=(args.cl_min, args.cl_max), br_range=(args.br_min, args.br_max),
        i_range=(args.i_min, args.i_max),
        isotope_bin_mztol=args.isotope_bin_mztol, max_isotope_cnt=args.max_isotope_cnt,
        rel_int_denoise_cutoff=args.rel_int_denoise_cutoff, top_n_per_50_da=args.top_n_per_50_da,
        data_dir=args.data_dir, data_files=data_files
    )

    if args.output:
        output_path = pathlib.Path(args.output)
    elif args.mgf or args.csv:
        # use the parent directory of the input file as the output directory
        output_path = pathlib.Path(args.mgf if args.mgf else args.csv).parent / 'msbuddy_output'
    else:
        raise ValueError('Please specify the output path.')

    engine = Msbuddy(msb_config)

    if args.mgf:
        engine.load_mgf(args.mgf)
    elif args.usi:
        engine.load_usi([args.usi])
    elif args.csv:
        # read and load CSV file, treat empty cells as empty strings
        df = pd.read_csv(args.csv, keep_default_na=False, na_values=None)
        # if df has >1 columns, treat the 2nd column as adduct strings
        if df.shape[1] > 1:
            engine.load_usi(usi_list=df.iloc[:, 0].tolist(),
                            adduct_list=df.iloc[:, 1].tolist())
        else:
            engine.load_usi(df.iloc[:, 0].tolist())
    else:
        raise ValueError('Please specify the input data source.')

    engine.annotate_formula_cmd(output_path, write_details=args.details)

    print('Job finished.')


if __name__ == '__main__':
    main()
