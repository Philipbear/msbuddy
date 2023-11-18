# ==============================================================================
# Copyright (C) 2023 Shipei Xing <s1xing@health.ucsd.edu>
#
# Licensed under the Apache License 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at https://github.com/Philipbear/msbuddy/blob/main/LICENSE
# ==============================================================================
"""
File: main_cmd.py
Author: Shipei Xing
Email: s1xing@health.ucsd.edu
GitHub: Philipbear
Description: Command line interface for msbuddy.
"""

import argparse
import pathlib

import pandas as pd

from msbuddy.main import Msbuddy, MsbuddyConfig


def main():
    parser = argparse.ArgumentParser(description="msbuddy command line interface.")
    parser.add_argument('-mgf', type=str, help='Path to the MGF file.')
    parser.add_argument('-usi', type=str, help='A single USI string.')
    parser.add_argument('-csv', type=str,
                        help='Path to the CSV file containing USI strings in the first column (no header row).')
    parser.add_argument('-output', '-o', type=str, help='The output file path.')
    parser.add_argument('-details', '-d', action='store_true',
                        help='Store true. Whether to write detailed results. Default: detailed results are not written.')
    parser.add_argument('-ms_instr', '-ms', type=str, default=None,
                        help='MS instrument type. Supported types: orbitrap, qtof, fticr.')
    parser.add_argument('-disable_ppm', action='store_false',
                        help='Store false. Whether to disable ppm for mass tolerance. Default: ppm is enabled.')
    parser.add_argument('-ms1_tol', type=float, default=5, help='MS1 tolerance. Default: 5.')
    parser.add_argument('-ms2_tol', type=float, default=10, help='MS2 tolerance. Default: 10.')
    parser.add_argument('-halogen', '-hal', action='store_true',
                        help='Store true. Whether to consider halogen atoms FClBrI. Default: halogen atoms are not considered.')
    parser.add_argument('-parallel', '-p', action='store_true',
                        help='Store true. Whether to use parallel computing. Default: parallel computing is disabled.')
    parser.add_argument('-n_cpu', type=int, default=-1, help='Number of CPUs to use. Default: -1, use all CPUs.')
    parser.add_argument('-timeout_secs', '-t', type=int, default=300, help='Timeout in seconds. Default: 300.')
    parser.add_argument('-batch_size', '-bs', type=int, default=1000,
                        help='Batch size. Default: 1000. A larger batch size needs more memory, but is faster.')
    parser.add_argument('-top_n_candidate', type=int, default=500, help='Max top N candidates to keep. Default: 500.')
    parser.add_argument('-c_min', type=int, default=0, help='Minimum number of C atoms. Default: 0.')
    parser.add_argument('-c_max', type=int, default=80, help='Maximum number of C atoms. Default: 80.')
    parser.add_argument('-h_min', type=int, default=0, help='Minimum number of H atoms. Default: 0.')
    parser.add_argument('-h_max', type=int, default=150, help='Maximum number of H atoms. Default: 150.')
    parser.add_argument('-n_min', type=int, default=0, help='Minimum number of N atoms. Default: 0.')
    parser.add_argument('-n_max', type=int, default=20, help='Maximum number of N atoms. Default: 20.')
    parser.add_argument('-o_min', type=int, default=0, help='Minimum number of O atoms. Default: 0.')
    parser.add_argument('-o_max', type=int, default=30, help='Maximum number of O atoms. Default: 30.')
    parser.add_argument('-p_min', type=int, default=0, help='Minimum number of P atoms. Default: 0.')
    parser.add_argument('-p_max', type=int, default=10, help='Maximum number of P atoms. Default: 10.')
    parser.add_argument('-s_min', type=int, default=0, help='Minimum number of S atoms. Default: 0.')
    parser.add_argument('-s_max', type=int, default=15, help='Maximum number of S atoms. Default: 15.')
    parser.add_argument('-f_min', type=int, default=0, help='Minimum number of F atoms. Default: 0.')
    parser.add_argument('-f_max', type=int, default=20, help='Maximum number of F atoms. Default: 20.')
    parser.add_argument('-cl_min', type=int, default=0, help='Minimum number of Cl atoms. Default: 0.')
    parser.add_argument('-cl_max', type=int, default=15, help='Maximum number of Cl atoms. Default: 15.')
    parser.add_argument('-br_min', type=int, default=0, help='Minimum number of Br atoms. Default: 0.')
    parser.add_argument('-br_max', type=int, default=10, help='Maximum number of Br atoms. Default: 10.')
    parser.add_argument('-i_min', type=int, default=0, help='Minimum number of I atoms. Default: 0.')
    parser.add_argument('-i_max', type=int, default=10, help='Maximum number of I atoms. Default: 10.')
    parser.add_argument('-isotope_bin_mztol', type=float, default=0.02,
                        help='m/z tolerance for isotope binning, used for MS1 isotope pattern, in Dalton. Default: 0.02.')
    parser.add_argument('-max_isotope_cnt', type=int, default=4,
                        help='Maximum isotope count, used for MS1 isotope pattern. Default: 4.')
    parser.add_argument('-disable_ms2_denoise', action='store_false',
                        help='Store false. Whether to disable denoising MS2 spectrum. Default: MS2 denoise is enabled.')
    parser.add_argument('-disable_rel_int_denoise', action='store_false',
                        help='Store false. Whether to disable relative intensity for MS2 denoise. Default: relative intensity denoise is enabled.')
    parser.add_argument('-rel_int_denoise_cutoff', type=float, default=0.01,
                        help='Relative intensity cutoff, used for MS2 denoise. Default: 0.01.')
    parser.add_argument('-max_noise_frag_ratio', type=float, default=0.90,
                        help='Maximum noise fragment ratio, used for MS2 denoise. Default: 0.90.')
    parser.add_argument('-max_noise_rsd', type=float, default=0.20,
                        help='Maximum noise RSD, used for MS2 denoise. Default: 0.20.')
    parser.add_argument('-max_frag_reserved', type=int, default=50,
                        help='Max fragment number reserved, used for MS2 data.')
    parser.add_argument('-use_all_frag', action='store_true',
                        help='Store true. Whether to use all fragments for annotation; by default, only top N fragments are used. top N is a function of precursor mass. Default: Not to use all fragments.')

    args = parser.parse_args()

    # run msbuddy
    # create a MsbuddyConfig object
    msb_config = MsbuddyConfig(
        ms_instr=args.ms_instr,
        ppm=~args.disable_ppm,
        ms1_tol=args.ms1_tol, ms2_tol=args.ms2_tol, halogen=args.halogen,
        parallel=args.parallel, n_cpu=args.n_cpu,
        timeout_secs=args.timeout_secs, batch_size=args.batch_size, top_n_candidate=args.top_n_candidate,
        c_range=(args.c_min, args.c_max), h_range=(args.h_min, args.h_max), n_range=(args.n_min, args.n_max),
        o_range=(args.o_min, args.o_max), p_range=(args.p_min, args.p_max), s_range=(args.s_min, args.s_max),
        f_range=(args.f_min, args.f_max), cl_range=(args.cl_min, args.cl_max), br_range=(args.br_min, args.br_max),
        i_range=(args.i_min, args.i_max),
        isotope_bin_mztol=args.isotope_bin_mztol, max_isotope_cnt=args.max_isotope_cnt,
        ms2_denoise=~args.disable_ms2_denoise,
        rel_int_denoise=~args.disable_rel_int_denoise,
        rel_int_denoise_cutoff=args.rel_int_denoise_cutoff, max_noise_frag_ratio=args.max_noise_frag_ratio,
        max_noise_rsd=args.max_noise_rsd, max_frag_reserved=args.max_frag_reserved,
        use_all_frag=args.use_all_frag
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
