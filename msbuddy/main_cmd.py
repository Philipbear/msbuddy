import argparse
from msbuddy.buddy import Buddy, BuddyParamSet
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="msbuddy command line interface.")
    parser.add_argument('--mgf', type=str, help='Path to the MGF file.')
    parser.add_argument('--usi', type=str, help='A single USI string.')
    parser.add_argument('--csv', type=str, help='Path to the CSV file containing USI strings.')
    parser.add_argument('--ppm', type=bool, default=True, help='Whether to use ppm for mass tolerance.')
    parser.add_argument('--ms1_tol', type=float, default=5, help='MS1 tolerance.')
    parser.add_argument('--ms2_tol', type=float, default=10, help='MS2 tolerance.')
    parser.add_argument('--halogen', type=bool, default=False, help='Whether to consider halogen atoms.')
    parser.add_argument('--c_min', type=int, default=0, help='Minimum number of C atoms.')
    parser.add_argument('--c_max', type=int, default=80, help='Maximum number of C atoms.')
    parser.add_argument('--h_min', type=int, default=0, help='Minimum number of H atoms.')
    parser.add_argument('--h_max', type=int, default=150, help='Maximum number of H atoms.')
    parser.add_argument('--n_min', type=int, default=0, help='Minimum number of N atoms.')
    parser.add_argument('--n_max', type=int, default=20, help='Maximum number of N atoms.')
    parser.add_argument('--o_min', type=int, default=0, help='Minimum number of O atoms.')
    parser.add_argument('--o_max', type=int, default=30, help='Maximum number of O atoms.')
    parser.add_argument('--p_min', type=int, default=0, help='Minimum number of P atoms.')
    parser.add_argument('--p_max', type=int, default=10, help='Maximum number of P atoms.')
    parser.add_argument('--s_min', type=int, default=0, help='Minimum number of S atoms.')
    parser.add_argument('--s_max', type=int, default=15, help='Maximum number of S atoms.')
    parser.add_argument('--f_min', type=int, default=0, help='Minimum number of F atoms.')
    parser.add_argument('--f_max', type=int, default=20, help='Maximum number of F atoms.')
    parser.add_argument('--cl_min', type=int, default=0, help='Minimum number of Cl atoms.')
    parser.add_argument('--cl_max', type=int, default=15, help='Maximum number of Cl atoms.')
    parser.add_argument('--br_min', type=int, default=0, help='Minimum number of Br atoms.')
    parser.add_argument('--br_max', type=int, default=10, help='Maximum number of Br atoms.')
    parser.add_argument('--i_min', type=int, default=0, help='Minimum number of I atoms.')
    parser.add_argument('--i_max', type=int, default=10, help='Maximum number of I atoms.')
    parser.add_argument('--isotope_bin_mztol', type=float, default=0.02,
                        help='m/z tolerance for isotope bin, used for MS1 isotope pattern.')
    parser.add_argument('--max_isotope_cnt', type=int, default=4,
                        help='Maximum isotope count, used for MS1 isotope pattern.')
    parser.add_argument('--ms2_denoise', type=bool, default=True, help='Whether to denoise MS2 spectrum.')
    parser.add_argument('--rel_int_denoise', type=bool, default=True,
                        help='Whether to use relative intensity for MS2 denoise.')
    parser.add_argument('--rel_int_denoise_cutoff', type=float, default=0.01,
                        help='Relative intensity cutoff, used for MS2 denoise.')
    parser.add_argument('--max_noise_frag_ratio', type=float, default=0.85,
                        help='Maximum noise fragment ratio, used for MS2 denoise.')
    parser.add_argument('--max_noise_rsd', type=float, default=0.20,
                        help='Maximum noise RSD, used for MS2 denoise.')
    parser.add_argument('--max_frag_reserved', type=int, default=50,
                        help='Max fragment number reserved, used for MS2 data.')
    parser.add_argument('--use_all_frag', type=bool, default=False,
                        help='Whether to use all fragments for annotation; by default, only top N fragments are used, '
                             'top N is a function of precursor mass.')


    args = parser.parse_args()

    buddy_param_set = BuddyParamSet(
        # Configure parameters based on args
    )

    buddy = Buddy(buddy_param_set)
    if args.mgf:
        buddy.load_mgf(args.mgf)

    # ... rest of the logic ...

    print('Operation completed successfully.')