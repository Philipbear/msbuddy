import argparse
from msbuddy.buddy import Buddy, BuddyParamSet
import pandas as pd
import pathlib


def main():
    parser = argparse.ArgumentParser(description="msbuddy command line interface.")
    parser.add_argument('--mgf', type=str, help='Path to the MGF file.')
    parser.add_argument('--usi', type=str, help='A single USI string.')
    parser.add_argument('--csv', type=str, help='Path to the CSV file containing USI strings.')
    parser.add_argument('--output', type=str, help='The output file path.')
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
        ppm=args.ppm, ms1_tol=args.ms1_tol, ms2_tol=args.ms2_tol, halogen=args.halogen,
        c_range=(args.c_min, args.c_max), h_range=(args.h_min, args.h_max), n_range=(args.n_min, args.n_max),
        o_range=(args.o_min, args.o_max), p_range=(args.p_min, args.p_max), s_range=(args.s_min, args.s_max),
        f_range=(args.f_min, args.f_max), cl_range=(args.cl_min, args.cl_max), br_range=(args.br_min, args.br_max),
        i_range=(args.i_min, args.i_max),
        isotope_bin_mztol=args.isotope_bin_mztol, max_isotope_cnt=args.max_isotope_cnt,
        ms2_denoise=args.ms2_denoise, rel_int_denoise=args.rel_int_denoise,
        rel_int_denoise_cutoff=args.rel_int_denoise_cutoff, max_noise_frag_ratio=args.max_noise_frag_ratio,
        max_noise_rsd=args.max_noise_rsd, max_frag_reserved=args.max_frag_reserved,
        use_all_frag=args.use_all_frag
    )

    if not args.output:
        raise ValueError('Please specify the output file path.')

    buddy = Buddy(buddy_param_set)
    if args.mgf:
        buddy.load_mgf(args.mgf)
    elif args.usi:
        buddy.load_usi([args.usi])
    elif args.csv:
        # read and load the first column of the CSV file
        df = pd.read_csv(args.csv)
        buddy.load_usi(df.iloc[:, 0].tolist())
    else:
        raise ValueError('Please specify the input data source.')

    # formula annotation
    buddy.annotate_formula()

    # create a DataFrame object, with columns: identifier, mz, rt, formula_rank_1, estimated_fdr
    # fill in the DataFrame object one by one
    result_df = pd.DataFrame(columns=['identifier', 'mz', 'rt', 'formula_rank_1', 'estimated_fdr'])
    for mf in buddy.data:
        individual_result = mf.summarize_result()
        result_df = result_df.append({
            'identifier': mf.identifier,
            'mz': mf.mz,
            'rt': mf.rt,
            'formula_rank_1': individual_result['formula_rank_1'],
            'estimated_fdr': individual_result['estimated_fdr']
        }, ignore_index=True)

    output_path = pathlib.Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # write the DataFrame object to the output file
    result_df.to_csv(output_path / 'buddy_result_summary.csv', sep="\t", index=False)

    print('Job completed.')


if __name__ == '__main__':
    main()
