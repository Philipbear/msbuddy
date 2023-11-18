# ==============================================================================
# Copyright (C) 2023 Shipei Xing <s1xing@health.ucsd.edu>
#
# Licensed under the Apache License 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at https://github.com/Philipbear/msbuddy/blob/main/LICENSE
# ==============================================================================
"""
File: export.py
Author: Shipei Xing
Email: s1xing@health.ucsd.edu
GitHub: Philipbear
Description: export results to files
"""

import pathlib

import pandas as pd


def write_batch_results_cmd(buddy_data, output_path: pathlib.Path, write_details: bool,
                            start_idx: int, end_idx: int) -> pd.DataFrame:
    """
    write out batch results
    :param buddy_data: buddy data
    :param output_path: output path
    :param write_details: whether to write out detailed results
    :param start_idx: start index of batch
    :param end_idx: end index of batch
    :return: updated summary results DataFrame
    """
    # get batch data
    batch_data = buddy_data[start_idx:end_idx]
    result_df_rows = []

    # update summary results DataFrame
    for mf in batch_data:
        individual_result = mf.summarize_result()
        result_df_rows.append({
            'identifier': mf.identifier,
            'mz': round(mf.mz, 4),
            'rt': round(mf.rt, 4) if mf.rt else 'NA',
            'adduct': mf.adduct.string,
            'formula_rank_1': individual_result['formula_rank_1'],
            'estimated_fdr': individual_result['estimated_fdr'] if individual_result[
                                                                       'estimated_fdr'] is not None else 'NA',
            'formula_rank_2': individual_result['formula_rank_2'],
            'formula_rank_3': individual_result['formula_rank_3'],
            'formula_rank_4': individual_result['formula_rank_4'],
            'formula_rank_5': individual_result['formula_rank_5']
        })
    result_df = pd.DataFrame(result_df_rows)

    # write out detailed results
    if write_details:
        for mf in batch_data:
            # make a directory for each mf
            # replace '/' with '_' in the identifier, remove special characters
            _id = str(mf.identifier).replace('/', '').replace(':', '').replace(' ', '').strip()
            folder_name = _id + '_mz_' + str(round(mf.mz, 4)) + '_rt_'
            folder_name += str(round(mf.rt, 2)) if mf.rt else 'NA'
            mf_path = pathlib.Path(output_path / folder_name)
            mf_path.mkdir(parents=True, exist_ok=True)

            # write the tsv file containing all the candidate formulas
            all_candidates_df_rows = []
            for m, cf in enumerate(mf.candidate_formula_list):
                # strings for explained ms2 peak
                if mf.ms2_processed:
                    if cf.ms2_raw_explanation:
                        exp_ms2_peak = len(cf.ms2_raw_explanation)
                        ms2_explan_idx = ','.join([str(x) for x in cf.ms2_raw_explanation.idx_array])
                        ms2_explan_str = ','.join([x.__str__() for x in cf.ms2_raw_explanation.explanation_array])
                    else:
                        exp_ms2_peak = '0'
                        ms2_explan_idx = 'None'
                        ms2_explan_str = 'None'
                else:
                    exp_ms2_peak = 'NA'
                    ms2_explan_idx = 'NA'
                    ms2_explan_str = 'NA'
                # theoretical mass
                theo_mass = (cf.formula.mass * mf.adduct.m + mf.adduct.net_formula.mass -
                             mf.adduct.charge * 0.0005486) / abs(mf.adduct.charge)
                mz_error_ppm = (mf.mz - theo_mass) / theo_mass * 1e6
                all_candidates_df_rows.append({
                    'rank': str(m + 1),
                    'formula': cf.formula.__str__(),
                    'formula_feasibility': cf.ml_a_prob if cf.ml_a_prob is not None else 'NA',
                    'ms1_isotope_similarity': round(cf.ms1_isotope_similarity,
                                                    5) if cf.ms1_isotope_similarity is not None else 'NA',
                    'mz_error_ppm': round(mz_error_ppm, 5),
                    'explained_ms2_peak': exp_ms2_peak,
                    'total_valid_ms2_peak': len(mf.ms2_processed) if mf.ms2_processed else 'NA',
                    'estimated_prob': cf.estimated_prob if cf.estimated_prob is not None else 'NA',
                    'normalized_estimated_prob': cf.normed_estimated_prob if cf.normed_estimated_prob is not None else 'NA',
                    'estimated_fdr': cf.estimated_fdr if cf.estimated_fdr is not None else 'NA',
                    'ms2_explanation_idx': ms2_explan_idx,
                    'ms2_explanation': ms2_explan_str
                })

            all_candidates_df = pd.DataFrame(all_candidates_df_rows)
            all_candidates_df.to_csv(mf_path / 'formula_results.tsv', sep="\t", index=False)

            # write the tsv file containing preprocessed spectrum
            if mf.ms1_processed:
                ms1_df_rows = []
                for m in range(len(mf.ms1_processed)):
                    ms1_df_rows.append({
                        'raw_idx': mf.ms1_processed.idx_array[m],
                        'mz': mf.ms1_processed.mz_array[m],
                        'intensity': mf.ms1_processed.int_array[m]
                    })
                ms1_df = pd.DataFrame(ms1_df_rows)
                ms1_df.to_csv(mf_path / 'ms1_preprocessed.tsv', sep="\t", index=False)
            if mf.ms2_processed:
                ms2_df_rows = []
                for m in range(len(mf.ms2_processed)):
                    ms2_df_rows.append({
                        'raw_idx': mf.ms2_processed.idx_array[m],
                        'mz': mf.ms2_processed.mz_array[m],
                        'intensity': mf.ms2_processed.int_array[m]
                    })
                ms2_df = pd.DataFrame(ms2_df_rows)
                ms2_df.to_csv(mf_path / 'ms2_preprocessed.tsv', sep="\t", index=False)

    return result_df


def round_to_sci(number, decimals):
    """
    Round a number to a given number of decimals in scientific notation.
    """
    # Convert the number to scientific notation with specified decimals
    sci_number = "{:e}".format(number)
    base, exponent = sci_number.split("e")

    # Round the base to the specified number of decimals
    rounded_base = round(float(base), decimals)

    # Construct the rounded scientific notation
    rounded_sci_number = f"{rounded_base}e{exponent}"

    return rounded_sci_number
