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
                            start_idx: int, end_idx: int, result_df: pd.DataFrame) -> pd.DataFrame:
    """
    write out batch results
    :param buddy_data: buddy data
    :param output_path: output path
    :param write_details: whether to write out detailed results
    :param start_idx: start index of batch
    :param end_idx: end index of batch
    :param result_df: result summary DataFrame
    :return: updated summary results DataFrame
    """
    # get batch data
    batch_data = buddy_data[start_idx:end_idx]

    # update summary results DataFrame
    for mf in batch_data:
        individual_result = mf.summarize_result()
        result_df = result_df.append({
            'identifier': mf.identifier,
            'mz': round(mf.mz, 4),
            'rt': round(mf.rt, 2) if mf.rt else 'None',
            'adduct': mf.adduct.string,
            'formula_rank_1': individual_result['formula_rank_1'],
            'estimated_fdr': round(individual_result['estimated_fdr'],
                                   4) if individual_result['estimated_fdr'] is not None else 'None',
            'formula_rank_2': individual_result['formula_rank_2'],
            'formula_rank_3': individual_result['formula_rank_3'],
            'formula_rank_4': individual_result['formula_rank_4'],
            'formula_rank_5': individual_result['formula_rank_5']
        }, ignore_index=True)

    # write out detailed results
    if write_details:
        for mf in batch_data:
            # make a directory for each mf
            # replace '/' with '_' in the identifier, remove special characters
            _id = str(mf.identifier).replace('/', '_').replace(':', '_').replace(' ', '_').strip()
            folder_name = _id + '_mz' + str(round(mf.mz, 4)) + '_rt'
            folder_name += str(round(mf.rt, 2)) if mf.rt else 'None'
            mf_path = pathlib.Path(output_path / folder_name)
            mf_path.mkdir(parents=True, exist_ok=True)

            # write the tsv file containing all the candidate formulas
            all_candidates_df = pd.DataFrame(columns=['rank', 'formula', 'formula_feasibility',
                                                      'ms1_isotope_similarity', 'explained_ms2_peak',
                                                      'total_valid_ms2_peak', 'estimated_prob',
                                                      'normalized_estimated_prob', 'estimated_fdr'])
            for m, cf in enumerate(mf.candidate_formula_list):
                # string for explained ms2 peak
                if mf.ms2_processed:
                    if cf.ms2_raw_explanation:
                        exp_ms2_peak = len(cf.ms2_raw_explanation)
                    else:
                        exp_ms2_peak = '0'
                else:
                    exp_ms2_peak = 'None'
                all_candidates_df = all_candidates_df.append({
                    'rank': m,
                    'formula': cf.formula.__str__(),
                    'formula_feasibility': round(cf.ml_a_prob, 4),
                    'ms1_isotope_similarity': round(cf.ms1_isotope_similarity,
                                                    4) if cf.ms1_isotope_similarity is not None else 'None',
                    'explained_ms2_peak': exp_ms2_peak,
                    'total_valid_ms2_peak': len(mf.ms2_processed) if mf.ms2_processed else 'None',
                    'estimated_prob': round(cf.estimated_prob,
                                            4) if cf.estimated_prob is not None else 'None',
                    'normalized_estimated_prob': round(cf.normed_estimated_prob,
                                                       4) if cf.normed_estimated_prob is not None else 'None',
                    'estimated_fdr': round(cf.estimated_fdr, 4) if cf.estimated_fdr is not None else 'None'
                }, ignore_index=True)
            all_candidates_df.to_csv(mf_path / 'formula_results.tsv', sep="\t", index=False)

            # write the tsv file containing preprocessed spectrum
            if mf.ms1_processed:
                ms1_df = pd.DataFrame(columns=['raw_idx', 'mz', 'intensity'])
                for m in range(len(mf.ms1_processed)):
                    ms1_df = ms1_df.append({
                        'raw_idx': mf.ms1_processed.idx_array[m],
                        'mz': round(mf.ms1_processed.mz_array[m], 4),
                        'intensity': round(mf.ms1_processed.int_array[m], 4)
                    }, ignore_index=True)
                ms1_df.to_csv(mf_path / 'ms1_preprocessed.tsv', sep="\t", index=False)
            if mf.ms2_processed:
                ms2_df = pd.DataFrame(columns=['raw_idx', 'mz', 'intensity'])
                for m in range(len(mf.ms2_processed)):
                    ms2_df = ms2_df.append({
                        'raw_idx': mf.ms2_processed.idx_array[m],
                        'mz': round(mf.ms2_processed.mz_array[m], 4),
                        'intensity': round(mf.ms2_processed.int_array[m], 4)
                    }, ignore_index=True)
                ms2_df.to_csv(mf_path / 'ms2_preprocessed.tsv', sep="\t", index=False)

    return result_df
