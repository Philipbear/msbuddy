# ==============================================================================
# Copyright (C) 2024 Shipei Xing <philipxsp@hotmail.com>
#
# Licensed under the Apache License 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at https://github.com/Philipbear/msbuddy/blob/main/LICENSE
# ==============================================================================
"""
File: ml.py
Author: Shipei Xing
Email: philipxsp@hotmail.com
GitHub: Philipbear
Description: machine learning functions: feature generation, prediction, etc.
             False discovery rate (FDR) estimation
"""
import math
import sys
import warnings

import numpy as np
from numba import njit
from scipy.stats import norm
from tqdm import tqdm

from msbuddy.query import common_nl_from_array, check_formula_existence

# ignore warnings
warnings.filterwarnings('ignore')


def _gen_arr_from_buddy_data(buddy_data) -> (np.array, np.array, np.array):
    """
    generate three arrays for ML
    :param buddy_data: List of MetaFeature objects
    :return: all_cand_form_arr, dbe_arr, mass_arr
    """
    all_cand_form_arr = np.empty((0, 12))  # formula array of all candidate formulas
    dbe_arr = np.array([])
    mass_arr = np.array([])

    for mf in buddy_data:
        if not mf.candidate_formula_list:
            continue
        # generate ML features for each candidate formula
        for cf in mf.candidate_formula_list:
            # add formula array to all_cand_form_arr
            all_cand_form_arr = np.append(all_cand_form_arr, [cf.formula.array], axis=0)
            dbe_arr = np.append(dbe_arr, cf.formula.dbe)
            mass_arr = np.append(mass_arr, cf.formula.mass)

    return all_cand_form_arr, dbe_arr, mass_arr


@njit
def _gen_form_feature(all_cf_arr, dbe_arr, mass_arr) -> np.array:
    """
    generate formula features for ML
    :param all_cand_form_arr: numpy array of all candidate formula arrays
    :param dbe_arr: numpy array of all candidate formula dbe
    :param mass_arr: numpy array of all candidate formula mass
    :return: numpy array of ML features
    """
    # calculate ML features
    ele_sum_1_arr = all_cf_arr[:, 2] + all_cf_arr[:, 3] + all_cf_arr[:, 4] + all_cf_arr[:, 5] + \
                    all_cf_arr[:, 6] + all_cf_arr[:, 8]
    ele_sum_2_arr = ele_sum_1_arr + all_cf_arr[:, 10] + all_cf_arr[:, 11]
    chon_arr = np.clip(ele_sum_2_arr, 0, 1)  # whether other eles other than C, H, O, N exist
    chonps_arr = np.clip(ele_sum_1_arr, 0, 1)  # whether other eles other than C, H, O, N, P, S exist

    hal_arr = np.sum(all_cf_arr[:, 2:6], axis=1)  # sum of halogen atoms
    ta_arr = np.sum(all_cf_arr, axis=1)  # total number of atoms
    f_exist_arr = np.clip(all_cf_arr[:, 4], 0, 1)  # whether F exists
    cl_exist_arr = np.clip(all_cf_arr[:, 3], 0, 1)  # whether Cl exists
    br_exist_arr = np.clip(all_cf_arr[:, 2], 0, 1)  # whether Br exists
    i_exist_arr = np.clip(all_cf_arr[:, 5], 0, 1)  # whether I exists
    hal_ele_type_arr = f_exist_arr + cl_exist_arr + br_exist_arr + i_exist_arr  # number of halogen elements
    hal_two = np.clip(hal_ele_type_arr - 1, 0, 1)  # whether more than one halogen element exists
    hal_three = np.clip(hal_ele_type_arr - 2, 0, 1)  # whether more than two halogen elements exist
    senior_1_1_arr = (6 * all_cf_arr[:, 11] + 5 * all_cf_arr[:, 10] + 4 * all_cf_arr[:, 0] + 3 * all_cf_arr[:, 7] +
                      2 * all_cf_arr[:, 9] + all_cf_arr[:, 1] + hal_arr)
    senior_1_2_arr = all_cf_arr[:, 7] + all_cf_arr[:, 10] + all_cf_arr[:, 1] + hal_arr

    # halogen to H ratio, fill 0 if H = 0
    hal_h_arr = np.zeros(len(hal_arr))
    # if H > 0
    h_bool_arr = all_cf_arr[:, 1] > 0
    hal_h_arr[h_bool_arr] = hal_arr[h_bool_arr] / all_cf_arr[h_bool_arr, 1]

    # O/P ratio, fill 1 if phosphorus = 0
    o_p_arr = np.ones(len(hal_arr))
    # if phosphorus > 0
    p_bool_arr = all_cf_arr[:, 10] > 0
    o_p_arr[p_bool_arr] = all_cf_arr[p_bool_arr, 9] / all_cf_arr[p_bool_arr, 10] / 3

    # generate output array
    out = np.empty((len(all_cf_arr), 26))
    # populate output array
    for i in range(len(all_cf_arr)):
        ta = ta_arr[i]
        chno = all_cf_arr[i, 0] + all_cf_arr[i, 1] + all_cf_arr[i, 7] + all_cf_arr[i, 9]
        # if C > 0
        if all_cf_arr[i, 0] > 0:
            out[i, :] = [1 - chon_arr[i], 1 - chonps_arr[i],
                         all_cf_arr[i, 0] / ta, all_cf_arr[i, 1] / ta,
                         all_cf_arr[i, 7] / ta,
                         all_cf_arr[i, 9] / ta, all_cf_arr[i, 10] / ta,
                         all_cf_arr[i, 11] / ta, chno / ta,
                         hal_arr[i] / ta, senior_1_1_arr[i], senior_1_2_arr[i], 2 * ta - 1, dbe_arr[i],
                         np.sqrt(dbe_arr[i] / mass_arr[i]), dbe_arr[i] / np.power(mass_arr[i] / 100, 2 / 3),
                         all_cf_arr[i, 1] / all_cf_arr[i, 0],
                         all_cf_arr[i, 7] / all_cf_arr[i, 0],
                         all_cf_arr[i, 9] / all_cf_arr[i, 0],
                         all_cf_arr[i, 10] / all_cf_arr[i, 0],
                         all_cf_arr[i, 11] / all_cf_arr[i, 0],
                         hal_arr[i] / all_cf_arr[i, 0],
                         hal_h_arr[i], o_p_arr[i], hal_two[i], hal_three[i]]
        else:
            out[i, :] = [1 - chon_arr[i], 1 - chonps_arr[i],
                         all_cf_arr[i, 0] / ta, all_cf_arr[i, 1] / ta,
                         all_cf_arr[i, 7] / ta,
                         all_cf_arr[i, 9] / ta, all_cf_arr[i, 10] / ta,
                         all_cf_arr[i, 11] / ta, chno / ta,
                         hal_arr[i] / ta, senior_1_1_arr[i], senior_1_2_arr[i], 2 * ta - 1, dbe_arr[i],
                         np.sqrt(dbe_arr[i] / mass_arr[i]), dbe_arr[i] / np.power(mass_arr[i] / 100, 2 / 3),
                         0, 0, 0, 0, 0, 0,
                         hal_h_arr[i], o_p_arr[i], hal_two[i], hal_three[i]]

    return out


def _fill_form_feature_arr_in_batch_data(batch_data, feature_arr) -> None:
    """
    fill ML features in batch data
    :param batch_data: List of MetaFeature objects
    :param feature_arr: 2D numpy array of ML features
    :return: None
    """
    cnt = 0
    for mf in batch_data:
        if not mf.candidate_formula_list:
            continue
        # fill in ML features for each candidate formula
        for cf in mf.candidate_formula_list:
            cf.formula_feature_array = feature_arr[cnt, :]
            cnt += 1
    return


def gen_ml_feature(meta_feature_list, ppm: bool, ms1_tol: float, ms2_tol: float, gd) -> np.array:
    """
    generate ML features for all metabolic features
    :param meta_feature_list: List of MetaFeature objects
    :param ppm: whether to use ppm error
    :param ms1_tol: m/z tolerance for MS1
    :param ms2_tol: m/z tolerance for MS2
    :param gd: global dependencies
    :return: numpy array of ML features
    """
    # generate feature array and group size array
    total_X_arr = np.array([])

    for mf in meta_feature_list:
        if not mf.candidate_formula_list:
            continue

        # generate ML features for each candidate formula
        for cf in mf.candidate_formula_list:
            # get ML features
            ml_feature_arr = gen_ml_feature_single(mf, cf, ppm, ms1_tol, ms2_tol, gd)

            # add to feature array
            if total_X_arr.size == 0:
                total_X_arr = ml_feature_arr
            else:
                total_X_arr = np.vstack((total_X_arr, ml_feature_arr))

    # ensure 2d array
    if len(total_X_arr.shape) == 1:
        total_X_arr = np.expand_dims(total_X_arr, axis=0)

    return total_X_arr


def gen_ml_feature_single(meta_feature, cand_form, ppm: bool, ms1_tol: float, ms2_tol: float, gd) -> np.array:
    """
    generate ML features for a single candidate formula
    :param meta_feature: MetaFeature object
    :param cand_form: CandidateFormula object
    :param ppm: whether to use ppm error
    :param ms1_tol: m/z tolerance for MS1
    :param ms2_tol: m/z tolerance for MS2
    :param gd: global dependencies
    :return: numpy array of ML features
    """

    this_adduct = meta_feature.adduct

    # mz error in ppm
    theo_mass = cand_form.charged_formula.mass / abs(this_adduct.charge)
    mz_error = (meta_feature.mz - theo_mass) / theo_mass * 1e6 if ppm else meta_feature.mz - theo_mass
    mz_error_log_p = _calc_log_p_norm(mz_error, ms1_tol / 3)

    # precursor charged formula
    pre_charged_arr = cand_form.charged_formula.array
    pre_dbe = cand_form.charged_formula.dbe
    pre_h2c = pre_charged_arr[1] / pre_charged_arr[0] if pre_charged_arr[0] > 0 else 0  # no carbon, assign 0

    # MS1 isotope similarity
    ms1_iso_sim = cand_form.ms1_isotope_similarity if cand_form.ms1_isotope_similarity else 0

    # MS/MS-related features
    ms2_feature_arr = _gen_ms2_feature(meta_feature, cand_form, pre_dbe, pre_h2c, ppm, ms2_tol, gd)

    # pos mode bool
    pos_mode = 1 if this_adduct.charge > 0 else 0

    # generate output array
    out = np.concatenate((np.array([ms1_iso_sim]), np.array([mz_error_log_p]), np.array([pos_mode]),  # 3
                          cand_form.formula_feature_array, ms2_feature_arr))  # 26 + 24

    return out


def _gen_ms2_feature(meta_feature, cand_form, pre_dbe: float, pre_h2c: float,
                     ppm: bool, ms2_tol: float, gd) -> np.array:
    """
    generate MS/MS-related features for a single candidate formula
    :param meta_feature: MetaFeature object
    :param cand_form: CandidateFormula object
    :param pre_dbe: precursor DBE
    :param pre_h2c: precursor H/C ratio
    :param ppm: whether to use ppm error
    :param ms2_tol: m/z tolerance for MS2
    :param gd: global dependencies
    :return: numpy array of MS/MS-related features
    """
    # MS2 explanation
    ms2_explanation = cand_form.ms2_raw_explanation

    # valid MS2 explanation
    if meta_feature.ms2_processed and ms2_explanation:
        # explained fragment ion
        exp_idx_arr = ms2_explanation.idx_array
        exp_int_arr = meta_feature.ms2_raw.int_array[exp_idx_arr]
        exp_mz_arr = meta_feature.ms2_raw.mz_array[exp_idx_arr]
        # valid fragment ion after MS2 processing
        valid_idx_arr = meta_feature.ms2_processed.idx_array
        valid_int_arr = meta_feature.ms2_raw.int_array[valid_idx_arr]

        # explained fragment ion count percentage
        exp_frag_cnt_pct = len(exp_idx_arr) / len(valid_idx_arr)

        # explained fragment ion intensity percentage
        exp_frag_int_pct = np.sum(exp_int_arr) / np.sum(valid_int_arr)

        frag_form_list = ms2_explanation.explanation_list  # list of fragment formulas, Formula objects

        pos_mode = meta_feature.adduct.pos_mode

        # check db existence of all explained fragments and neutral losses
        frag_db_existed, frag_common = np.array([], dtype=bool), np.array([], dtype=bool)
        nl_db_existed, nl_common = np.array([], dtype=bool), np.array([], dtype=bool)
        for f in frag_form_list:
            frag_db_bool, frag_common_bool = check_formula_existence(f.array, pos_mode, True, gd)
            frag_db_existed = np.append(frag_db_existed, frag_db_bool)
            frag_common = np.append(frag_common, frag_common_bool)
            nl_db_bool, nl_common_bool = check_formula_existence(cand_form.charged_formula.array - f.array,
                                                                 pos_mode, False, gd)
            nl_db_existed = np.append(nl_db_existed, nl_db_bool)
            nl_common = np.append(nl_common, nl_common_bool)

        # logical OR of fragment/nl
        fragnl_db_existed = np.logical_or(frag_db_existed, nl_db_existed)
        fragnl_common = np.logical_or(frag_common, nl_common)

        # explained and db existed fragment/nl ion count percentage
        exp_db_frag_cnt_pct = np.sum(frag_db_existed) / len(valid_idx_arr)
        exp_db_nl_cnt_pct = np.sum(nl_db_existed) / len(valid_idx_arr)
        exp_db_fragnl_cnt_pct = np.sum(fragnl_db_existed) / len(valid_idx_arr)

        # explained and db existed fragment/nl ion intensity percentage
        exp_db_frag_int_pct = np.sum(exp_int_arr[frag_db_existed]) / np.sum(valid_int_arr)
        exp_db_nl_int_pct = np.sum(exp_int_arr[nl_db_existed]) / np.sum(valid_int_arr)
        exp_db_fragnl_int_pct = np.sum(exp_int_arr[fragnl_db_existed]) / np.sum(valid_int_arr)

        # common fragment/nl ion count percentage
        exp_common_frag_cnt_pct = np.sum(frag_common) / len(valid_idx_arr)
        exp_common_nl_cnt_pct = np.sum(nl_common) / len(valid_idx_arr)
        exp_common_fragnl_cnt_pct = np.sum(fragnl_common) / len(valid_idx_arr)

        # common fragment/nl ion intensity percentage
        exp_common_frag_int_pct = np.sum(exp_int_arr[frag_common]) / np.sum(valid_int_arr)
        exp_common_nl_int_pct = np.sum(exp_int_arr[nl_common]) / np.sum(valid_int_arr)
        exp_common_fragnl_int_pct = np.sum(exp_int_arr[fragnl_common]) / np.sum(valid_int_arr)

        # subformula count: how many frags are subformula of other frags
        subform_score, subform_common_loss_score = _calc_subformula_score(frag_form_list, gd)

        # radical ion count percentage (out of all explained fragment ions)
        radical_cnt_pct = np.sum([1 for frag_form in frag_form_list if frag_form.dbe % 1 == 0]) / len(frag_form_list)

        # normalized explained intensity array
        normed_exp_int_arr = exp_int_arr / np.sum(exp_int_arr)

        # weighted average of fragment DBEs
        frag_dbe_wavg = np.sum(np.array([frag_form.dbe for frag_form in frag_form_list]) * normed_exp_int_arr)

        # weighted average of fragment H/C ratios
        frag_h2c_wavg = np.sum(np.array([frag_form.array[1] / frag_form.array[0] if frag_form.array[0] > 0 else pre_h2c
                                         for frag_form in frag_form_list]) * normed_exp_int_arr)

        # weighted average of fragment m/z ppm errors
        if ppm:
            frag_mz_err_wavg = np.sum(np.array([_calc_log_p_norm((frag_form.mass - mz) / frag_form.mass * 1e6,
                                                                 ms2_tol / 3)
                                                for frag_form, mz in
                                                zip(frag_form_list, exp_mz_arr)]) * normed_exp_int_arr)
        else:
            frag_mz_err_wavg = np.sum(np.array([_calc_log_p_norm(frag_form.mass - mz, ms2_tol / 3)
                                                for frag_form, mz in
                                                zip(frag_form_list, exp_mz_arr)]) * normed_exp_int_arr)

        # weighted average of fragment-nl DBE difference
        frag_nl_dbe_diff_wavg = np.sum(np.array([frag_form.dbe - (pre_dbe - frag_form.dbe + 1)
                                                 for frag_form in frag_form_list]) * normed_exp_int_arr)

        out_arr = np.array([exp_frag_cnt_pct, exp_frag_int_pct, exp_db_frag_cnt_pct, exp_db_nl_cnt_pct,
                            exp_db_fragnl_cnt_pct, exp_db_frag_int_pct, exp_db_nl_int_pct, exp_db_fragnl_int_pct,
                            exp_common_frag_cnt_pct, exp_common_nl_cnt_pct, exp_common_fragnl_cnt_pct,
                            exp_common_frag_int_pct, exp_common_nl_int_pct, exp_common_fragnl_int_pct,
                            subform_score, subform_common_loss_score,
                            radical_cnt_pct, frag_dbe_wavg, frag_h2c_wavg, frag_mz_err_wavg, frag_nl_dbe_diff_wavg,
                            len(valid_idx_arr), math.sqrt(exp_frag_cnt_pct), math.sqrt(exp_frag_int_pct)])
    else:
        out_arr = np.array([0] * 24)

    return out_arr


def _calc_subformula_score(frag_form_arr, gd) -> (float, float):
    """
    calculate how many formulas are subformula of other formulas, generate corresponding scores
    :param frag_form_arr: list of Formula objects
    :return: subformula count, subformula count with common loss
    """
    # 0 or 1 frag explanation
    if len(frag_form_arr) <= 1:
        return 0, 0

    exp_frag_cnt = 0  # explained fragment count, except for isotope peaks
    # loop through all fragment formulas, stack formula arrays that are not isotope peaks
    all_frag_arr = frag_form_arr[0].array
    for i in range(1, len(frag_form_arr)):
        if frag_form_arr[i].isotope > 0:
            continue
        all_frag_arr = np.vstack((all_frag_arr, frag_form_arr[i].array))
        exp_frag_cnt += 1

    if exp_frag_cnt <= 1:
        return 0, 0

    # subformula check & subformula common loss check
    subform_cnt, subform_common_loss_cnt = _subformula_check(all_frag_arr, gd['common_loss_db'])

    # generate scores, normalized by the number of all possible combinations
    subform_score = 2 * subform_cnt / (exp_frag_cnt * (exp_frag_cnt - 1))
    subform_common_loss_score = 2 * subform_common_loss_cnt / (exp_frag_cnt * (exp_frag_cnt - 1))

    return subform_score, subform_common_loss_score


@njit
def _subformula_check(all_frag_arr: np.array, nl_db: np.array):
    """
    check if a formula is subformula of another formula
    :param all_frag_arr: array of fragment formulas, array of formula arrays
    :param nl_db: common loss database
    :return: subformula count, subformula count with common loss
    """
    subform_cnt = 0
    subform_common_loss_cnt = 0

    for i in range(len(all_frag_arr)):
        for j in range(i + 1, len(all_frag_arr)):
            delta_arr = all_frag_arr[j] - all_frag_arr[i]
            # check if subformula; check only C > 0, N > 0
            if np.all(delta_arr >= 0) and np.sum(delta_arr[1:]) > 0 and (np.sum(delta_arr) - delta_arr[7]) > 0:
                subform_cnt += 1
                # check if there is common loss
                if common_nl_from_array(delta_arr, nl_db):
                    subform_common_loss_cnt += 1

    return subform_cnt, subform_common_loss_cnt


def _calc_log_p_norm(arr: np.array, sigma: float) -> np.array:
    """
    calculate log(p) for each element in an array, where p is the probability of a normal distribution
    :param arr: numpy array
    :param sigma: sigma for normal distribution
    :return: numpy array
    """
    arr_norm_p = norm.cdf(arr, loc=0, scale=sigma)
    return _calc_log_p_norm_helper(arr_norm_p)


@njit
def _calc_log_p_norm_helper(arr_norm_p) -> np.array:
    """
    calculate log(p) for a single element, where p is the probability of a normal distribution
    :param arr_norm_p: array of probabilities
    :return: numpy array of log(p)
    """
    arr_norm_p = 1 - arr_norm_p if arr_norm_p > 0.5 else arr_norm_p
    log_p = np.log(arr_norm_p * 2)
    # clip to the range of [-2, 0]
    log_p = -2 if log_p < -2 else log_p

    return log_p


def _predict_ml(meta_feature_list, group_no: int, ppm: bool, ms1_tol: float, ms2_tol: float, gd) -> np.array:
    """
    ml prediction
    :param meta_feature_list: List of MetaFeature objects
    :param group_no: group number; 0: ms1 ms2; 1: ms1 noms2; 2: noms1 ms2; 3: noms1 noms2
    :param ppm: whether to use ppm error
    :param ms1_tol: m/z tolerance for MS1
    :param ms2_tol: m/z tolerance for MS2
    :param gd: global dependencies
    :return: numpy array of prediction results
    """
    # generate feature array
    X_arr = gen_ml_feature(meta_feature_list, ppm, ms1_tol, ms2_tol, gd)

    if X_arr.size == 0:
        return np.array([])

    # load model
    if group_no == 0:
        model = gd['model_ms1_ms2']
    elif group_no == 1:
        model = gd['model_ms1_noms2']
        X_arr = X_arr[:, :-24]  # remove MS2-related features
    elif group_no == 2:
        model = gd['model_noms1_ms2']
        X_arr = X_arr[:, 1:]  # remove MS1 isotope similarity
    else:
        model = gd['model_noms1_noms2']
        X_arr = X_arr[:, 1:]  # remove MS1 isotope similarity
        X_arr = X_arr[:, :-24]  # remove MS2-related features

    # predict formula probability
    score_arr = model.predict(X_arr)
    return score_arr


def predict_formula_probability(buddy_data, batch_start_idx: int, batch_end_idx: int, config, gd):
    """
    predict formula probability
    :param buddy_data: buddy data
    :param batch_start_idx: batch start index
    :param batch_end_idx: batch end index
    :param config: config object
    :param gd: global dependencies
    :return: fill in estimated_prob in candidate formula objects
    """
    ppm = config.ppm
    ms1_tol = config.ms1_tol
    ms2_tol = config.ms2_tol

    # batch data
    batch_data = buddy_data[batch_start_idx:batch_end_idx]

    # generate three arrays from buddy data
    cand_form_arr, dbe_arr, mass_arr = _gen_arr_from_buddy_data(batch_data)

    # if no candidate formula, return
    if cand_form_arr.size == 0:
        return

    # generate ML feature array
    feature_arr = _gen_form_feature(cand_form_arr, dbe_arr, mass_arr)

    # fill in batch_data
    _fill_form_feature_arr_in_batch_data(batch_data, feature_arr)

    # split buddy data into 4 groups and store their indices in a dictionary
    group_dict = dict()
    for i in range(4):
        group_dict[i] = []

    for i, meta_feature in enumerate(batch_data):
        if not meta_feature.candidate_formula_list:
            continue
        if meta_feature.ms1_raw:
            if meta_feature.ms2_raw:
                group_dict[0].append(i)
            else:
                group_dict[1].append(i)
        else:
            if meta_feature.ms2_raw:
                group_dict[2].append(i)
            else:
                group_dict[3].append(i)

    # predict formula probability
    for i in range(4):
        if not group_dict[i]:
            continue
        # predict formula probability
        prob_arr = _predict_ml([batch_data[j] for j in group_dict[i]], i, ppm, ms1_tol, ms2_tol, gd)
        # Platt calibration
        prob_arr = _platt_calibrated_probability(prob_arr, gd['platt_a_' + str(i)], gd['platt_b_' + str(i)])
        # add prediction results to candidate formula objects in the list
        cnt = 0
        for j in group_dict[i]:
            for candidate_formula in batch_data[j].candidate_formula_list:
                candidate_formula.estimated_prob = prob_arr[cnt]
                cnt += 1

    # update buddy data
    buddy_data[batch_start_idx:batch_end_idx] = batch_data
    return


def _platt_calibrated_probability(score, a, b):
    """
    Platt calibration for FDR estimation
    :param score: predicted scores
    :param a: coefficient of the sigmoid function
    :param b: intercept of the sigmoid function
    :return: calibrated probability
    """
    probability = 1 / (1 + np.exp(-(a * score + b)))
    return probability


def calc_fdr(buddy_data, batch_start_idx: int, batch_end_idx: int):
    """
    calculate FDR for candidate formulas
    :param buddy_data: buddy data
    :param batch_start_idx: batch start index
    :param batch_end_idx: batch end index
    :return: fill in FDR in candidate formula objects
    """
    batch_data = buddy_data[batch_start_idx:batch_end_idx]

    # sort candidate formula list for each metabolic feature
    for meta_feature in tqdm(batch_data, desc="FDR calculation: ", file=sys.stdout, colour="green"):
        if not meta_feature.candidate_formula_list:
            continue
        # sort candidate formula list by estimated probability, in descending order
        meta_feature.candidate_formula_list.sort(key=lambda x: x.estimated_prob, reverse=True)

        # sum of estimated probabilities
        prob_sum = np.sum([cand_form.estimated_prob for cand_form in meta_feature.candidate_formula_list])

        # calculate normed estimated prob and FDR considering all candidate formulas
        sum_normed_estimated_prob = 0
        for i, cand_form in enumerate(meta_feature.candidate_formula_list):
            this_normed_estimated_prob = cand_form.estimated_prob / prob_sum
            sum_normed_estimated_prob += this_normed_estimated_prob

            cand_form.normed_estimated_prob = this_normed_estimated_prob
            cand_form.estimated_fdr = 1 - (sum_normed_estimated_prob / (i + 1))

        # if meta_feature.candidate_formula_list[0].estimated_prob > 0.1:
        #     # calculate normed estimated prob and FDR considering all candidate formulas
        #     sum_normed_estimated_prob = 0
        #     for i, cand_form in enumerate(meta_feature.candidate_formula_list):
        #         this_normed_estimated_prob = cand_form.estimated_prob / prob_sum
        #         sum_normed_estimated_prob += this_normed_estimated_prob
        #
        #         cand_form.normed_estimated_prob = this_normed_estimated_prob
        #         cand_form.estimated_fdr = 1 - (sum_normed_estimated_prob / (i + 1))
        # else:
        #     # scale estimated prob using softmax, to reduce the effect of very small probs
        #     prob_sum = np.sum(
        #         [np.exp(cand_form.estimated_prob) for cand_form in meta_feature.candidate_formula_list])
        #     sum_normed_estimated_prob = 0
        #     for i, cand_form in enumerate(meta_feature.candidate_formula_list):
        #         this_normed_estimated_prob = np.exp(cand_form.estimated_prob) / prob_sum
        #         sum_normed_estimated_prob += this_normed_estimated_prob
        #
        #         cand_form.normed_estimated_prob = this_normed_estimated_prob
        #         cand_form.estimated_fdr = 1 - (sum_normed_estimated_prob / (i + 1))

    # update back
    buddy_data[batch_start_idx:batch_end_idx] = batch_data
    return
