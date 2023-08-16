# candidate ranking with machine learning (model B)
import numpy as np
from numba import njit
from msbuddy.service import common_nl_from_array
from msbuddy.utils import dependencies
from scipy.stats import norm
import warnings


# ignore warnings
warnings.filterwarnings('ignore')


def gen_ml_a_array_from_buddy_data(buddy_data) -> (np.array, np.array, np.array):
    """
    generate three arrays for ML model A
    :param buddy_data: List of MetaFeature objects
    :return: all_cand_form_arr, dbe_arr, mass_arr
    """
    all_cand_form_arr = np.empty((0, 12))  # formula array of all candidate formulas
    dbe_arr = np.array([])
    mass_arr = np.array([])

    for meta_feature in buddy_data:
        if not meta_feature.candidate_formula_list:
            continue
        # generate ML features for each candidate formula
        for candidate_formula in meta_feature.candidate_formula_list:
            # add formula array to all_cand_form_arr
            all_cand_form_arr = np.append(all_cand_form_arr, [candidate_formula.formula.array], axis=0)
            dbe_arr = np.append(dbe_arr, candidate_formula.formula.dbe)
            mass_arr = np.append(mass_arr, candidate_formula.formula.mass)

    return all_cand_form_arr, dbe_arr, mass_arr


def gen_ml_a_feature(all_cand_form_arr, dbe_arr, mass_arr) -> np.array:
    """
    generate ML features for model A (formula feasibility)
    :param all_cand_form_arr: numpy array of all candidate formula arrays
    :param dbe_arr: numpy array of all candidate formula dbe
    :param mass_arr: numpy array of all candidate formula mass
    :return: numpy array of ML features
    """
    # calculate ML features
    hal_arr = np.sum(all_cand_form_arr[:, 2:6], axis=1)  # sum of halogen atoms
    ta_arr = np.sum(all_cand_form_arr, axis=1)  # total number of atoms
    f_exist_arr = np.clip(all_cand_form_arr[:, 4], 0, 1)  # whether F exists
    cl_exist_arr = np.clip(all_cand_form_arr[:, 3], 0, 1)  # whether Cl exists
    br_exist_arr = np.clip(all_cand_form_arr[:, 2], 0, 1)  # whether Br exists
    i_exist_arr = np.clip(all_cand_form_arr[:, 5], 0, 1)  # whether I exists
    hal_ele_type_arr = f_exist_arr + cl_exist_arr + br_exist_arr + i_exist_arr  # number of halogen elements
    hal_two = np.clip(hal_ele_type_arr - 1, 0, 1)  # whether more than one halogen element exists
    hal_three = np.clip(hal_ele_type_arr - 2, 0, 1)  # whether more than two halogen elements exist
    senior_1_1_arr = 6 * all_cand_form_arr[:, 11] + 5 * all_cand_form_arr[:, 10] + 4 * all_cand_form_arr[:, 0] + \
        3 * all_cand_form_arr[:, 7] + 2 * all_cand_form_arr[:, 9] + all_cand_form_arr[:, 1] + hal_arr
    senior_1_2_arr = all_cand_form_arr[:, 7] + all_cand_form_arr[:, 10] + all_cand_form_arr[:, 1] + hal_arr

    # halogen to hydrogen ratio, fill 0 if hydrogen = 0
    hal_h_arr = np.zeros(len(hal_arr))
    # if hydrogen > 0
    h_bool_arr = all_cand_form_arr[:, 1] > 0
    hal_h_arr[h_bool_arr] = hal_arr[h_bool_arr] / all_cand_form_arr[h_bool_arr, 1]

    # O/P ratio, fill 1 if phosphorus = 0
    o_p_arr = np.ones(len(hal_arr))
    # if phosphorus > 0
    p_bool_arr = all_cand_form_arr[:, 10] > 0
    o_p_arr[p_bool_arr] = all_cand_form_arr[p_bool_arr, 9] / all_cand_form_arr[p_bool_arr, 10] / 3

    # DBE binary, 1 if DBE > 0, 0 if DBE = 0
    dbe_binary_arr = np.clip(dbe_arr, 0, 1)

    # generate output array
    out = np.array(
        [all_cand_form_arr[:, 0] / ta_arr, all_cand_form_arr[:, 1] / ta_arr, all_cand_form_arr[:, 7] / ta_arr,
         all_cand_form_arr[:, 9] / ta_arr, all_cand_form_arr[:, 10] / ta_arr, all_cand_form_arr[:, 11] / ta_arr,
         hal_arr / ta_arr, senior_1_1_arr, senior_1_2_arr, 2 * ta_arr - 1, dbe_arr, dbe_binary_arr,
         np.sqrt(dbe_arr / mass_arr), dbe_arr / np.power(mass_arr / 100, 2 / 3), hal_two, hal_three,
         all_cand_form_arr[:, 1] / all_cand_form_arr[:, 0], all_cand_form_arr[:, 7] / all_cand_form_arr[:, 0],
         all_cand_form_arr[:, 9] / all_cand_form_arr[:, 0], all_cand_form_arr[:, 10] / all_cand_form_arr[:, 0],
         all_cand_form_arr[:, 11] / all_cand_form_arr[:, 0], hal_arr / all_cand_form_arr[:, 0],
         hal_h_arr, o_p_arr]).T

    # normalize each col using mean_arr and std_arr in global dependencies
    for i in range(out.shape[1]):
        out[:, i] = (out[:, i] - dependencies['model_a_mean_arr'][i]) / dependencies['model_a_std_arr'][i]

    # for columns 16-21, fill inf and nan with 0
    out[:, 16:22] = np.nan_to_num(out[:, 16:22], nan=0, posinf=0, neginf=0)

    return out


def predict_ml_a(feature_arr: np.array) -> np.array:
    """
    predict formula feasibility using model A
    :param feature_arr: numpy array of ML features
    :return: numpy array of prediction results
    """
    # model A should be loaded in global dependencies
    prob_arr = dependencies['model_a'].predict_proba(feature_arr)

    return prob_arr[:, 1]


def predict_formula_feasibility(buddy_data) -> None:
    """
    predict formula feasibility using ML model A
    :param buddy_data: buddy data
    :return: fill in ml_a_prob in candidate formula objects
    """
    # generate three arrays from buddy data
    cand_form_arr, dbe_arr, mass_arr = gen_ml_a_array_from_buddy_data(buddy_data)
    # generate ML feature array
    feature_arr = gen_ml_a_feature(cand_form_arr, dbe_arr, mass_arr)
    # predict formula feasibility
    prob_arr = predict_ml_a(feature_arr)

    # add prediction results to candidate formula objects in the list
    cnt = 0
    for meta_feature in buddy_data:
        if not meta_feature.candidate_formula_list:
            continue
        # generate ML features for each candidate formula
        for candidate_formula in meta_feature.candidate_formula_list:
            candidate_formula.ml_a_prob = prob_arr[cnt]
            cnt += 1


def gen_ml_B_feature(buddy_data, ppm, ms2_tol) -> np.array:
    """
    generate ML features for model B, for all metabolic features
    :param buddy_data: List of MetaFeature objects
    :param ppm: whether to use ppm error
    :param ms1_tol: m/z tolerance for MS1
    :param ms2_tol: m/z tolerance for MS2
    :return: numpy array of ML features
    """
    # generate feature array
    total_feature_arr = np.array([])

    for meta_feature in buddy_data:
        if not meta_feature.candidate_formula_list:
            continue
        # generate ML features for each candidate formula
        for candidate_formula in meta_feature.candidate_formula_list:
            # get ML features
            ml_feature_arr = gen_ml_B_feature_single(meta_feature, candidate_formula, ppm, ms2_tol)
            # add to feature array
            if total_feature_arr.size == 0:
                total_feature_arr = ml_feature_arr
            else:
                total_feature_arr = np.vstack((total_feature_arr, ml_feature_arr))

    return total_feature_arr


def gen_ml_B_feature_single(meta_feature, cand_form, ppm, ms1_tol, ms2_tol) -> np.array:
    """
    generate ML features for model B for a single candidate formula
    :param meta_feature: MetaFeature object
    :param cand_form: CandidateFormula object
    :param ppm: whether to use ppm error
    :param ms1_tol: m/z tolerance for MS1
    :param ms2_tol: m/z tolerance for MS2
    :return: numpy array of ML features
    """

    this_adduct = meta_feature.adduct
    this_form = cand_form.formula  # neutral formula

    # mz error in ppm
    theo_mass = (this_form.mass * this_adduct.m + this_adduct.net_formula.mass -
                 this_adduct.charge * 0.0005486) / abs(this_adduct.charge)
    mz_error = (meta_feature.mz - theo_mass) / theo_mass * 1e6 if ppm else meta_feature.mz - theo_mass
    mz_error_feature = _calc_log_p_norm(mz_error, ms1_tol/3)

    # precursor charged formula
    pre_charged_arr = this_form.array * this_adduct.m + this_adduct.net_formula.array
    pre_dbe = this_form.dbe * this_adduct.m - this_adduct.m + this_adduct.net_formula.dbe
    pre_h2c = pre_charged_arr[1] / pre_charged_arr[0] if pre_charged_arr[0] > 0 else 2  # no carbon, assign 2

    # MS1 isotope similarity
    ms1_iso_sim = cand_form.ms1_isotope_similarity if cand_form.ms1_isotope_similarity else 0

    # MS/MS-related features
    ms2_feature_arr = _gen_ms2_feature_array(meta_feature, cand_form.ms2_raw_explanation, pre_dbe, pre_h2c,
                                             ppm, ms2_tol)

    # generate output array
    out = np.array([ms1_iso_sim, cand_form.ml_a_prob, mz_error_feature, pre_dbe, pre_h2c])
    out = np.append(out, ms2_feature_arr)

    return out


def _gen_ms2_feature_array(meta_feature, ms2_explanation, pre_dbe, pre_h2c, ppm, ms2_tol) -> np.array:
    """
    generate MS/MS-related features for a single candidate formula
    :param meta_feature: MetaFeature object
    :param ms2_explanation: MS2Explanation object
    :param pre_dbe: precursor DBE
    :param pre_h2c: precursor H/C ratio
    :param ppm: whether to use ppm error
    :param ms2_tol: m/z tolerance for MS2
    :return: numpy array of MS/MS-related features
    """
    # valid MS2 explanation
    if meta_feature.ms2_processed and ms2_explanation:
        # explained fragment ion count percentage
        exp_frag_cnt_pct = len(ms2_explanation) / len(meta_feature.ms2_processed)

        # explained fragment ion
        exp_idx_arr = ms2_explanation.idx_array
        exp_int_arr = meta_feature.ms2_raw.int_array[exp_idx_arr]
        exp_mz_arr = meta_feature.ms2_raw.mz_array[exp_idx_arr]
        # valid fragment ion after MS2 processing
        valid_idx_arr = meta_feature.ms2_processed.idx_array
        valid_int_arr = meta_feature.ms2_raw.int_array[valid_idx_arr]

        # explained fragment ion intensity percentage
        exp_frag_int_pct = np.sum(exp_int_arr) / np.sum(valid_int_arr)

        frag_form_arr = ms2_explanation.explanation_array  # array of fragment formulas, Formula objects

        # subformula count: how many frags are subformula of other frags
        subform_score, subform_common_loss_score = _calc_subformula_score(frag_form_arr)

        # radical ion count percentage (out of all explained fragment ions)
        radical_cnt_pct = np.sum([1 for frag_form in frag_form_arr if frag_form.dbe % 1 == 0]) / len(frag_form_arr)

        # normalized explained intensity array
        normed_exp_int_arr = exp_int_arr / np.sum(exp_int_arr)

        # weighted average of fragment DBEs
        frag_dbe_wavg = np.sum(np.array([frag_form.dbe for frag_form in frag_form_arr]) * normed_exp_int_arr)

        # weighted average of fragment H/C ratios
        frag_h2c_wavg = np.sum(np.array([frag_form.array[1] / frag_form.array[0] if frag_form.array[0] > 0 else pre_h2c
                                         for frag_form in frag_form_arr]) * normed_exp_int_arr)

        # weighted average of fragment m/z ppm errors
        if ppm:
            frag_mz_err_wavg = np.sum(np.array([_calc_log_p_norm((frag_form.mass - mz) / frag_form.mass * 1e6,
                                                                 ms2_tol/3)
                                                for frag_form, mz in
                                                zip(frag_form_arr, exp_mz_arr)]) * normed_exp_int_arr)
        else:
            frag_mz_err_wavg = np.sum(np.array([_calc_log_p_norm(frag_form.mass - mz, ms2_tol/3)
                                                for frag_form, mz in
                                                zip(frag_form_arr, exp_mz_arr)]) * normed_exp_int_arr)

        # weighted average of fragment-nl DBE difference
        frag_nl_dbe_diff_wavg = np.sum(np.array([frag_form.dbe - (pre_dbe - frag_form.dbe + 1)
                                                 for frag_form in frag_form_arr]) * normed_exp_int_arr)

        out_arr = np.array([exp_frag_cnt_pct, exp_frag_int_pct, subform_score, subform_common_loss_score,
                            radical_cnt_pct, frag_dbe_wavg, frag_h2c_wavg, frag_mz_err_wavg, frag_nl_dbe_diff_wavg])
    else:
        out_arr = np.array([0] * 9)

    return out_arr


def _calc_subformula_score(frag_form_arr) -> (float, float):
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
    subform_cnt, subform_common_loss_cnt = _subformula_check(all_frag_arr, dependencies['common_loss_db'])

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
    arr_norm_p = np.where(arr_norm_p > 0.5, 1 - arr_norm_p, arr_norm_p)
    log_p_arr = np.log(arr_norm_p * 2)
    # clip to the range of [-4, 0]
    log_p_arr = np.clip(log_p_arr, -4, 0)

    return log_p_arr


if __name__ == '__main__':

    # from file_io.init_db import init_db
    # import time
    #
    # start_time = time.time()
    # init_db(0)
    # print('init_db time: ', time.time() - start_time)
    #
    # # test _subformula_check
    # start_time = time.time()
    # for i in range(1000):
    #     # create a test array of shape (10, 12), fill with random integers
    #     test_arr = np.random.randint(0, 10, size=(30, 12))
    #     # print(test_arr)
    #
    #     subform_count, subform_common_loss_count = _subformula_check(test_arr, dependencies['common_loss_db'])
    # print('subformula check time: ', time.time() - start_time)

    import time
    from msbuddy.file_io import load_usi
    from msbuddy.base_class import ProcessedMS2
    import numpy as np
    from file_io import init_db
    from msbuddy.gen_candidate import gen_candidate_formula

    start_1 = time.time()
    init_db(0)
    print("init db: " + str(time.time() - start_1))

    usi_1 = 'mzspec:GNPS:TASK-c95481f0c53d42e78a61bf899e9f9adb-spectra/specs_ms.mgf:scan:1943'
    usi_2 = 'mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00005436077'
    usi_3 = 'mzspec:MSV000078547:120228_nbut_3610_it_it_take2:scan:389'

    _data = load_usi([usi_1, usi_2, usi_3])

    for feature in _data:
        feature.ms2_processed = ProcessedMS2(feature.mz, feature.ms2_raw, 20, True, True, True, 0.02, 0.80, 0.25, 50)

        # generate formula candidates
        start = time.time()
        gen_candidate_formula(feature, True, 10, 20, 0, False, np.array([0] * 12),
                              np.array([50, 200, 5, 5, 5, 5, 0, 30, 0, 20, 10, 10]), 4)
        print("query: " + str(time.time() - start))
        print("length:  " + str(len(feature.candidate_formula_list)))

    # predict formula feasibility
    start = time.time()
    feat_arr = gen_ml_B_feature(_data)
    print("feature gen: " + str(time.time() - start))
    print(feat_arr)
    print("Done.")