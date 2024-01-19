import joblib
import numpy as np
from numba import njit
from scipy.stats import norm
from tqdm import tqdm

from msbuddy import form_arr_to_str
from msbuddy.base import Formula, CandidateFormula
from msbuddy.cand import _calc_ms1_iso_sim
from msbuddy.load import init_db
from msbuddy.main import Msbuddy, MsbuddyConfig, _gen_subformula
from msbuddy.ml import _gen_arr_from_buddy_data, _gen_form_feature, _fill_form_feature_arr_in_batch_data, \
    gen_ml_feature_single


def fill_form_feature_arr_batch(data, batch_size=1000):
    n_batch = int(np.ceil(len(data) / batch_size))
    for n in tqdm(range(n_batch)):
        start_idx = n * batch_size
        end_idx = min((n + 1) * batch_size, len(data))
        batch_data = data[start_idx:end_idx]

        # generate three arrays from buddy data
        cand_form_arr, dbe_arr, mass_arr = _gen_arr_from_buddy_data(batch_data)

        # generate ML feature array
        feature_arr = _gen_form_feature(cand_form_arr, dbe_arr, mass_arr)

        # fill in batch_data
        _fill_form_feature_arr_in_batch_data(batch_data, feature_arr)
        data[start_idx:end_idx] = batch_data
    return data


def fill_formula_feature_array(ms='qtof'):
    # main
    data_name = 'gnps_' + ms + '_mf_ls_cand_1.joblib'
    data = joblib.load(data_name)
    new_data = fill_form_feature_arr_batch(data)
    new_data_name = 'gnps_' + ms + '_mf_ls_cand_2_new.joblib'
    joblib.dump(new_data, new_data_name)  # new, 2 more features in the start


def split_into_batches(ms='qtof', batch_size=3000):
    # main
    data_name = 'gnps_' + ms + '_mf_ls_cand_2_new.joblib'
    data = joblib.load(data_name)
    n_batch = int(np.ceil(len(data) / batch_size))
    for n in tqdm(range(n_batch)):
        start_idx = n * batch_size
        end_idx = min((n + 1) * batch_size, len(data))
        batch_data = data[start_idx:end_idx]
        batch_data_name = 'batch/gnps_' + ms + '_mf_ls_cand_2_new_batch_' + str(n) + '.joblib'
        joblib.dump(batch_data, batch_data_name)


def gen_subform_and_X_y_arr(instru='qtof', n_batch=0, batch_size=3000):
    print('batch: ' + str(n_batch))
    # read data
    data_name = 'batch/gnps_' + instru + '_mf_ls_cand_2_new_batch_' + str(n_batch) + '.joblib'
    data = joblib.load(data_name)
    print('data loaded')

    X_arr = np.array([])
    y_arr = np.array([])
    group_arr = np.array([])

    if instru == 'qtof':
        ms1_tol = 10
        ms2_tol = 20
    elif instru == 'orbi':
        ms1_tol = 5
        ms2_tol = 10
    else:  # FT-ICR
        ms1_tol = 2
        ms2_tol = 5

    param_set = MsbuddyConfig(ms1_tol=ms1_tol, ms2_tol=ms2_tol, halogen=True)
    buddy = Msbuddy(param_set)
    shared_data_dict = init_db()  # database initialization

    buddy.add_data(data)
    del data

    gt_name = 'gnps_' + instru + '_gt_ls.joblib'
    gt_ls = joblib.load(gt_name)
    start_idx = n_batch * batch_size
    end_idx = min((n_batch + 1) * batch_size, len(gt_ls))
    gt_ls = gt_ls[start_idx:end_idx]

    print('generating subform and X, y arrays...')
    for k, meta_feature in enumerate(buddy.data):
        print('k: ' + str(k) + ' out of ' + str(len(buddy.data)))
        gt_form_arr = gt_ls[k]
        gt_form_str = form_arr_to_str(gt_form_arr)
        if not meta_feature.candidate_formula_list:
            continue

        # modify the candidate formula list, such that the ground truth formula is the first one
        form = Formula(gt_form_arr, 0)
        this_cf = CandidateFormula(formula=form,
                                   charged_formula=Formula(gt_form_arr + meta_feature.adduct.net_formula.array,
                                                           meta_feature.adduct.charge))
        this_cf.formula_feature_array = _calc_form_feature_array(gt_form_arr, form.mass, form.dbe)
        cand_form_ls = [this_cf]

        cand_cnt = 0
        for cf in meta_feature.candidate_formula_list:
            # if cand_cnt > 200:
            #     break
            if gt_form_str == form_arr_to_str(cf.formula.array):
                continue
            cand_form_ls.append(cf)
            cand_cnt += 1

        meta_feature.candidate_formula_list = cand_form_ls
        group_arr = np.append(group_arr, len(cand_form_ls))

        # assign subformula annotation
        mf = _gen_subformula(meta_feature, buddy.config)

        # generate ML features for each candidate formula
        for n, cf in enumerate(mf.candidate_formula_list):
            # print('n: ' + str(n) + ' out of ' + str(len(mf.candidate_formula_list)))
            # print('cf: ' + str(cf.formula.array))
            # calc ms1 iso similarity
            cf.ms1_isotope_similarity = _calc_ms1_iso_sim(cf, mf, 4)
            this_true = True if n == 0 else False

            # get ML features
            ml_feature_arr = gen_ml_feature_single(mf, cf, True, ms1_tol, ms2_tol, shared_data_dict)

            # if true gt, perform precursor simulation
            if this_true:
                mz_shift = np.random.normal(0, ms1_tol / 5)
                mz_shift_p = norm.cdf(mz_shift, loc=0, scale=ms1_tol / 3)
                mz_shift_p = mz_shift_p if mz_shift_p < 0.5 else 1 - mz_shift_p
                log_p = np.log(mz_shift_p * 2)
                ml_feature_arr[1] = np.clip(log_p, -2, 0)

            # add to feature array
            if X_arr.size == 0:
                X_arr = ml_feature_arr
                y_arr = np.array([1 if this_true else 0])
            else:
                X_arr = np.vstack((X_arr, ml_feature_arr))
                y_arr = np.append(y_arr, 1 if this_true else 0)
        del mf
        buddy.data[k] = None

    # write out
    print('writing out...')
    X_name = 'batch/gnps_X_arr_' + instru + '_batch_' + str(n_batch) + '.joblib'
    y_name = 'batch/gnps_y_arr_' + instru + '_batch_' + str(n_batch) + '.joblib'
    group_name = 'batch/gnps_group_arr_' + instru + '_batch_' + str(n_batch) + '.joblib'
    joblib.dump(X_arr, X_name)
    joblib.dump(y_arr, y_name)
    joblib.dump(group_arr, group_name)


def aggregate_X_and_y(ms='qtof', total_batch=9):
    X = joblib.load('batch/gnps_X_arr_' + ms + '_batch_0.joblib')
    y = joblib.load('batch/gnps_y_arr_' + ms + '_batch_0.joblib')
    group = joblib.load('batch/gnps_group_arr_' + ms + '_batch_0.joblib')
    for i in range(1, total_batch):
        X = np.vstack((X, joblib.load('batch/gnps_X_arr_' + ms + '_batch_' + str(i) + '.joblib')))
        y = np.append(y, joblib.load('batch/gnps_y_arr_' + ms + '_batch_' + str(i) + '.joblib'))
        group = np.append(group, joblib.load('batch/gnps_group_arr_' + ms + '_batch_' + str(i) + '.joblib'))
    joblib.dump(X, 'gnps_X_arr_' + ms + '.joblib')
    joblib.dump(y, 'gnps_y_arr_' + ms + '.joblib')
    joblib.dump(group, 'gnps_group_arr_' + ms + '.joblib')


@njit
def _calc_form_feature_array(form_arr, mass, dbe):
    # calculate ML features
    ele_sum_1_arr = form_arr[2] + form_arr[3] + form_arr[4] + form_arr[5] + form_arr[6] + form_arr[8]
    ele_sum_2_arr = ele_sum_1_arr + form_arr[10] + form_arr[11]
    chon_only = 1 - np.clip(ele_sum_2_arr, 0, 1)  # whether only C, H, O, N exist
    chonps_only = 1 - np.clip(ele_sum_1_arr, 0, 1)  # whether only C, H, O, N, P, S exist

    chno = form_arr[0] + form_arr[1] + form_arr[7] + form_arr[9]
    hal = np.sum(form_arr[2:6])  # sum of halogen atoms
    ta = np.sum(form_arr)  # total number of atoms
    f_exist = 1 if form_arr[4] >= 1 else 0
    cl_exist = 1 if form_arr[3] >= 1 else 0
    br_exist = 1 if form_arr[2] >= 1 else 0
    i_exist = 1 if form_arr[5] >= 1 else 0
    hal_ele_type_arr = f_exist + cl_exist + br_exist + i_exist  # number of halogen elements
    hal_two = 1 if hal_ele_type_arr >= 2 else 0  # whether more than one halogen element exists
    hal_three = 1 if hal_ele_type_arr >= 3 else 0  # whether more than two halogen elements exist
    senior_1_1 = (6 * form_arr[11] + 5 * form_arr[10] + 4 * form_arr[0] + 3 * form_arr[7] + 2 * form_arr[9] +
                  form_arr[1] + hal)
    senior_1_2 = form_arr[7] + form_arr[10] + form_arr[1] + hal

    # halogen to H ratio
    hal_h = 0 if form_arr[1] == 0 else hal / form_arr[1]

    # O/P ratio, fill 1 if phosphorus = 0
    o_p = 1 if form_arr[10] == 0 else form_arr[9] / form_arr[10] / 3

    # if C > 0
    if form_arr[0] > 0:
        out = np.array([chon_only, chonps_only,
                        form_arr[0] / ta, form_arr[1] / ta,
                        form_arr[7] / ta,
                        form_arr[9] / ta, form_arr[10] / ta,
                        form_arr[11] / ta, chno / ta,
                        hal / ta, senior_1_1, senior_1_2, 2 * ta - 1, dbe,
                        np.sqrt(dbe / mass), dbe / np.power(mass / 100, 2 / 3),
                        form_arr[1] / form_arr[0],
                        form_arr[7] / form_arr[0],
                        form_arr[9] / form_arr[0],
                        form_arr[10] / form_arr[0],
                        form_arr[11] / form_arr[0],
                        hal / form_arr[0],
                        hal_h, o_p, hal_two, hal_three])
    else:
        out = np.array([chon_only, chonps_only,
                        form_arr[0] / ta, form_arr[1] / ta,
                        form_arr[7] / ta,
                        form_arr[9] / ta, form_arr[10] / ta,
                        form_arr[11] / ta, chno / ta,
                        hal / ta, senior_1_1, senior_1_2, 2 * ta - 1, dbe,
                        np.sqrt(dbe / mass), dbe / np.power(mass / 100, 2 / 3),
                        0, 0, 0, 0, 0, 0,
                        hal_h, o_p, hal_two, hal_three])

    return out


if __name__ == '__main__':

    # fill_formula_feature_array('qtof')
    # fill_formula_feature_array('orbi')

    # split_into_batches('orbi', 10000)

    gen_subform_and_X_y_arr('orbi', 0, 10000)
    # gen_subform_and_X_y_arr('orbi', 1, 10000)
    # gen_subform_and_X_y_arr('orbi', 2, 10000)
    # gen_subform_and_X_y_arr('orbi', 3, 10000)
    # gen_subform_and_X_y_arr('orbi', 4, 10000)
    # gen_subform_and_X_y_arr('orbi', 5, 10000)
    # gen_subform_and_X_y_arr('orbi', 6, 10000)
    # gen_subform_and_X_y_arr('orbi', 7, 10000)
    # gen_subform_and_X_y_arr('orbi', 8, 10000)
    #
    # aggregate_X_and_y('orbi', 9)
    #
    #
    # split_into_batches('qtof', 3000)
    #
    # gen_subform_and_X_y_arr('qtof', 0, 3000)
    # gen_subform_and_X_y_arr('qtof', 1, 3000)
    # gen_subform_and_X_y_arr('qtof', 2, 3000)
    # gen_subform_and_X_y_arr('qtof', 3, 3000)
    # gen_subform_and_X_y_arr('qtof', 4, 3000)
    # gen_subform_and_X_y_arr('qtof', 5, 3000)
    # gen_subform_and_X_y_arr('qtof', 6, 3000)
    # gen_subform_and_X_y_arr('qtof', 7, 3000)
    # gen_subform_and_X_y_arr('qtof', 8, 3000)
    #
    # aggregate_X_and_y('qtof', 9)

