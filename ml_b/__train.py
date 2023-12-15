import argparse
import json
from brainpy import isotopic_variants
import numpy as np
from tqdm import tqdm
import pandas as pd
from numba import njit
import joblib
from scipy.stats import norm
from msbuddy.base import MetaFeature, Spectrum, Formula, CandidateFormula, Adduct
from msbuddy.main import Msbuddy, MsbuddyConfig, _gen_subformula
from msbuddy.load import init_db
from msbuddy.ml import gen_ml_b_feature_single, pred_formula_feasibility
from msbuddy.cand import _calc_ms1_iso_sim
from msbuddy.utils import form_arr_to_str, read_formula

import lightgbm as lgb
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import ndcg_score

from __train_gnps_cmd import send_hotmail_email, sim_ms1_iso_pattern


def load_gnps_data(path):
    """
    load GNPS library
    :param path: path to GNPS library
    """
    db = pd.read_csv(path, sep='\t', index_col=0)

    # test
    print('db size: ' + str(len(db)))

    qtof_mf_ls = []  # metaFeature
    orbi_mf_ls = []
    ft_mf_ls = []
    qtof_gt_ls = []  # ground truth formula
    orbi_gt_ls = []
    ft_gt_ls = []
    for i in range(len(db)):
        # parse formula info
        formula = db['FORMULA'][i]
        gt_form_arr = read_formula(formula)

        # skip if formula is not valid
        if gt_form_arr is None or np.sum(gt_form_arr) == 0:
            continue

        # calculate theoretical mass
        theo_mass = Formula(gt_form_arr, 0).mass
        # print(db['ADDUCT'][i])
        adduct = Adduct(db['ADDUCT'][i], True if db['IONMODE'][i] == 'positive' else False, True)
        theo_mz = theo_mass + adduct.net_formula.mass / adduct.charge
        theo_mz = theo_mz - 0.00054858 * adduct.charge

        # simulate ms1 isotope pattern
        # print(db.index[i])
        ms1_gt_arr, ms1_sim_arr = sim_ms1_iso_pattern(gt_form_arr)
        # create a numpy array of ms1 mz, with length equal to the length of ms1_sim_arr, step size = 1.003355
        ms1_mz_arr = np.array([theo_mz + x * 1.003355 for x in range(len(ms1_sim_arr))])

        # parse ms2 info
        ms2_mz = np.array(db['ms2mz'][i].split(',')).astype(np.float64)
        ms2_int = np.array(db['ms2int'][i].split(',')).astype(np.float64)

        mf = MetaFeature(identifier=db.index[i],
                         mz=theo_mz,
                         charge=1 if db['IONMODE'][i] == 'positive' else -1,
                         ms1=Spectrum(ms1_mz_arr, ms1_sim_arr),
                         ms2=Spectrum(ms2_mz, ms2_int))

        # mz tolerance, depends on the instrument
        if db['INSTRUMENT_TYPE'][i] == 'qtof':
            qtof_gt_ls.append(gt_form_arr)  # add to ground truth formula list
            qtof_mf_ls.append(mf)
        elif db['INSTRUMENT_TYPE'][i] == 'orbitrap':
            orbi_gt_ls.append(gt_form_arr)  # add to ground truth formula list
            orbi_mf_ls.append(mf)
        else:  # FT-ICR
            ft_gt_ls.append(gt_form_arr)  # add to ground truth formula list
            ft_mf_ls.append(mf)

    # save to joblib file one by one
    joblib.dump(qtof_mf_ls, 'gnps_qtof_mf_ls.joblib')
    joblib.dump(orbi_mf_ls, 'gnps_orbi_mf_ls.joblib')
    joblib.dump(ft_mf_ls, 'gnps_ft_mf_ls.joblib')
    joblib.dump(qtof_gt_ls, 'gnps_qtof_gt_ls.joblib')
    joblib.dump(orbi_gt_ls, 'gnps_orbi_gt_ls.joblib')
    joblib.dump(ft_gt_ls, 'gnps_ft_gt_ls.joblib')


def pred_formula_feasibility_batch(data, db_mode, shared_data_dict, batch_size):
    n_batch = int(np.ceil(len(data) / batch_size))
    for n in tqdm(range(n_batch)):
        start_idx = n * batch_size
        end_idx = min((n + 1) * batch_size, len(data))
        pred_formula_feasibility(data, start_idx, end_idx, db_mode, shared_data_dict)

    return data


def calc_gnps_data(n_cpu, timeout_secs, instru):
    # main
    param_set = MsbuddyConfig(ms1_tol=10, ms2_tol=20, parallel=True, n_cpu=n_cpu, batch_size=999999,
                              halogen=True, timeout_secs=timeout_secs)
    buddy = Msbuddy(param_set)
    shared_data_dict = init_db()  # database initialization

    if instru == 'qtof':
        qtof_mf_ls = joblib.load('gnps_qtof_mf_ls.joblib')
        buddy.add_data(qtof_mf_ls)
        buddy._preprocess_and_generate_candidate_formula(0, len(buddy.data))
        joblib.dump(buddy.data, 'gnps_qtof_mf_ls_cand_1.joblib')
        new_data = pred_formula_feasibility_batch(buddy.data, 1, shared_data_dict, 1000)
        joblib.dump(new_data, 'gnps_qtof_mf_ls_cand_2.joblib')
    elif instru == 'orbi':
        orbi_mf_ls = joblib.load('gnps_orbi_mf_ls.joblib')
        # update parameters
        buddy.update_config(ms1_tol=5, ms2_tol=10, parallel=True, n_cpu=n_cpu,
                            halogen=True, batch_size=999999, timeout_secs=timeout_secs)
        buddy.add_data(orbi_mf_ls)
        buddy._preprocess_and_generate_candidate_formula(0, len(buddy.data))
        joblib.dump(buddy.data, 'gnps_orbi_mf_ls_cand_1.joblib')
        new_data = pred_formula_feasibility_batch(buddy.data, 1, shared_data_dict, 1000)
        joblib.dump(new_data, 'gnps_orbi_mf_ls_cand_2.joblib')
    else:  # FT-ICR
        ft_mf_ls = joblib.load('gnps_ft_mf_ls.joblib')
        # update parameters
        buddy.update_config(ms1_tol=2, ms2_tol=5, parallel=True, n_cpu=n_cpu,
                            halogen=True, batch_size=999999, timeout_secs=timeout_secs)
        buddy.add_data(ft_mf_ls)
        buddy._preprocess_and_generate_candidate_formula(0, len(buddy.data))
        joblib.dump(buddy.data, 'gnps_ft_mf_ls_cand_1.joblib')
        new_data = pred_formula_feasibility_batch(buddy.data, 1, shared_data_dict, 1000)
        joblib.dump(new_data, 'gnps_ft_mf_ls_cand_2.joblib')

    return shared_data_dict


@njit
def _calc_ml_a_array(form_arr, mass, dbe):
    # calculate ML features
    hal = np.sum(form_arr[2:6])  # sum of halogen atoms
    ta = np.sum(form_arr)  # total number of atoms
    f_exist = 1 if 0 <= form_arr[4] <= 1 else 0
    cl_exist = 1 if 0 <= form_arr[3] <= 1 else 0
    br_exist = 1 if 0 <= form_arr[2] <= 1 else 0
    i_exist = 1 if 0 <= form_arr[5] <= 1 else 0
    hal_ele_type_arr = f_exist + cl_exist + br_exist + i_exist  # number of halogen elements
    hal_two = 1 if hal_ele_type_arr >= 2 else 0  # whether more than one halogen element exists
    hal_three = 1 if hal_ele_type_arr >= 3 else 0  # whether more than two halogen elements exist
    senior_1_1 = 6 * form_arr[11] + 5 * form_arr[10] + 4 * form_arr[0] + \
                 3 * form_arr[7] + 2 * form_arr[9] + form_arr[1] + hal
    senior_1_2 = form_arr[7] + form_arr[10] + form_arr[1] + hal

    # halogen to H ratio
    hal_h = 0 if form_arr[1] == 0 else hal / form_arr[1]

    # O/P ratio, fill 1 if phosphorus = 0
    o_p = 1 if form_arr[10] == 0 else form_arr[9] / form_arr[10]

    # if C > 0
    if form_arr[0] > 0:
        out = np.array([form_arr[0] / ta, form_arr[1] / ta,
                        form_arr[7] / ta,
                        form_arr[9] / ta, form_arr[10] / ta,
                        form_arr[11] / ta,
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
        out = np.array([form_arr[0] / ta, form_arr[1] / ta,
                        form_arr[7] / ta,
                        form_arr[9] / ta, form_arr[10] / ta,
                        form_arr[11] / ta,
                        hal / ta, senior_1_1, senior_1_2, 2 * ta - 1, dbe,
                        np.sqrt(dbe / mass), dbe / np.power(mass / 100, 2 / 3),
                        0, 0, 0, 0, 0, 0,
                        hal_h, o_p, hal_two, hal_three])

    return out


def assign_subform_gen_training_data(instru):
    """
    assign subformula annotation and generate training data; reduce memory usage
    """
    # generate ML features for each candidate formula, for ML model B
    # generate feature array
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

    print('loading data...')
    data_name = 'gnps_' + instru + '_mf_ls_cand_2.joblib'
    data = joblib.load(data_name)
    buddy.add_data(data)
    del data

    gt_name = 'gnps_' + instru + '_gt_ls.joblib'
    gt_ls = joblib.load(gt_name)

    for k, meta_feature in enumerate(buddy.data):
        print('k: ' + str(k) + ' out of ' + str(len(buddy.data)))
        gt_form_arr = gt_ls[k]
        gt_form_str = form_arr_to_str(gt_form_arr)
        if not meta_feature.candidate_formula_list:
            continue

        # modify the candidate formula list, such that the ground truth formula is the first one
        # ml_a_prob = buddy.predict_formula_feasibility(gt_form_arr)
        # ml_a_prob = 1
        form = Formula(gt_form_arr, 0)
        this_cf = CandidateFormula(form)
        this_cf.ml_a_array = _calc_ml_a_array(gt_form_arr, form.mass, form.dbe)
        cand_form_ls = [this_cf]

        cand_cnt = 0
        for cf in meta_feature.candidate_formula_list:
            if cand_cnt > 200:
                break
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
            # calc ms1 iso similarity
            cf.ms1_isotope_similarity = _calc_ms1_iso_sim(cf, mf, 4)
            this_true = True if n == 0 else False

            # get ML features
            ml_feature_arr = gen_ml_b_feature_single(mf, cf, True, ms1_tol, ms2_tol, shared_data_dict)

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

    print('y_arr sum: ' + str(np.sum(y_arr)))
    X_arr_name = 'gnps_X_arr_' + instru + '.joblib'
    y_arr_name = 'gnps_y_arr_' + instru + '.joblib'
    joblib.dump(X_arr, X_arr_name)
    joblib.dump(y_arr, y_arr_name)
    joblib.dump(group_arr, 'gnps_group_arr_' + instru + '.joblib')


def combine_and_clean_X_y():
    # load training data
    X_arr_qtof = joblib.load('gnps_X_arr_qtof.joblib')
    y_arr_qtof = joblib.load('gnps_y_arr_qtof.joblib')
    X_arr_orbi = joblib.load('gnps_X_arr_orbi.joblib')
    y_arr_orbi = joblib.load('gnps_y_arr_orbi.joblib')
    X_arr_ft = joblib.load('gnps_X_arr_ft.joblib')
    y_arr_ft = joblib.load('gnps_y_arr_ft.joblib')
    group_arr_qtof = joblib.load('gnps_group_arr_qtof.joblib')
    group_arr_orbi = joblib.load('gnps_group_arr_orbi.joblib')
    group_arr_ft = joblib.load('gnps_group_arr_ft.joblib')

    X_arr = np.vstack((X_arr_qtof, X_arr_orbi, X_arr_ft))
    y_arr = np.append(y_arr_qtof, np.append(y_arr_orbi, y_arr_ft))
    group_arr = np.append(group_arr_qtof, np.append(group_arr_orbi, group_arr_ft))
    print('X_arr shape: ' + str(X_arr.shape))
    print('y_arr shape: ' + str(y_arr.shape))
    print('group_arr shape: ' + str(group_arr.shape))

    # remove rows with nan
    X_arr = np.where(X_arr == None, np.nan, X_arr)
    X_arr = X_arr.astype(np.float64)
    nan_rows = np.any(np.isnan(X_arr), axis=1)
    # Remove those rows from X_arr and y_arr
    X_arr = X_arr[~nan_rows]
    y_arr = y_arr[~nan_rows]
    group_arr = group_arr[~nan_rows]
    print('X_arr shape: ' + str(X_arr.shape))
    print('y_arr shape: ' + str(y_arr.shape))
    print('group_arr shape: ' + str(group_arr.shape))

    # save to joblib file
    joblib.dump(X_arr, 'gnps_X_arr.joblib')
    joblib.dump(y_arr, 'gnps_y_arr.joblib')
    joblib.dump(group_arr, 'gnps_group_arr.joblib')


def train_model(ms1_iso, ms2_spec, pswd):
    """
    train ML model B
    :return: trained model
    """
    # load training data
    X_arr = joblib.load('gnps_X_arr.joblib')
    y_arr = joblib.load('gnps_y_arr.joblib')
    group_arr = joblib.load('gnps_group_arr.joblib')

    print("Training model ...")
    if not ms1_iso:
        # discard the ms1 iso feature in X_arr
        X_arr = X_arr[:, 1:]
    if not ms2_spec:
        # discard the last 14 features in X_arr
        X_arr = X_arr[:, :-14]

    print('splitting...')
    # split training and testing data
    X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
        X_arr, y_arr, group_arr, test_size=0.2, random_state=42)

    # Create LightGBM datasets
    train_data = lgb.Dataset(data=X_train, label=y_train, group=groups_train)
    test_data = lgb.Dataset(data=X_test, label=y_test, group=groups_test)

    # Parameters for training
    params = {
        'objective': 'rank_xendcg',
        'metric': 'ndcg',
        'ndcg_at': [1],
        'learning_rate': 0.05,
        'num_leaves': 31,
        'num_iterations': 100,
        'num_boost_round': 100,
        'early_stopping_rounds': 10,
        'verbose': 1
    }
    # Train the model
    print("Training model...")
    gbm = lgb.train(params, train_data, valid_sets=[test_data])

    # Predict on test data
    print("Evaluating model...")
    test_preds = gbm.predict(X_test)
    test_group_size = groups_test[0]
    test_ndcg_score = ndcg_score(
        [y_test[i:i + test_group_size] for i in range(0, len(y_test), test_group_size)],
        [test_preds[i:i + test_group_size] for i in range(0, len(test_preds), test_group_size)],
        k=1
    )
    out_str = f'Final NDCG@1 score on test data: {test_ndcg_score}'

    # Save the model
    model_filename = 'ranking_model.joblib'
    joblib.dump(gbm, model_filename)

    return out_str


def parse_args():
    """
    parse command line arguments
    :return: parsed arguments
    """
    parser = argparse.ArgumentParser(description='ML model B training')
    parser.add_argument('-calc', action='store_true', help='calculate gnps data')
    parser.add_argument('-gen', action='store_true', help='generate training data')
    parser.add_argument('-ms', type=str, help='instrument type')
    parser.add_argument('-cpu', type=int, default=4, help='number of CPU cores to use')
    parser.add_argument('-to', type=int, default=1200, help='timeout in seconds')
    parser.add_argument('-ms1', action='store_true', help='ms1 iso similarity included')
    parser.add_argument('-ms2', action='store_true', help='MS/MS spec included')
    parser.add_argument('-p', type=str, help='password for email')
    args = parser.parse_args()
    return args


# test
if __name__ == '__main__':
    import time

    start_time = time.time()
    __package__ = "msbuddy"

    # parse arguments
    args = parse_args()

    # test here
    # args = argparse.Namespace(calc=True, ms='ft', cpu=1, to=6000,
    #                           ms1=False, ms2=True)

    # load training data
    # load_gnps_data('ms2db_selected_with_ms2.tsv')

    email_body = ''

    if args.calc:
        calc_gnps_data(args.cpu, args.to, args.ms)  # qtof, orbi, ft
        # assign_subform_gen_training_data(instru=args.ms)

    elif args.gen:
        combine_and_clean_X_y()
        # z_norm()  # z-normalization

    else:  # train model
        email_body = train_model(args.ms1, args.ms2, args.p)

    time_elapsed = time.time() - start_time
    time_elapsed = time_elapsed / 3600

    email_body += '\n\nTime elapsed: ' + str(time_elapsed) + ' hrs'

    send_hotmail_email("Job finished", email_body,
                       "s1xing@health.ucsd.edu", smtp_password=args.p)
