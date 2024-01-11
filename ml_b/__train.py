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
        out = np.array([form_arr[0], form_arr[1],
                        form_arr[7], form_arr[9], form_arr[10],
                        form_arr[11], hal, ta,
                        form_arr[0] / ta, form_arr[1] / ta,
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
        out = np.array([form_arr[0], form_arr[1],
                        form_arr[7], form_arr[9], form_arr[10],
                        form_arr[11], hal, ta,
                        form_arr[0] / ta, form_arr[1] / ta,
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
            # print('n: ' + str(n) + ' out of ' + str(len(mf.candidate_formula_list)))
            # print('cf: ' + str(cf.formula.array))
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


def combine_and_clean_x_y():
    # load training data
    X_arr_qtof = joblib.load('gnps_X_arr_qtof_cor.joblib')
    y_arr_qtof = joblib.load('gnps_y_arr_qtof.joblib')
    X_arr_orbi = joblib.load('gnps_X_arr_orbi_cor.joblib')
    y_arr_orbi = joblib.load('gnps_y_arr_orbi.joblib')
    X_arr_ft = joblib.load('gnps_X_arr_ft_cor.joblib')
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

    # # remove rows with nan
    # X_arr = np.where(X_arr == None, np.nan, X_arr)
    # X_arr = X_arr.astype(np.float64)
    # nan_rows = np.any(np.isnan(X_arr), axis=1)
    # # Remove those rows from X_arr and y_arr
    # X_arr = X_arr[~nan_rows]
    # y_arr = y_arr[~nan_rows]
    # group_arr = group_arr[~nan_rows]
    # print('X_arr shape: ' + str(X_arr.shape))
    # print('y_arr shape: ' + str(y_arr.shape))
    # print('group_arr shape: ' + str(group_arr.shape))

    # for every element in X arr, round to 8 decimal places
    X_arr = np.round(X_arr, 8)

    # save to joblib file
    joblib.dump(X_arr, 'gnps_X_arr.joblib')
    joblib.dump(y_arr, 'gnps_y_arr.joblib')
    joblib.dump(group_arr, 'gnps_group_arr.joblib')


def tune_hyperparams(X_arr, y_arr, group_arr):
    # Parameters for training
    param_grid = {
        'objective': ['lambdarank'],  # 'rank_xendcg', 'lambdarank'
        'metric': ['ndcg'],
        'ndcg_at': [[1]],
        'learning_rate': [0.01],
        'num_leaves': [500],  # [500, 750, 1000, 1250],
        'max_depth': [-1],
        'min_data_in_leaf': [20],  # [10, 20, 30],
        'max_bin': [200],
        'bagging_fraction': [0.9],  # [0.9, 1], optimized
        'bagging_freq': [1],
        'feature_fraction': [1],
        'lambda_l1': [0],
        'lambda_l2': [0],
        'seed': [24],
        'verbose': [0]
    }

    print("Grid search for hyperparameter tuning...")
    best_params, best_score = grid_search_cv(X_arr, y_arr, group_arr, param_grid)
    print(f"Best Parameters: {best_params}, Best Score: {best_score}")

    return best_params


def train_model(ms1_iso, ms2_spec):
    """
    train ML model B
    :return: trained model
    """
    # load training data
    X_arr = joblib.load('gnps_X_arr.joblib')
    y_arr = joblib.load('gnps_y_arr.joblib')
    group_arr = joblib.load('gnps_group_arr.joblib')

    # group arr as int
    group_arr = group_arr.astype(np.int32)
    assert np.sum(group_arr) == len(X_arr) == len(y_arr)

    if not ms1_iso:
        # discard the ms1 iso feature in X_arr
        X_arr = X_arr[:, 1:]
    if not ms2_spec:
        # discard the last 14 features in X_arr
        X_arr = X_arr[:, :-14]

    # hyperparameter tuning
    # best_params = tune_hyperparams(X_arr, y_arr, group_arr)
    best_params = {'objective': 'lambdarank', 'metric': 'ndcg', 'ndcg_at': [1],
                   'learning_rate': 0.01, 'num_leaves': 1500, 'max_depth': -1, 'min_data_in_leaf': 20,
                   'lambda_l1': 0, 'lambda_l2': 0, 'seed': 24, 'verbose': 0}

    # Split training and testing data
    (X_train, X_val, X_test, y_train, y_val, y_test,
     groups_train, groups_val, groups_test) = _train_val_test_split(X_arr, y_arr, group_arr,
                                                                    val_size=0.1, test_size=0.1, random_state=24)

    # Create LightGBM datasets
    train_data = lgb.Dataset(data=X_train, label=y_train, group=groups_train)
    val_data = lgb.Dataset(data=X_val, label=y_val, group=groups_val)
    # test_data = lgb.Dataset(data=X_test, label=y_test, group=groups_test)

    print("Training model...")
    # Train the model
    gbm = lgb.train(best_params, train_data, valid_sets=[val_data], num_boost_round=1000,
                    callbacks=[lgb.early_stopping(stopping_rounds=30)])

    # Predict on test data
    print("Evaluating model...")
    # Predict on test data
    test_preds = gbm.predict(X_test)

    top_1_accuracies = []  # Including all group sizes

    start_idx = 0
    for group_size in groups_test:
        end_idx = start_idx + group_size
        true_labels = y_test[start_idx:end_idx]
        predicted_scores = test_preds[start_idx:end_idx]
        # Find the index of the highest predicted score
        top_prediction_idx = np.argmax(predicted_scores)
        # Check if the top prediction is correct
        is_correct = true_labels[top_prediction_idx] == 1
        top_1_accuracies.append(is_correct)
        start_idx = end_idx

    # Calculate top-1 accuracy
    top_1_accuracy = np.mean(top_1_accuracies)
    print(f"Top-1 Accuracy (including single-item groups): {top_1_accuracy}")

    # Calculate NDCG score for each group with more than one item and average
    ndcg_scores = []
    start_idx = 0
    for group_size in groups_test:
        end_idx = start_idx + group_size
        # Only calculate NDCG for groups with more than one item
        if group_size > 1:
            true_labels = y_test[start_idx:end_idx]
            predicted_scores = test_preds[start_idx:end_idx]
            if np.any(true_labels):  # Check if there are any positive labels
                ndcg_scores.append(ndcg_score([true_labels], [predicted_scores], k=1))
        start_idx = end_idx

    # Calculate average NDCG score
    avg_ndcg_score = np.mean(ndcg_scores)
    out_str = f'Average NDCG@1 score on test data: {avg_ndcg_score}'
    print(out_str)

    # heuristic baseline model
    # mass error if MS2 only
    feature_idx = 1 if ms1_iso else 0
    avg_ndcg_score_heuristic = heuristic_ranking_baseline(X_test, y_test, groups_test, feature_idx, ascending=True)
    print(f'Average NDCG@1 score with heuristic baseline: {avg_ndcg_score_heuristic}')

    # exp_frag_int_pct if MS2 only
    avg_ndcg_score_heuristic = heuristic_ranking_baseline(X_test, y_test, groups_test, -12, ascending=False)
    print(f'Average NDCG@1 score with heuristic baseline: {avg_ndcg_score_heuristic}')

    # Save the model
    model_name = 'ml_b'
    model_name += '_ms1' if ms1_iso else ''
    model_name += '_ms2' if ms2_spec else ''
    model_name += '.joblib'
    joblib.dump(gbm, model_name)

    return


def _train_val_test_split(X_arr, y_arr, group_arr, val_size=0.1, test_size=0.1, random_state=42):
    # Calculate the cumulative sum of group sizes
    cumulative_group_sizes = np.cumsum(group_arr)

    # Indices where each group starts and ends
    group_indices = np.zeros((len(group_arr), 2), dtype=int)
    group_indices[1:, 0] = cumulative_group_sizes[:-1]
    group_indices[:, 1] = cumulative_group_sizes

    # Decide which groups go into training, validation, and testing
    np.random.seed(random_state)
    random_values = np.random.rand(len(group_arr))
    is_val_group = random_values < val_size
    is_test_group = (random_values >= val_size) & (random_values < val_size + test_size)
    is_train_group = random_values >= val_size + test_size

    # Corrected slicing using group indices
    X_train, X_val, X_test = [], [], []
    y_train, y_val, y_test = [], [], []
    groups_train, groups_val, groups_test = [], [], []

    for i, is_train in enumerate(is_train_group):
        if is_train:
            X_train.extend(X_arr[group_indices[i, 0]:group_indices[i, 1]])
            y_train.extend(y_arr[group_indices[i, 0]:group_indices[i, 1]])
            groups_train.append(group_arr[i])
        elif is_val_group[i]:
            X_val.extend(X_arr[group_indices[i, 0]:group_indices[i, 1]])
            y_val.extend(y_arr[group_indices[i, 0]:group_indices[i, 1]])
            groups_val.append(group_arr[i])
        elif is_test_group[i]:
            X_test.extend(X_arr[group_indices[i, 0]:group_indices[i, 1]])
            y_test.extend(y_arr[group_indices[i, 0]:group_indices[i, 1]])
            groups_test.append(group_arr[i])

    # Convert lists to arrays
    X_train = np.array(X_train)
    X_val = np.array(X_val)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_val = np.array(y_val)
    y_test = np.array(y_test)

    return X_train, X_val, X_test, y_train, y_val, y_test, groups_train, groups_val, groups_test


def _train_test_split(X_arr, y_arr, group_arr, test_size=0.1, random_state=24):
    # Calculate the cumulative sum of group sizes
    cumulative_group_sizes = np.cumsum(group_arr)

    # Indices where each group starts and ends
    group_indices = np.zeros((len(group_arr), 2), dtype=int)
    group_indices[1:, 0] = cumulative_group_sizes[:-1]
    group_indices[:, 1] = cumulative_group_sizes

    # Decide which groups go into training, validation, and testing
    np.random.seed(random_state)
    random_values = np.random.rand(len(group_arr))
    is_train_group = random_values >= test_size

    # Corrected slicing using group indices
    X_train, X_test = [], []
    y_train, y_test = [], []
    groups_train, groups_test = [], []

    for i, is_train in enumerate(is_train_group):
        if is_train:
            X_train.extend(X_arr[group_indices[i, 0]:group_indices[i, 1]])
            y_train.extend(y_arr[group_indices[i, 0]:group_indices[i, 1]])
            groups_train.append(group_arr[i])
        else:
            X_test.extend(X_arr[group_indices[i, 0]:group_indices[i, 1]])
            y_test.extend(y_arr[group_indices[i, 0]:group_indices[i, 1]])
            groups_test.append(group_arr[i])

    # Convert lists to arrays
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return X_train, X_test, y_train, y_test, groups_train, groups_test


def perform_cross_validation(X, y, group_sizes, params, n_splits=5):
    """
    Perform cross-validation.
    """
    X_1, X_test, y_1, y_test, groups_1, groups_test = _train_test_split(X, y, group_sizes)

    # Transform group_sizes to group assignments
    groups = np.repeat(np.arange(len(groups_1)), groups_1)

    group_kfold = GroupKFold(n_splits=n_splits)
    val_ndcg_scores = []
    test_ndcg_scores = []

    for train_idx, test_idx in group_kfold.split(X_1, y_1, groups):
        X_train, X_val = X_1[train_idx], X_1[test_idx]
        y_train, y_val = y_1[train_idx], y_1[test_idx]

        # Extract train and test group information
        train_groups = np.array([len(np.where(groups == g)[0]) for g in np.unique(groups[train_idx])])
        val_groups = np.array([len(np.where(groups == g)[0]) for g in np.unique(groups[test_idx])])

        train_data = lgb.Dataset(data=X_train, label=y_train, group=train_groups)
        val_data = lgb.Dataset(data=X_val, label=y_val, group=val_groups)

        gbm = lgb.train(params, train_data, valid_sets=[val_data], num_boost_round=1000,
                        callbacks=[lgb.early_stopping(stopping_rounds=30)])

        val_ndcg_scores.append(gbm.best_score['valid_0']['ndcg@1'])

        # predict on test data
        test_preds = gbm.predict(X_test)
        start_idx = 0
        _test_ndcg_scores = []
        for group_size in groups_test:
            end_idx = start_idx + group_size
            # Only calculate NDCG for groups with more than one item
            if group_size > 1:
                true_labels = y_test[start_idx:end_idx]
                predicted_scores = test_preds[start_idx:end_idx]
                if np.any(true_labels):  # Check if there are any positive labels
                    this_ndcg_score = ndcg_score([true_labels], [predicted_scores], k=1)
                    _test_ndcg_scores.append(this_ndcg_score)
            start_idx = end_idx
        test_ndcg_scores.append(np.mean(_test_ndcg_scores))
        print(f"Test NDCG@1 Score (mean): {np.mean(_test_ndcg_scores)}")

    avg_val_ndcg_score = np.mean(val_ndcg_scores)
    avg_test_ndcg_score = np.mean(test_ndcg_scores)
    return avg_val_ndcg_score, avg_test_ndcg_score


def grid_search_cv(X, y, groups, param_grid, n_splits=5):
    best_score = 0
    best_params = None

    # Generate all combinations of hyperparameters
    from itertools import product
    keys, values = zip(*param_grid.items())
    for param_values in product(*values):
        params = dict(zip(keys, param_values))

        # Perform cross-validation
        val_score, test_score = perform_cross_validation(X, y, groups, params, n_splits=n_splits)

        if test_score > best_score:
            best_score = test_score
            best_params = params

        print(f"Tested params: {params}, test NDCG@1 Score: {test_score}, val NDCG@1 Score: {val_score}")

    return best_params, best_score


def heuristic_ranking_baseline(X_test, y_test, groups_test, feature_index, ascending=True):
    ndcg_scores = []
    start_idx = 0
    for group_size in groups_test:
        end_idx = start_idx + group_size
        true_labels = y_test[start_idx:end_idx]
        heuristic_scores = X_test[start_idx:end_idx, feature_index] if ascending else -X_test[start_idx:end_idx, feature_index]
        if group_size > 1 and np.any(true_labels):
            ndcg_scores.append(ndcg_score([true_labels], [heuristic_scores], k=1))
        start_idx = end_idx
    return np.mean(ndcg_scores) if ndcg_scores else None


def get_feature_importance(gbm, ms1, ms2):
    feature_names = ['ms1_iso_sim', 'mz_error_log_p', 'pos_mode', 'chon', 'chonps',
                     'c_ta', 'h_ta', 'n_ta', 'o_ta', 'p_ta', 's_ta', 'hal_ta', 'senior_1_1', 'senior_1_2', '2ta_1',
                     'dbe', 'dbe_mass_1', 'dbe_mass_2', 'h_c', 'n_c', 'o_c', 'p_c', 's_c', 'hal_c', 'hal_h',
                     'o_p', 'hal_two', 'hal_three', 'exp_frag_cnt_pct', 'exp_db_frag_cnt_pct', 'exp_db_frag_int_pct',
                     'subform_score', 'subform_common_loss_score',
                     'radical_cnt_pct', 'frag_dbe_wavg', 'frag_h2c_wavg', 'frag_mz_err_wavg', 'frag_nl_dbe_diff_wavg',
                     'valid_ms2_peak', 'exp_frag_cnt_pct_sqrt', 'exp_frag_int_pct_sqrt']
    if not ms1:
        feature_names = feature_names[1:]
    if not ms2:
        feature_names = feature_names[:-14]
    # Assuming 'gbm' is your trained LightGBM model
    # and 'feature_names' is the list of names of the features

    # Feature importance by split
    importance_split = gbm.feature_importance(importance_type='split')
    feature_importance_split = sorted(zip(feature_names, importance_split), key=lambda x: x[1], reverse=True)

    # Feature importance by gain
    importance_gain = gbm.feature_importance(importance_type='gain')
    feature_importance_gain = sorted(zip(feature_names, importance_gain), key=lambda x: x[1], reverse=True)

    # print
    print('feature importance by split:')
    for feature_name, score in feature_importance_split:
        print(f'{feature_name}: {score}')
    print('\n\nfeature importance by gain:')
    for feature_name, score in feature_importance_gain:
        print(f'{feature_name}: {score}')

    return feature_importance_split, feature_importance_gain


def correct_x_z_norm():
    _, mean_arr, std_arr = joblib.load('ml_a_v0.2.4.joblib')
    mean_arr = mean_arr[8:]
    std_arr = std_arr[8:]
    for ms in ['qtof', 'orbi', 'ft']:
        X_arr = joblib.load('gnps_X_arr_' + ms + '.joblib')
        y_arr = joblib.load('gnps_y_arr_' + ms + '.joblib')
        non_correct_bool = y_arr == 0
        # revert z-normalization
        X_arr[non_correct_bool, 5:26] = X_arr[non_correct_bool, 5:26] * std_arr + mean_arr
        joblib.dump(X_arr, 'gnps_X_arr_' + ms + '.joblib')


def correct_x_ml_a_for_gt():
    for instru in ['ft', 'qtof', 'orbi']:
        print('loading data...')
        data_name = 'gnps_' + instru + '_mf_ls_cand_2.joblib'
        data = joblib.load(data_name)

        gt_name = 'gnps_' + instru + '_gt_ls.joblib'
        gt_ls = joblib.load(gt_name)

        X_arr = joblib.load('gnps_X_arr_' + instru + '.joblib')
        y_arr = joblib.load('gnps_y_arr_' + instru + '.joblib')
        correct_idx = np.where(y_arr == 1)[0]

        tmp = 0
        for k, meta_feature in enumerate(data):
            if not meta_feature.candidate_formula_list:
                continue
            gt_form_arr = gt_ls[k]

            # modify the candidate formula list, such that the ground truth formula is the first one
            form = Formula(gt_form_arr, 0)
            new_ml_a_array = _calc_ml_a_array(gt_form_arr, form.mass, form.dbe)

            # fill in the new ml_a_array
            X_arr[correct_idx[tmp], 5:28] = new_ml_a_array[8:].copy()
            tmp += 1

        assert tmp == len(correct_idx)
        X_arr_name = 'gnps_X_arr_' + instru + '_cor.joblib'
        joblib.dump(X_arr, X_arr_name)

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

    # ###############
    # # cmd
    # # parse arguments
    # args = parse_args()
    #
    # # load training data
    # # load_gnps_data('ms2db_selected_with_ms2.tsv')
    #
    # email_body = ''
    #
    # if args.calc:
    #     # calc_gnps_data(args.cpu, args.to, args.ms)  # qtof, orbi, ft
    #     assign_subform_gen_training_data(instru=args.ms)
    #
    # elif args.gen:
    #     combine_and_clean_x_y(test=False)
    #     # z_norm()  # z-normalization
    #
    # else:  # train model
    #     email_body = train_model(args.ms1, args.ms2)
    #
    # time_elapsed = time.time() - start_time
    # time_elapsed = time_elapsed / 3600
    #
    # email_body += '\n\nTime elapsed: ' + str(time_elapsed) + ' hrs'
    #
    # send_hotmail_email("Job finished", email_body,
    #                    "s1xing@health.ucsd.edu", smtp_password=args.p)

    ###############
    # local
    args = argparse.Namespace(calc=False, gen=True, ms='ft', cpu=1, to=999999,
                              ms1=True, ms2=True)

    # correct_x_z_norm()
    # correct_x_ml_a_for_gt()

    # combine_and_clean_x_y()
    train_model(args.ms1, args.ms2)

    # get_feature_importance(joblib.load('ml_b_ms1_ms2.joblib'), True, True)
    # get_feature_importance(joblib.load('ml_b_ms1.joblib'), True, False)
    # get_feature_importance(joblib.load('ml_b_ms2.joblib'), False, True)
    # get_feature_importance(joblib.load('ml_b.joblib'), False, False)

    print('done')