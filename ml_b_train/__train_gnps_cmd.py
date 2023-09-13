import argparse
import json

import joblib
import numpy as np
from brainpy import isotopic_variants
from imblearn.over_sampling import SMOTE
from scipy.stats import norm
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier

from msbuddy.base import read_formula, MetaFeature, Spectrum, Formula
from msbuddy.buddy import Buddy, BuddyParamSet
from msbuddy.load import init_db
from msbuddy.ml import gen_ml_b_feature_single, pred_formula_feasibility
from msbuddy.gen_candidate import _calc_ms1_iso_sim

# This MLP model is trained using GNPS library.
# 4 models will be generated in total: ms1 iso similarity included or not, MS/MS spec included or not.


def sim_ms1_iso_pattern(form_arr):
    """
    simulate MS1 isotope pattern
    :param form_arr: numpy array of formula
    :return: theoretical & simulated isotope pattern
    """

    # calculate theoretical isotope pattern
    # mapping to a dictionary
    arr_dict = {}
    for i, element in enumerate(Formula.alphabet):
        arr_dict[element] = form_arr[i]

    # calculate isotope pattern
    isotope_pattern = isotopic_variants(arr_dict, npeaks=4)
    int_arr = np.array([iso.intensity for iso in isotope_pattern])

    # simulation
    sim_int_arr = int_arr.copy()
    a1, a2, a3 = 2, 2, 2
    b1, b2, b3 = -1, -1, -1

    # M + 1
    while a1 * b1 < -1:
        a1 = abs(np.random.normal(0, 0.11))
        b1 = np.random.choice([-1, 1])
    sim_int_arr[1] = sim_int_arr[1] * (1 + a1 * b1)

    # M + 2
    if len(int_arr) >= 3:
        while a2 * b2 < -1:
            a2 = abs(np.random.normal(0, 0.16))
            # random.choice([-1, 1]), 0.7 probability to be 1
            b2 = np.random.choice([-b1, b1], p=[0.3, 0.7])
        sim_int_arr[2] = sim_int_arr[2] * (1 + a2 * b2)

    # M + 3
    if len(int_arr) >= 4:
        while a3 * b3 < -1:
            a3 = abs(np.random.normal(0, 0.19))
            b3 = np.random.choice([-b2, b2], p=[0.3, 0.7])
        sim_int_arr[3] = sim_int_arr[3] * (1 + a3 * b3)

    return int_arr, sim_int_arr


def load_gnps_data(path):
    """
    load GNPS library
    :param path: path to GNPS library
    """
    db = joblib.load(path)

    # # test
    # print('db size: ' + str(len(db)))
    db = db[:200]

    qtof_mf_ls = []  # metaFeature
    orbi_mf_ls = []
    ft_mf_ls = []
    qtof_gt_ls = []  # ground truth formula
    orbi_gt_ls = []
    ft_gt_ls = []
    for i in range(len(db)):
        # parse formula info
        formula = db['formula'][i]
        gt_form_arr = read_formula(formula)

        # skip if formula is not valid
        if gt_form_arr is None:
            continue

        # calculate theoretical mass
        theo_mass = Formula(gt_form_arr, 0).mass
        theo_mz = theo_mass + 1.007276 if db['ionmode'][i] == 'positive' else theo_mass - 1.007276

        # simulate ms1 isotope pattern
        ms1_gt_arr, ms1_sim_arr = sim_ms1_iso_pattern(gt_form_arr)
        # create a numpy array of ms1 mz, with length equal to the length of ms1_sim_arr, step size = 1.003355
        ms1_mz_arr = np.array([theo_mz + x * 1.003355 for x in range(len(ms1_sim_arr))])

        # parse ms2 info
        ms2_mz = np.array(json.loads(db['ms2mz'][i]))
        ms2_int = np.array(json.loads(db['ms2int'][i]))

        mf = MetaFeature(identifier=i,
                         mz=theo_mz,
                         charge=1 if db['ionmode'][i] == 'positive' else -1,
                         ms1=Spectrum(ms1_mz_arr, ms1_sim_arr),
                         ms2=Spectrum(ms2_mz, ms2_int))

        # mz tolerance, depends on the instrument
        if db['instrument'][i] == 'qtof':
            qtof_gt_ls.append(gt_form_arr)  # add to ground truth formula list
            qtof_mf_ls.append(mf)
        elif db['instrument'][i] == 'orbitrap':
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


def calc_gnps_data(parallel, n_cpu, timeout_secs, instru='qtof'):

    # main
    param_set = BuddyParamSet(ms1_tol=10, ms2_tol=20, parallel=parallel, n_cpu=n_cpu,
                              halogen=True, timeout_secs=timeout_secs)
    buddy = Buddy(param_set)
    shared_data_dict = init_db(buddy.param_set.db_mode)  # database initialization

    if instru == 'qtof':
        # qtof_mf_ls = joblib.load('gnps_qtof_mf_ls.joblib')
        # buddy.add_data(qtof_mf_ls)
        # buddy.preprocess_and_generate_candidate_formula()
        # joblib.dump(buddy.data, 'gnps_qtof_mf_ls_cand_1.joblib')
        # pred_formula_feasibility(buddy.data, shared_data_dict)
        # joblib.dump(buddy.data, 'gnps_qtof_mf_ls_cand_2.joblib')
        data = joblib.load('gnps_qtof_mf_ls_cand_2.joblib')
        buddy.add_data(data)
        buddy.assign_subformula_annotation()
        joblib.dump(buddy.data, 'gnps_qtof_mf_ls_cand.joblib')
    elif instru == 'orbi':
        # orbi_mf_ls = joblib.load('gnps_orbi_mf_ls.joblib')
        # update parameters
        buddy.update_param_set(BuddyParamSet(ms1_tol=5, ms2_tol=10, parallel=parallel, n_cpu=n_cpu,
                                             halogen=True,
                                             timeout_secs=timeout_secs))
        # buddy.clear_data()
        # buddy.add_data(orbi_mf_ls)
        # buddy.preprocess_and_generate_candidate_formula()
        # joblib.dump(buddy.data, 'gnps_orbi_mf_ls_cand_1.joblib')
        # pred_formula_feasibility(buddy.data, shared_data_dict)
        # joblib.dump(buddy.data, 'gnps_orbi_mf_ls_cand_2.joblib')
        data = joblib.load('gnps_orbi_mf_ls_cand_2.joblib')
        buddy.add_data(data)
        buddy.assign_subformula_annotation()
        joblib.dump(buddy.data, 'gnps_orbi_mf_ls_cand.joblib')
    else:  # FT-ICR
        # ft_mf_ls = joblib.load('gnps_ft_mf_ls.joblib')
        # update parameters
        buddy.update_param_set(BuddyParamSet(ms1_tol=2, ms2_tol=5, parallel=parallel, n_cpu=n_cpu,
                                             halogen=True,
                                             timeout_secs=timeout_secs))
        # buddy.clear_data()
        # buddy.add_data(ft_mf_ls)
        # buddy.preprocess_and_generate_candidate_formula()
        # joblib.dump(buddy.data, 'gnps_ft_mf_ls_cand_1.joblib')
        # pred_formula_feasibility(buddy.data, shared_data_dict)
        # joblib.dump(buddy.data, 'gnps_ft_mf_ls_cand_2.joblib')
        data = joblib.load('gnps_ft_mf_ls_cand_2.joblib')
        buddy.add_data(data)
        buddy.assign_subformula_annotation()
        joblib.dump(buddy.data, 'gnps_ft_mf_ls_cand.joblib')

    return shared_data_dict


def gen_training_data(gd):
    """
    generate training data for ML model B, including precursor simulation
    :param gd: global data
    :return: write to joblib file
    """
    mf_ls_ls = [joblib.load('gnps_qtof_mf_ls_cand.joblib'),
                joblib.load('gnps_orbi_mf_ls_cand.joblib'),
                joblib.load('gnps_ft_mf_ls_cand.joblib')]
    gt_ls_ls = [joblib.load('gnps_qtof_gt_ls.joblib'),
                joblib.load('gnps_orbi_gt_ls.joblib'),
                joblib.load('gnps_ft_gt_ls.joblib')]

    # generate ML features for each candidate formula, for ML model B
    # generate feature array
    X_arr = np.array([])
    y_arr = np.array([])

    for cnt1, mf_ls in enumerate(mf_ls_ls):
        gt_ls = gt_ls_ls[cnt1]
        if cnt1 == 0:  # Q-TOF
            ms1_tol = 10
            ms2_tol = 20
        elif cnt1 == 1:  # Orbitrap
            ms1_tol = 5
            ms2_tol = 10
        else:  # FT-ICR
            ms1_tol = 2
            ms2_tol = 5
        for cnt2, mf in enumerate(mf_ls):
            gt_form_arr = gt_ls[cnt2]
            if not mf.candidate_formula_list:
                continue
            # generate ML features for each candidate formula
            for cf in mf.candidate_formula_list:
                # calc ms1 iso similarity
                cf.ms1_isotope_similarity = _calc_ms1_iso_sim(cf, mf, 4)
                this_true = False
                if (gt_form_arr == cf.formula.array).all():
                    this_true = True
                # get ML features
                ml_feature_arr = gen_ml_b_feature_single(mf, cf, True, ms1_tol, ms2_tol, gd)

                # if true gt, perform precursor simulation
                if this_true:
                    mz_shift = np.random.normal(0, ms1_tol / 5)
                    mz_shift_p = norm.cdf(mz_shift, loc=0, scale=ms1_tol / 3)
                    mz_shift_p = mz_shift_p if mz_shift_p < 0.5 else 1 - mz_shift_p
                    log_p = np.log(mz_shift_p * 2)
                    ml_feature_arr[3] = np.clip(log_p, -4, 0)

                # add to feature array
                if X_arr.size == 0:
                    X_arr = ml_feature_arr
                    y_arr = np.array([1 if this_true else 0])
                else:
                    X_arr = np.vstack((X_arr, ml_feature_arr))
                    y_arr = np.append(y_arr, 1 if this_true else 0)

    print('y_arr sum: ' + str(np.sum(y_arr)))
    joblib.dump(X_arr, 'gnps_X_arr.joblib')
    joblib.dump(y_arr, 'gnps_y_arr.joblib')


def train_model(X_arr, y_arr, ms1_iso, ms2_spec):
    """
    train ML model B
    :param X_arr: feature array
    :param y_arr: label array
    :param ms1_iso: True for ms1 iso similarity included, False for not included
    :param ms2_spec: True for MS/MS spec included, False for not included
    :return: trained model
    """

    print('SMOTE...')
    # Apply SMOTE to the training data
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_arr, y_arr)

    joblib.dump(X_resampled, 'gnps_X_arr_SMOTE.joblib')
    joblib.dump(y_resampled, 'gnps_y_arr_SMOTE.joblib')

    print("Training model ...")
    if not ms1_iso:
        # discard the 2nd feature in X_arr
        X_resampled = np.delete(X_resampled, 1, axis=1)
    if not ms2_spec:
        # discard the last 10 features in X_arr
        X_resampled = X_resampled[:, :-10]

    print('splitting...')
    # split training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=0)

    # grid search
    all_param_grid = {
        'hidden_layer_sizes': [(1024,), (512,), (256,), (128,), (64,),
                               (512, 512), (256, 256), (128, 128), (128, 64), (64, 64), (64, 32), (32, 32),
                               (64, 32, 32), (32, 32, 32), (32, 32, 16), (16, 32, 16), (32, 16, 16),
                               (128, 64, 64, 32), (64, 64, 32, 32), (32, 32, 16, 16),
                               (32, 64, 64, 32, 16), (32, 32, 32, 16, 8)],
        'activation': ['relu'],
        'alpha': [1e-5, 1e-4, 1e-3, 1e-2],
        'max_iter': [800]
    }

    # grid search
    mlp = MLPClassifier(random_state=1)
    clf = GridSearchCV(mlp, all_param_grid, cv=5, n_jobs=-1, scoring='roc_auc', verbose=1)
    clf.fit(X_train, y_train)

    # print best parameters
    print("Best parameters set found on development set:")
    print(clf.best_params_)
    best_params = clf.best_params_

    print("Grid scores on development set:")
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))

    print("train model...")
    # train model with best params for 3 times, and choose the best one
    best_score = 0
    for i in range(3):
        mlp = MLPClassifier(**best_params, random_state=i).fit(X_train, y_train)
        score = mlp.score(X_test, y_test)
        if score > best_score:
            best_score = score
            best_mlp = mlp

    score = best_mlp.score(X_test, y_test)  # accuracy on test data
    print("MLP acc.: " + str(score))

    # predict on test data
    y_pred = best_mlp.predict(X_test)

    # print performance
    print("Classification report for classifier %s:\n%s\n"
          % (best_mlp, metrics.classification_report(y_test, y_pred)))

    # save model
    model_name = 'model_b'
    model_name += '_ms1' if ms1_iso else '_noms1'
    model_name += '_ms2' if ms2_spec else '_noms2_gnps'
    joblib.dump(best_mlp, model_name + '.joblib')

    return best_mlp


def parse_args():
    """
    parse command line arguments
    :return: parsed arguments
    """
    parser = argparse.ArgumentParser(description='ML model B training')
    parser.add_argument('-gen', action='store_true', help='generate training data')
    parser.add_argument('-calc', action='store_true', help='calculate gnps data')
    parser.add_argument('-instru', type=str, default='qtof', help='instrument type')
    parser.add_argument('-parallel', action='store_true', help='parallel mode')
    parser.add_argument('-n_cpu', type=int, default=16, help='number of CPU cores to use')
    parser.add_argument('-to', type=int, default=600, help='timeout in seconds')
    parser.add_argument('-ms1', action='store_true', help='ms1 iso similarity included')
    parser.add_argument('-ms2', action='store_true', help='MS/MS spec included')
    args = parser.parse_args()
    return args


# test
if __name__ == '__main__':
    # __package__ = "msbuddy"
    # parse arguments
    # args = parse_args()

    # # test here
    args = argparse.Namespace(gen=False, calc=True, instru='ft',
                              parallel=False, n_cpu=1, to=1000,
                              ms1=True, ms2=True)

    # /Users/philip/Documents/projects/ms2/gnps/

    # load training data
    if args.gen:
        load_gnps_data('gnps_ms2db_preprocessed_20230910.joblib')

    elif args.calc:
        gd = calc_gnps_data(args.parallel, args.n_cpu, args.to, args.instru)
        gen_training_data(gd)
        print("Done.")

    else:  # train model
        X = joblib.load('gnps_X_arr.joblib')
        y = joblib.load('gnps_y_arr.joblib')

        # train models
        train_model(X, y, args.ms1, args.ms2)

    print("Done.")

    # param_set = BuddyParamSet(ms1_tol=10, ms2_tol=20,
    #                           halogen=True,
    #                           parallel=False,
    #                           n_cpu=4, timeout_secs=1000)
    # buddy = Buddy(param_set)
    # data = joblib.load('/Users/philip/Documents/projects/msbuddy/ml_b_train/gnps_qtof_mf_ls_cand_2.joblib')
    # buddy.add_data(data)
    # buddy.assign_subformula_annotation()
    # joblib.dump(buddy.data, '/Users/philip/Documents/projects/msbuddy/ml_b_train/gnps_qtof_mf_ls_cand.joblib')
