import numpy as np
import joblib
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from tqdm import tqdm


from base_class.Formula import read_formula
from base_class.ProcessedMS1 import ProcessedMS1
from base_class.ProcessedMS2 import ProcessedMS2
from buddy.ml_candidate_ranking import gen_ml_B_feature_single
from __ms1_iso_algo import sim_ms1_iso_pattern
from base_class.MetaFeature import MetaFeature
from base_class.Spectrum import Spectrum
from buddy.ml_formula_feasibility import predict_formula_feasibility
from main.gen_candidate import gen_candidate_formula
from file_io.init_db import init_db


# This MLP model is trained using NIST20 library.
# 8 models will be generated in total: pos/neg mode, ms1 iso similarity included or not, MS/MS spec included or not.


def load_nist_gen_training_data(path, pos):
    """
    load NIST library (joblib format) and generate training data
    :param path: path to NIST library
    :param pos: True for positive mode, False for negative mode
    """
    db = joblib.load(path)
    # reset index
    db = db.reset_index(drop=True)

    meta_feature_list = []
    gt_formula_list = []
    orbi_list = []
    for i in range(50):  # len(db)
        print(i)
        # parse formula info
        formula = db['Formula'][i]
        gt_form_arr = read_formula(formula)

        # skip if formula is not valid
        if gt_form_arr is None:
            continue
        gt_formula_list.append(gt_form_arr)  # add to ground truth formula list

        # precursor mz
        precursor_mz = float(db['Precursor_mz'][i])

        # mz tolerance, depends on the instrument
        ms1_tol = 5
        ms2_tol = 10
        if db['Instrument_type'][i] == 'Q-TOF':
            ms1_tol = 10
            ms2_tol = 20
            orbi_list.append(False)
        else:
            orbi_list.append(True)

        # simulate ms1 isotope pattern
        ms1_gt_arr, ms1_sim_arr = sim_ms1_iso_pattern(gt_form_arr)
        # create a numpy array of ms1 mz, with length equal to the length of ms1_sim_arr, step size = 1.003355
        ms1_mz_arr = np.array([precursor_mz + x * 1.003355 for x in range(len(ms1_sim_arr))])

        # parse ms2 info
        ms2_mz = np.array([float(x) for x in db['MS2mz'][i].split(';')])
        ms2_int = np.array([float(x) for x in db['MS2int'][i].split(';')])

        mf = MetaFeature(identifier=i,
                         mz=precursor_mz,
                         charge=1 if db['Ion_mode'][i] == 'P' else -1,
                         ms1=Spectrum(ms1_mz_arr, ms1_sim_arr),
                         ms2=Spectrum(ms2_mz, ms2_int))

        # add processed ms1 and ms2
        mf.ms2_processed = ProcessedMS2(mf.mz, mf.ms2_raw, ms2_tol, True,
                                        True, True, 0.01,
                                        0.85, 0.20, 50, False)
        mf.ms1_processed = ProcessedMS1(mf.mz, mf.ms1_raw, mf.adduct.charge, ms1_tol, True, 0.02, 4)

        # generate formula candidates
        gen_candidate_formula(mf, True, ms1_tol, ms2_tol, 1, False,
                              np.array([0] * 12),
                              np.array([80, 150, 10, 15, 20, 10, 0, 20, 0, 30, 10, 15]),
                              4)

        meta_feature_list.append(mf)

    # predict formula feasibility, using ML model A
    predict_formula_feasibility(meta_feature_list)

    # generate ML features for each candidate formula, for ML model B
    # generate feature array
    X_arr = np.array([])
    y_arr = np.array([])

    for cnt, mf in enumerate(meta_feature_list):
        gt_form_arr = gt_formula_list[cnt]
        ms1_tol = 5 if orbi_list[cnt] else 10
        ms2_tol = 10 if orbi_list[cnt] else 20
        if not mf.candidate_formula_list:
            continue
        # generate ML features for each candidate formula
        for cf in mf.candidate_formula_list:
            # get ML features
            ml_feature_arr = gen_ml_B_feature_single(mf, cf, True, ms1_tol, ms2_tol)
            # add to feature array
            if X_arr.size == 0:
                X_arr = ml_feature_arr
                y_arr = np.array([1 if np.array_equal(gt_form_arr, cf.formula.array) else 0])
            else:
                X_arr = np.vstack((X_arr, ml_feature_arr))
                y_arr = np.append(y_arr, 1 if np.array_equal(gt_form_arr, cf.formula.array) else 0)

    print('y_arr sum: ' + str(np.sum(y_arr)))

    # save to joblib file one by one
    mf_ls_name = 'nist_meta_feature_list_' + 'pos' if pos else 'neg'
    gt_ls_name = 'nist_gt_formula_list_' + 'pos' if pos else 'neg'
    X_arr_name = 'nist_X_arr_' + 'pos' if pos else 'neg'
    y_arr_name = 'nist_y_arr_' + 'pos' if pos else 'neg'
    joblib.dump(meta_feature_list, mf_ls_name + '.joblib')
    joblib.dump(gt_formula_list, gt_ls_name + '.joblib')
    joblib.dump(X_arr, X_arr_name + '.joblib')
    joblib.dump(y_arr, y_arr_name + '.joblib')

    return


def train_model(X_arr, y_arr, pos, ms1_iso, ms2_spec):
    """
    train ML model B
    :param X_arr: ML feature array
    :param y_arr: label array
    :param pos: True for positive mode, False for negative mode
    :param ms1_iso: True for ms1 iso similarity included, False for not included
    :param ms2_spec: True for MS/MS spec included, False for not included
    :return: trained model
    """

    if not ms1_iso:
        # discard the first feature in X_arr
        X_arr = X_arr[:, 1:]
    if not ms2_spec:
        # discard the last 9 features in X_arr
        X_arr = X_arr[:, :-9]

    # split training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X_arr, y_arr, test_size=0.2, random_state=0)

    # grid search
    param_grid = {
        'hidden_layer_sizes': [(1024,), (512,), (256,),
                               (1024, 512), (512, 512), (512, 256), (256, 256), (256, 128),
                               (512, 512, 256), (512, 256, 256), (512, 256, 128), (256, 256, 128),
                               (128, 64, 64, 32), (64, 64, 32, 32)],
        'activation': ['relu'],
        'alpha': [1e-6, 1e-5, 1e-4, 1e-3],
        'max_iter': [400, 800]
    }

    # grid search
    mlp = MLPClassifier(random_state=1)
    clf = GridSearchCV(mlp, param_grid, cv=5, n_jobs=-1, scoring='roc_auc', verbose=1)
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
    # train model with best params
    # best_params = {'activation': 'relu', 'alpha': 1e-6, 'hidden_layer_sizes': (64, 32), 'max_iter': 200}
    mlp = MLPClassifier(**best_params).fit(X_train, y_train)
    score = mlp.score(X_test, y_test)  # accuracy on test data
    print("MLP acc.: " + str(score))

    # predict on test data
    y_pred = mlp.predict(X_test)

    # print performance
    print("Classification report for classifier %s:\n%s\n"
          % (mlp, metrics.classification_report(y_test, y_pred)))

    # save model
    model_name = 'model_B_'
    model_name += 'pos' if pos else 'neg'
    model_name += '_ms1' if ms1_iso else '_noms1'
    model_name += '_ms2' if ms2_spec else '_noms2'
    joblib.dump(mlp, model_name + '.joblib')

    return mlp


# test
if __name__ == '__main__':

    # initiate databases
    init_db(1)

    # load training data
    load_nist_gen_training_data('/Users/philip/Documents/projects/pyms2/nist20/nist20_pos.joblib', True)

    # load_nist_gen_training_data('/Users/philip/Documents/projects/pyms2/nist20/nist20_neg.joblib', False)
    #
    # # train models
    # X = joblib.load('nist_X_arr_pos.joblib')
    # y = joblib.load('nist_y_arr_pos.joblib')
    #
    # X = joblib.load('nist_X_arr_neg.joblib')
    # y = joblib.load('nist_y_arr_neg.joblib')
    #
    # # train models
    # train_model(X, y, True, True, True)

    print("Done.")
