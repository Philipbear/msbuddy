import argparse
import json
import sys

import joblib
import numpy as np
from brainpy import isotopic_variants
from imblearn.over_sampling import SMOTE
from scipy.stats import norm
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm

from msbuddy.base import read_formula, MetaFeature, Spectrum, Formula, CandidateFormula
from msbuddy.main import Msbuddy, MsbuddyConfig, _gen_subformula
from msbuddy.load import init_db
from msbuddy.ml import gen_ml_b_feature_single, pred_formula_feasibility
from msbuddy.cand import _calc_ms1_iso_sim
from msbuddy.utils import form_arr_to_str


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


def pred_formula_feasibility_batch(data, top_n, db_mode, shared_data_dict, batch_size):
    n_batch = int(np.ceil(len(data) / batch_size))
    for n in tqdm(range(n_batch)):
        start_idx = n * batch_size
        end_idx = min((n + 1) * batch_size, len(data))
        pred_formula_feasibility(data, start_idx, end_idx, top_n, db_mode, shared_data_dict)

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
        new_data = pred_formula_feasibility_batch(buddy.data, 800, 1, shared_data_dict, 1000)
        joblib.dump(new_data, 'gnps_qtof_mf_ls_cand_2.joblib')
    elif instru == 'orbi':
        orbi_mf_ls = joblib.load('gnps_orbi_mf_ls.joblib')
        # update parameters
        buddy.update_config(MsbuddyConfig(ms1_tol=5, ms2_tol=10, parallel=True, n_cpu=n_cpu,
                                          halogen=True, batch_size=999999,
                                          timeout_secs=timeout_secs))
        buddy.add_data(orbi_mf_ls)
        buddy._preprocess_and_generate_candidate_formula(0, len(buddy.data))
        joblib.dump(buddy.data, 'gnps_orbi_mf_ls_cand_1.joblib')
        new_data = pred_formula_feasibility_batch(buddy.data, 800, 1, shared_data_dict, 1000)
        joblib.dump(new_data, 'gnps_orbi_mf_ls_cand_2.joblib')
    else:  # FT-ICR
        ft_mf_ls = joblib.load('gnps_ft_mf_ls.joblib')
        # update parameters
        buddy.update_config(MsbuddyConfig(ms1_tol=2, ms2_tol=5, parallel=True, n_cpu=n_cpu,
                                          halogen=True, batch_size=999999,
                                          timeout_secs=timeout_secs))
        buddy.add_data(ft_mf_ls)
        buddy._preprocess_and_generate_candidate_formula(0, len(buddy.data))
        joblib.dump(buddy.data, 'gnps_ft_mf_ls_cand_1.joblib')
        new_data = pred_formula_feasibility_batch(buddy.data, 800, 1, shared_data_dict, 1000)
        joblib.dump(new_data, 'gnps_ft_mf_ls_cand_2.joblib')

    return shared_data_dict


def assign_subform_gen_training_data(instru):
    """
    assign subformula annotation and generate training data; reduce memory usage
    """
    # generate ML features for each candidate formula, for ML model B
    # generate feature array
    X_arr = np.array([])
    y_arr = np.array([])

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
        ml_a_prob = buddy.predict_formula_feasibility(gt_form_arr)
        ml_a_prob = 0.5 if ml_a_prob < 0.5 else ml_a_prob
        this_cf = CandidateFormula(Formula(gt_form_arr, 0))
        this_cf.ml_a_prob = ml_a_prob
        cand_form_ls = [this_cf]
        cand_cnt = 0
        for cf in meta_feature.candidate_formula_list:
            if cand_cnt > 100:
                break
            if gt_form_str == form_arr_to_str(cf.formula.array):
                continue
            cand_form_ls.append(cf)
            cand_cnt += 1

        meta_feature.candidate_formula_list = cand_form_ls

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
                ml_feature_arr[7] = np.clip(log_p, -2, 0)

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


def fill_model_a_prob(instru):
    """
    fill in model a prob for ground truth formula
    """
    # generate ML features for each candidate formula, for ML model B
    # generate feature array

    X_name = 'gnps_X_arr_' + instru + '_3k.joblib'
    y_name = 'gnps_y_arr_' + instru + '_3k.joblib'
    X_arr = joblib.load(X_name)
    y_arr = joblib.load(y_name)

    buddy = Msbuddy()

    data_name = 'gnps_' + instru + '_mf_ls_cand_2.joblib'
    data = joblib.load(data_name)
    buddy.add_data(data)
    del data

    gt_name = 'gnps_' + instru + '_gt_ls.joblib'
    gt_ls = joblib.load(gt_name)

    cnt = 0
    for k, meta_feature in enumerate(buddy.data):
        if k < 4000:
            continue
        # print('k: ' + str(k) + ' out of ' + str(len(buddy.data)))
        gt_form_arr = gt_ls[k]
        gt_form_str = form_arr_to_str(gt_form_arr)
        if not meta_feature.candidate_formula_list:
            continue

        # modify the candidate formula list, such that the ground truth formula is the first one
        cand_form_ls = [CandidateFormula(Formula(gt_form_arr, 0))]
        cand_cnt = 0
        for cf in meta_feature.candidate_formula_list:
            if cand_cnt >= 100:
                break
            if gt_form_str == form_arr_to_str(cf.formula.array):
                continue
            cand_form_ls.append(cf)
            cand_cnt += 1

        meta_feature.candidate_formula_list = cand_form_ls

        # generate ML features for each candidate formula
        for n, cf in enumerate(meta_feature.candidate_formula_list):
            if n == 0:
                X_arr[cnt, 2] = buddy.predict_formula_feasibility(gt_form_arr)
                if y_arr[cnt] != 1:
                    raise Exception('y_arr[cnt] != 1')
            cnt += 1

    print('y_arr sum: ' + str(np.sum(y_arr)))
    X_arr_name = 'gnps_X_arr_' + instru + '_3k_filled.joblib'
    joblib.dump(X_arr, X_arr_name)


def combine_and_clean_X_y():
    # load training data
    X_arr_qtof = joblib.load('gnps_X_arr_qtof.joblib')
    y_arr_qtof = joblib.load('gnps_y_arr_qtof.joblib')
    X_arr_orbi = joblib.load('gnps_X_arr_orbi.joblib')
    y_arr_orbi = joblib.load('gnps_y_arr_orbi.joblib')
    X_arr_ft = joblib.load('gnps_X_arr_ft.joblib')
    y_arr_ft = joblib.load('gnps_y_arr_ft.joblib')

    X_arr = np.vstack((X_arr_qtof, X_arr_orbi, X_arr_ft))
    y_arr = np.append(y_arr_qtof, np.append(y_arr_orbi, y_arr_ft))
    print('X_arr shape: ' + str(X_arr.shape))
    print('y_arr shape: ' + str(y_arr.shape))

    # remove rows with nan
    X_arr = np.where(X_arr == None, np.nan, X_arr)
    X_arr = X_arr.astype(np.float64)
    nan_rows = np.any(np.isnan(X_arr), axis=1)
    # Remove those rows from X_arr and y_arr
    X_arr = X_arr[~nan_rows]
    y_arr = y_arr[~nan_rows]
    print('X_arr shape: ' + str(X_arr.shape))
    print('y_arr shape: ' + str(y_arr.shape))

    # save to joblib file
    joblib.dump(X_arr, 'gnps_X_arr.joblib')
    joblib.dump(y_arr, 'gnps_y_arr.joblib')


def z_norm_smote():
    """
    z-normalization of X_arr
    """
    X_arr = joblib.load('gnps_X_arr.joblib')
    # z-normalization
    X_mean = np.mean(X_arr[:, 5:], axis=0)
    X_std = np.std(X_arr[:, 5:], axis=0)
    X_arr[:, 5:] = (X_arr[:, 5:] - X_mean) / X_std

    joblib.dump(X_arr, 'gnps_X_arr_z_norm.joblib')
    joblib.dump(X_mean, 'ml_b_mean_arr.joblib')
    joblib.dump(X_std, 'ml_b_std_arr.joblib')

    y_arr = joblib.load('gnps_y_arr.joblib')
    smote = SMOTE(random_state=0)
    X_arr, y_arr = smote.fit_resample(X_arr, y_arr)

    joblib.dump(X_arr, 'gnps_X_arr_SMOTE.joblib')
    joblib.dump(y_arr, 'gnps_y_arr_SMOTE.joblib')


def train_model(ms1_iso, ms2_spec, pswd, n_cpu):
    """
    train ML model B
    :return: trained model
    """
    # load training data
    X_arr = joblib.load('gnps_X_arr_SMOTE.joblib')
    y_arr = joblib.load('gnps_y_arr_SMOTE.joblib')

    print("Training model ...")
    if not ms1_iso:
        # discard the 5th feature in X_arr
        X_arr = np.delete(X_arr, 5, axis=1)
    if not ms2_spec:
        # discard the last 14 features in X_arr
        X_arr = X_arr[:, :-14]

    print('splitting...')
    # split training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X_arr, y_arr, test_size=0.2, random_state=0)

    # print('grid search...')
    # # grid search
    # all_param_grid = {
    #     'hidden_layer_sizes': [
    #         (512, 512, 256, 256), (1024, 512, 512, 256), (1024, 1024, 512, 512),
    #     ],
    #     'max_iter': [800]
    # }
    #
    # # grid search
    # mlp = MLPClassifier(random_state=1)
    # clf = GridSearchCV(mlp, all_param_grid, cv=3, n_jobs=n_cpu, scoring='accuracy', verbose=1)
    # clf.fit(X_train, y_train)
    #
    # # print best parameters
    # print("Best parameters set found on development set:")
    # print(clf.best_params_)
    # email_body = "Best parameters set found on development set:\n"
    # email_body += str(clf.best_params_)
    #
    # best_params = clf.best_params_
    #
    # print("Grid scores on development set:")
    # email_body += "\nGrid scores on development set:\n"
    #
    # means = clf.cv_results_['mean_test_score']
    # stds = clf.cv_results_['std_test_score']
    #
    # for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    #     print("%0.5f (+/-%0.05f) for %r"
    #           % (mean, std * 2, params))
    #     email_body += "%0.5f (+/-%0.05f) for %r\n" % (mean, std * 2, params)
    #
    # # send email
    # send_hotmail_email("Grid search finished", email_body, "s1xing@health.ucsd.edu",
    #                    smtp_password=pswd)
    #

    #
    # # best parameters
    best_params = {'hidden_layer_sizes': (1024, 1024, 512, 512), 'max_iter': 800}
    #
    print("train model...")

    # train model with best params for 3 times, save all
    best_score = 0
    for m in range(3):
        this_mlp, score = _train(m, X_train, y_train, X_test, y_test, best_params)
        if score > best_score:
            best_score = score
            print("MLP acc.: " + str(score))
            model_name = 'model_b'
            model_name += '_ms1' if ms1_iso else '_noms1'
            model_name += '_ms2_' if ms2_spec else '_noms2_'
            model_name += str(m)
            joblib.dump(this_mlp, model_name + '.joblib')
            body_str = "model " + str(m) + "  MLP acc.: " + str(score) + "\n"
            body_str += 'ms1_' if ms1_iso else 'noms1_'
            body_str += 'ms2' if ms2_spec else 'noms2'
            send_hotmail_email("mlp trained", body_str,
                               "s1xing@health.ucsd.edu", smtp_password=pswd)

    #
    # # top 3 models with highest accuracy
    # top3_idx = np.argsort(scores)[-3:]
    # top3_scores = np.array(scores)[top3_idx]
    # top3_mlps = np.array(mlps)[top3_idx]
    #
    # # save top 3 models
    # for i in range(3):
    #     best_mlp = top3_mlps[i]
    #     score = top3_scores[i]
    #     print("MLP acc.: " + str(score))
    #     model_name = 'model_b'
    #     model_name += '_ms1' if ms1_iso else '_noms1'
    #     model_name += '_ms2' if ms2_spec else '_noms2'
    #     model_name += '_' + str(i)
    #     joblib.dump(best_mlp, model_name + '.joblib')

    return


def _train(random_state, X_train, y_train, X_test, y_test, best_params):
    mlp_cls = MLPClassifier(random_state=random_state, **best_params)
    mlp_cls.fit(X_train, y_train)
    mlp_score = mlp_cls.score(X_test, y_test)
    return mlp_cls, mlp_score


import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


def send_hotmail_email(subject, body, to_email, smtp_server='smtp-mail.outlook.com', smtp_port=587,
                       smtp_username='philipxsp@hotmail.com', smtp_password='Xsp123456'):
    """
    Send an email using Hotmail's SMTP.

    Parameters:
        subject: The subject of the email.
        body: The body text of the email.
        to_email: The recipient's email address.
        smtp_server: The SMTP server to use.
        smtp_port: The SMTP port to use.
        smtp_username: The SMTP username (usually the email sender's address).
        smtp_password: The SMTP password.
    """

    # Create the message
    msg = MIMEMultipart()
    msg['From'] = smtp_username
    msg['To'] = to_email
    msg['Subject'] = subject

    # Attach the body text
    msg.attach(MIMEText(body, 'plain'))

    try:
        # Set up the server
        server = smtplib.SMTP(host=smtp_server, port=smtp_port)
        server.starttls()

        # Log into the server
        server.login(smtp_username, smtp_password)

        # Send the email
        server.sendmail(smtp_username, to_email, msg.as_string())

        # Terminate the SMTP session and close the connection
        server.quit()

        print(f"Email sent successfully to {to_email}.")

    except:
        print("Failed to send email.")


def parse_args():
    """
    parse command line arguments
    :return: parsed arguments
    """
    parser = argparse.ArgumentParser(description='ML model B training')
    parser.add_argument('-calc', action='store_true', help='calculate gnps data')
    parser.add_argument('-gen', action='store_true', help='generate training data')
    parser.add_argument('-ms', type=str, help='instrument type')
    parser.add_argument('-cpu', type=int, default=16, help='number of CPU cores to use')
    parser.add_argument('-to', type=int, default=600, help='timeout in seconds')
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
    # args = argparse.Namespace(calc=True, cpu=1, ms1=False, ms2=True)

    # load training data
    # load_gnps_data('gnps_ms2db_preprocessed_20231005.joblib')

    if args.calc:
        # calc_gnps_data(args.cpu, args.to, args.ms)
        assign_subform_gen_training_data(instru=args.ms)
        send_hotmail_email("job finished", "assign_subform and gen_training_data finished.",
                           "s1xing@health.ucsd.edu", smtp_password=args.p)

    elif args.gen:
        combine_and_clean_X_y()
        z_norm_smote()  # z-normalization and SMOTE

    else:  # train model
        train_model(args.ms1, args.ms2, args.p, args.cpu)

    print("--- %s seconds ---" % (time.time() - start_time))

    time_elapsed = time.time() - start_time
    time_elapsed = time_elapsed / 3600

    send_hotmail_email("Server job finished", "Job finished in " + str(time_elapsed) + " hrs",
                       "s1xing@health.ucsd.edu", smtp_password=args.p)
