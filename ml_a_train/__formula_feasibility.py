from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
import pandas as pd
import numpy as np
import joblib
from numba import njit


# This MLP model is for predicting the feasibility of a molecular formula
# The input is a molecular formula, and the output is a binary value indicating whether the formula is feasible
# for training data, positive examples are formulas with PubChem >= 1, other databases >= 1
# negative examples are randomly generated formulas


def data_process(file) -> int:
    """
    process formula db
    """
    db = pd.read_csv(file)
    print("db original size: " + str(db.shape[0]))

    # only reserve formula entries with CHNOPSFClBrI
    db = db.loc[(db['b'] + db['k'] + db['na'] + db['se'] + db['si']) == 0, :]
    # c > 0
    db = db.loc[db['c'] > 0, :]
    # dbe >= 0
    db['dbe'] = db['c'] + 1 - (db['h'] + db['f'] + db['cl'] + db['br'] + db['i']) / 2 + (db['n'] + db['p']) / 2
    db = db.loc[db['dbe'] >= 0, :]
    # SENIOR rules
    bool_arr = _senior_rules(db)
    db = db.loc[bool_arr, :]

    print("db size: " + str(db.shape[0]))

    # mass
    db['mass'] = 12 * db['c'] + 1.007825 * db['h'] + 78.918336 * db['br'] + 34.968853 * db['cl'] + 18.998403 * db['f'] + \
                 126.904473 * db['i'] + 14.003074 * db['n'] + 15.994915 * db['o'] + 30.973762 * db['p'] + 31.972071 * \
                 db['s']

    # pubchem > 0
    filter_arr1 = db['PubChem'] > 0
    db = db.loc[filter_arr1, :]
    db.reset_index(drop=True, inplace=True)
    print("db size after pubchem filter: " + str(db.shape[0]))

    # filter_arr 2, where mass > 100
    filter_arr2 = db['mass'] > 100
    db = db.loc[filter_arr2, :]
    db.reset_index(drop=True, inplace=True)
    print("db size after mass filter: " + str(db.shape[0]))

    # other db existence
    db_sum = np.zeros(db.shape[0])
    for i in range(26):
        col_v = np.array(db.iloc[:, i + 18])
        col_v = col_v > 0
        col_v = col_v.astype(int)
        db_sum += col_v
    # filter_arr 3, where db_sum >= 1
    filter_arr3 = db_sum >= 1
    db = db.loc[filter_arr3, :]
    print("db size after db_sum filter: " + str(db.shape[0]))

    # drop the original columns
    db.drop(columns=['formula_str', 'b', 'k', 'na', 'se', 'si', 'dbe',
                     'ANPDB', 'BLEXP', 'BMDB', 'ChEBI', 'COCONUT', 'DrugBank', 'DSSTOX', 'ECMDB', 'FooDB', 'HMDB',
                     'HSDB', 'KEGG', 'LMSD', 'MaConDa', 'MarkerDB', 'MCDB', 'NORMAN', 'NPASS', 'NPAtlas', 'Plantcyc',
                     'SMPDB', 'STOFF_IDENT', 'T3DB', 'TTD', 'UNPD', 'YMDB', 'PubChem'], inplace=True)

    # write to csv
    db.to_csv("formula_data.csv", index=False)
    return db.shape[0]


def data_analysis(file):
    """
    data analysis of positive training data
    """
    db = pd.read_csv(file)

    # for H
    mean_h_c = np.mean(db['h'] / db['c'])
    std_h_c = np.std(db['h'] / db['c'])

    # br,cl,f,i,n,o,p,s
    mean_ls = []

    for i in range(3, 11):
        col = db.iloc[:, i]
        col = np.array(col)
        mean_ls.append(np.mean(col))

    return mean_h_c, std_h_c, mean_ls


def gen_neg_sample(file, mean_h_c, std_h_c, mean_ls, h_std_factor=4, factor=1, write_file=True, switch_cnt=30):
    """
    generate negative training data, cnt = pos_cnt
    generate random formulas with ratio of each element to carbon following the normal distribution
    mean & std from data_analysis()
    random shuffle element counts within each formula

    generated formulas should pass the following rules:
    1. dbe >= 0
    2. SENIOR rules
    :param factor: factor for poisson distribution
    """
    tmp = pd.read_csv(file)
    # sort by c, ascending
    tmp.sort_values(by=['c'], inplace=True)
    db = tmp.copy()

    # generate random formulas
    # for H  -- normal distribution
    col = np.random.normal(mean_h_c, std_h_c * h_std_factor, db.shape[0])
    col = np.array(col * db['c']).astype(int)
    col = np.clip(col, 0, None)
    db.iloc[:, 2] = col
    # for br,cl,f,i,n,o,p,s   -- poisson distribution
    for i in range(3, 11):
        col = np.random.poisson(mean_ls[i - 3] * factor, db.shape[0])
        col = np.array(col).astype(int)
        col = np.clip(col, 0, None)
        # if non-H, some elements are randomly set to 0
        col = np.where(np.random.choice([True, False], db.shape[0], replace=True), col, 0)
        db.iloc[:, i] = col

    # random switch element counts within each formula
    for i in range(switch_cnt):
        # rows to switch, a random bool array
        rows_shuffle = np.random.choice([True, False], db.shape[0], replace=True)
        # column pair to switch
        col_pair = np.random.choice(range(3, 11), 2, replace=False)
        # switch
        col_0_arr = db.iloc[rows_shuffle, col_pair[0]].copy()
        col_1_arr = db.iloc[rows_shuffle, col_pair[1]].copy()
        db.iloc[rows_shuffle, col_pair[0]] = col_1_arr
        db.iloc[rows_shuffle, col_pair[1]] = col_0_arr

    # whether the formula pass all the rules
    unique_bool = _exist_check(np.array(db.iloc[:, 1:]), np.array(tmp.iloc[:, 1:]), None)
    # SENIOR rules
    bool_arr = _senior_rules(db)
    # drop formulas that do not pass dbe >= 0
    dbe_arr = db['c'] + 1 - (db['h'] + db['f'] + db['cl'] + db['br'] + db['i']) / 2 + (db['n'] + db['p']) / 2
    # o >= 3*p
    o_p_bool = db.apply(lambda row: row['o'] >= 3 * row['p'] if row['p'] > 0 else True, axis=1)

    bool_arr = bool_arr & (dbe_arr >= 0)
    bool_arr = bool_arr & o_p_bool
    bool_arr = bool_arr & unique_bool

    # generate negative samples until all generated formulas pass the rules
    while not all(bool_arr):
        print("remaining: " + str(sum(~bool_arr)))

        # random sampling
        # for H  -- normal distribution
        col = np.random.normal(mean_h_c, std_h_c * h_std_factor, db.shape[0])
        col = np.array(col * db['c']).astype(int)
        col = np.clip(col, 0, None)
        db.iloc[~np.array(bool_arr), 2] = col[~bool_arr]
        # for br,cl,f,i,n,o,p,s   -- poisson distribution
        for i in range(3, 11):
            col = np.random.poisson(mean_ls[i - 3] * factor, db.shape[0])
            col = np.array(col).astype(int)
            col = np.clip(col, 0, None)
            # if non-H, some elements are randomly set to 0
            col = np.where(np.random.choice([True, False], db.shape[0], replace=True), col, 0)
            db.loc[~bool_arr, db.columns[i]] = col[~bool_arr]

        # random switch element counts within each formula
        for i in range(switch_cnt):
            # rows to switch, a random bool array
            rows_shuffle = np.random.choice([True, False], db.shape[0], replace=True)
            # only switch the rows that do not pass the rules
            rows_shuffle = rows_shuffle & ~bool_arr
            rows_shuffle = np.array(rows_shuffle)
            # column pair to switch
            col_pair = np.random.choice(range(3, 11), 2, replace=False)
            # switch
            col_0_arr = db.iloc[rows_shuffle, col_pair[0]].copy()
            col_1_arr = db.iloc[rows_shuffle, col_pair[1]].copy()
            db.iloc[rows_shuffle, col_pair[0]] = col_1_arr
            db.iloc[rows_shuffle, col_pair[1]] = col_0_arr


        # newly generated formulas do not overlap with the original formulas, np.array.equal
        unique_bool = _exist_check(np.array(db.iloc[:, 1:]), np.array(tmp.iloc[:, 1:]), np.array(bool_arr))

        # SENIOR rules
        bool_arr = _senior_rules(db)
        # drop formulas that do not pass dbe >= 0
        dbe_arr = db['c'] + 1 - (db['h'] + db['f'] + db['cl'] + db['br'] + db['i']) / 2 + (db['n'] + db['p']) / 2
        # o >= 3*p
        o_p_bool = db.apply(lambda row: row['o'] >= 3 * row['p'] if row['p'] > 0 else True, axis=1)

        bool_arr = bool_arr & (dbe_arr >= 0)
        bool_arr = bool_arr & o_p_bool
        bool_arr = bool_arr & unique_bool

    if write_file:
        db.to_csv("formula_data_neg.csv", index=False)
    print("negative sample size: " + str(db.shape[0]))
    return db


@njit
def _exist_check(db, db_ori, bool_arr) -> np.array:
    """
    check whether the generated formulas overlap with the original formulas;
    check whether the generated formulas overlap with each other
    :param db: newly generated formulas
    :param db_ori: original formulas
    :param bool_arr: do not have to perform the check for these formulas
    :return: boolean array, True if the formula does not overlap with the original formulas, False otherwise
    """
    out = np.array([True] * db.shape[0])
    if bool_arr is None:
        for i in range(db.shape[0]):
            for j in range(db_ori.shape[0]):
                if db_ori[j, 0] != db[i, 0]:
                    continue
                if db_ori[j, 0] > db[i, 0]:
                    break
                if (db[i, :] == db_ori[j, :]).all():
                    out[i] = False
                    break
        for i in range(db.shape[0] - 1):
            if not out[i]:
                continue
            for j in range(i + 1, db.shape[0]):
                if db[i, 0] != db[j, 0]:
                    continue
                if db[j, 0] > db[i, 0]:
                    break
                if (db[i, :] == db[j, :]).all():
                    out[i] = False
                    break
    else:
        for i in range(db.shape[0]):
            if bool_arr[i]:
                continue
            for j in range(db_ori.shape[0]):
                if db_ori[j, 0] != db[i, 0]:
                    continue
                if db_ori[j, 0] > db[i, 0]:
                    break
                if (db[i, :] == db_ori[j, :]).all():
                    out[i] = False
                    break
        for i in range(db.shape[0] - 1):
            if bool_arr[i]:
                continue
            if not out[i]:
                continue
            for j in range(i + 1, db.shape[0]):
                if db[i, 0] != db[j, 0]:
                    continue
                if db[j, 0] > db[i, 0]:
                    break
                if (db[i, :] == db[j, :]).all():
                    out[i] = False
                    break
    return out

def _senior_rules(form):
    """
    SENIOR rules
    :param form: pandas dataframe, each row is a formula
    :return: boolean list, True if the formula passes the rules, False otherwise
    """
    # ["C", "H", "Br", "Cl", "F", "I", "K", "N", "Na", "O", "P", "S"]
    # int senior_1_1 = 6 * s + 5 * p + 4 * c + 3 * n + 2 * o + h + f + cl + br + i + na + k
    # int senior_1_2 = p + n + h + f + cl + br + i + na + k
    # int senior_2 = c + h + n + o + p + f + cl + br + i + s + na + k

    senior_1_1 = 6 * form['s'] + 5 * form['p'] + 4 * form['c'] + 3 * form['n'] + 2 * form['o'] + form['h'] + \
                 form['f'] + form['cl'] + form['br'] + form['i']
    senior_1_2 = form['p'] + form['n'] + form['h'] + form['f'] + form['cl'] + form['br'] + form['i']
    senior_2 = form['c'] + form['h'] + form['n'] + form['o'] + form['p'] + form['f'] + form['cl'] + \
               form['br'] + form['i'] + form['s']

    out_bool = [True] * form.shape[0]

    # The sum of valences or the total number of atoms having odd valences is even
    out_bool = out_bool & (senior_1_1 % 2 == 0) & (senior_1_2 % 2 == 0)

    # The sum of valences is greater than or equal to twice the number of atoms minus 1
    out_bool = out_bool & (senior_1_1 >= 2 * senior_2 - 1)

    return out_bool


def gen_ml_matrix(file_pos, file_neg):
    """
    generate training data from processed formula data
    """
    # feature design
    # c, h, n, o, p, s, hal, dbe, dbe/mass, total atom/mass, h/c, n/c, o/c, p/c, s/c, hal/c, c/ta

    db_pos = pd.read_csv(file_pos)
    db_neg = pd.read_csv(file_neg)

    # concatenate pos & neg
    db = pd.concat([db_pos, db_neg], axis=0)

    db['dbe'] = db['c'] + 1 - (db['h'] + db['f'] + db['cl'] + db['br'] + db['i']) / 2 + (db['n'] + db['p']) / 2
    db['mass'] = 12 * db['c'] + 1.007825 * db['h'] + 78.918336 * db['br'] + 34.968853 * db['cl'] + 18.998403 * db['f'] + \
                 126.904473 * db['i'] + 14.003074 * db['n'] + 15.994915 * db['o'] + 30.973762 * db['p'] + 31.972071 * \
                 db['s']
    # halogen
    db['hal'] = db['f'] + db['cl'] + db['br'] + db['i']
    # total atom
    db['ta'] = db['c'] + db['h'] + db['hal'] + db['n'] + db['o'] + db['p'] + db['s']

    db['n_exist'] = db['n'].apply(lambda x: 1 if x > 0 else 0)
    db['f_exist'] = db['f'].apply(lambda x: 1 if x > 0 else 0)
    db['cl_exist'] = db['cl'].apply(lambda x: 1 if x > 0 else 0)
    db['br_exist'] = db['br'].apply(lambda x: 1 if x > 0 else 0)
    db['i_exist'] = db['i'].apply(lambda x: 1 if x > 0 else 0)
    db['hal_ele_type'] = db['f_exist'] + db['cl_exist'] + db['br_exist'] + db['i_exist']

    # co-occurrence of halogen
    db['hal_two'] = db['hal_ele_type'].apply(lambda x: 1 if x >= 2 else 0)
    db['hal_three'] = db['hal_ele_type'].apply(lambda x: 1 if x >= 3 else 0)

    db['senior_1_1'] = 6 * db['s'] + 5 * db['p'] + 4 * db['c'] + 3 * db['n'] + 2 * db['o'] + db['h'] + db['hal']
    db['senior_1_2'] = db['p'] + db['n'] + db['h'] + db['hal']
    db['2ta-1'] = 2 * db['ta'] - 1
    # db['3o'] = 3 * db['o']

    db['dbe_0'] = db['dbe'].apply(lambda x: 0 if x == 0 else 1)
    db['dbe_mass_1'] = np.sqrt(db['dbe'] / db['mass'])
    db['dbe_mass_2'] = db['dbe'] / np.power((db['mass'] / 100), 2 / 3)
    # db['ta_mass'] = db['ta'] / db['mass']
    db['h_c'] = db['h'] / db['c']
    db['n_c'] = db['n'] / db['c']
    db['o_c'] = db['o'] / db['c']
    db['p_c'] = db['p'] / db['c']
    db['s_c'] = db['s'] / db['c']
    db['hal_c'] = db['hal'] / db['c']
    # hal to h, 0 if h = 0
    db['hal_h'] = db['hal'] / db['h']
    # fill inf with 0
    db['hal_h'] = db['hal_h'].replace([float('inf'), float('-inf')], 0)
    # fill nan with 0
    db['hal_h'] = db['hal_h'].fillna(0)

    # o to p
    db['o_p'] = db['o'] / db['p']
    # fill inf with 3
    db['o_p'] = db['o_p'].replace([float('inf'), float('-inf')], 3)
    # fill nan with 3
    db['o_p'] = db['o_p'].fillna(3)
    # clip
    db['o_p'] = db['o_p'].clip(0, 3)
    db['o_p'] = db['o_p'] / 3

    db['c_ta'] = db['c'] / db['ta']
    db['h_ta'] = db['h'] / db['ta']
    db['n_ta'] = db['n'] / db['ta']
    db['o_ta'] = db['o'] / db['ta']
    db['p_ta'] = db['p'] / db['ta']
    db['s_ta'] = db['s'] / db['ta']
    db['hal_ta'] = db['hal'] / db['ta']

    out = db.loc[:, ['c', 'h', 'n', 'o', 'p', 's', 'hal', 'ta',
                     'c_ta', 'h_ta', 'n_ta', 'o_ta', 'p_ta', 's_ta', 'hal_ta',
                     'senior_1_1', 'senior_1_2', '2ta-1',
                     'dbe', 'dbe_mass_1', 'dbe_mass_2',
                     'h_c', 'n_c', 'o_c', 'p_c', 's_c', 'hal_c', 'hal_h', 'o_p',
                     'hal_two', 'hal_three']]
    print("training X size: " + str(out.shape[0]))

    # # feature normalization using z-score, save mean and std
    # mean_arr = np.array([])
    # std_arr = np.array([])
    # for col in out.columns:
    #     mean = out[col].mean()
    #     std = out[col].std()
    #     mean_arr = np.append(mean_arr, mean)
    #     std_arr = np.append(std_arr, std)
    #     out[col] = (out[col] - mean) / std
    # joblib.dump(mean_arr, "mean_arr.joblib")
    # joblib.dump(std_arr, "std_arr.joblib")

    out.to_csv("formula_training_X.csv", index=False)


def train_cls_model(X_file):
    """
    classification model training
    """
    X = pd.read_csv(X_file)

    # first half is pos, second half is neg
    y = np.array([1] * int(X.shape[0] / 2))
    y = np.append(y, [0] * int(X.shape[0] / 2))

    print("train test split")
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        train_size=0.8,
                                                        random_state=2)

    print("cross validation & grid search...")
    # grid search
    # param_grid = {'hidden_layer_sizes': [(256, ), (128, ),
    #                                          (128, 128), (128, 64), (64, 64), (64, 32),
    #                                          (64, 64, 32), (64, 32, 32), (32, 32, 16), (32, 16, 16), (16, 16, 8),
    #                                          (32, 16, 8, 8), (16, 16, 8, 8)],
    #                   'alpha': [0.000001, 0.00001, 0.0001, 0.001],
    #                   'activation': ['relu'], # optimized, ['relu', 'tanh']
    #                   'max_iter': [200, 500]} # optimized, [100, 200]

    param_grid = {'hidden_layer_sizes': [(64, 32)],
                  'alpha': [1e-6],
                  'activation': ['relu'],
                  'max_iter': [200]}

    mlp = MLPClassifier(random_state=1)
    clf = GridSearchCV(mlp, param_grid, cv=5, n_jobs=-1, scoring='f1', verbose=1)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print(clf.best_params_)  # {'activation': 'relu', 'alpha': 1e-6, 'hidden_layer_sizes': (64, 32), 'max_iter': 200}
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
    # train model with best params for multiple times, and choose the best one
    best_score = 0
    for i in range(5):
        print("train model " + str(i) + "...")
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

    score = best_mlp.score(X_test, y_test)
    print("MLP acc.: " + str(score))

    # save model
    joblib.dump(best_mlp, "model_a.joblib")


def predict_prob(X_file):
    """
    predict probability
    """
    X = pd.read_csv(X_file)
    model = joblib.load("model_a.joblib")
    prob = model.predict_proba(X)
    return prob[:, 1]


# test
if __name__ == '__main__':
    # data_process("../data_prepare/formulaDB_20230316.csv")
    # #
    # mean_h_c, std_h_c, mean_ls = data_analysis("formula_data.csv")
    # neg = gen_neg_sample("formula_data.csv", mean_h_c, std_h_c, mean_ls, write_file=True)
    # # #
    # pos = pd.read_csv("formula_data.csv")
    # pos['mass'] = 12*pos['c']+1.007825*pos['h']+78.918336*pos['br']+34.968853*pos['cl']+18.998403*pos['f']+\
    #              126.904473*pos['i']+14.003074*pos['n']+15.994915*pos['o']+30.973762*pos['p']+31.972071*pos['s']
    # pos['ta'] = pos['c'] + pos['h'] + pos['n'] + pos['o'] + pos['p'] + pos['s'] + pos['f'] + pos['cl'] + pos['br'] + pos['i']
    # neg = pd.read_csv("formula_data_neg.csv")
    # neg['mass'] = 12*neg['c']+1.007825*neg['h']+78.918336*neg['br']+34.968853*neg['cl']+18.998403*neg['f']+\
    #                 126.904473*neg['i']+14.003074*neg['n']+15.994915*neg['o']+30.973762*neg['p']+31.972071*neg['s']
    # neg['ta'] = neg['c'] + neg['h'] + neg['n'] + neg['o'] + neg['p'] + neg['s'] + neg['f'] + neg['cl'] + neg['br'] + neg['i']
    # print("pos mass mean: " + str(pos['mass'].mean()))
    # print("neg mass mean: " + str(neg['mass'].mean()))
    # print("pos mass std: " + str(pos['mass'].std()))
    # print("neg mass std: " + str(neg['mass'].std()))
    #
    # print("pos ta mean: " + str(pos['ta'].mean()))
    # print("neg ta mean: " + str(neg['ta'].mean()))
    # print("pos ta std: " + str(pos['ta'].std()))
    # print("neg ta std: " + str(neg['ta'].std()))

    # train
    gen_ml_matrix("formula_data.csv", "formula_data_neg.csv")
    # 421222 samples in total, 210611 pos, 210611 neg
    # train_cls_model("formula_training_X.csv")

    # predict
    # prob_arr = predict_prob("../ml_a/formula_training_X.csv")
    # print(prob_arr)

    # # # feature importance
    # X = pd.read_csv("../ml_a/formula_training_X.csv")
    # y = np.array([1] * int(X.shape[0] / 2))
    # y = np.append(y, [0] * int(X.shape[0] / 2))
    # from sklearn.inspection import permutation_importance
    # model = joblib.load( "../ml_a/model_a.joblib")
    # fi_result = permutation_importance(model, X, y, n_repeats=10, random_state=1, n_jobs=3)
    # print(fi_result.importances_mean)
    # joblib.dump(fi_result, "../ml_a/formula_feature_importance.joblib")
    #
    # fi_result = joblib.load("../ml_a/formula_feature_importance.joblib")
    # # print out the feature importance in descending order
    # for i in np.argsort(fi_result.importances_mean)[::-1]:
    #     print(f"{X.columns[i]:<8} "
    #           f"{fi_result.importances_mean[i]:.4f}"
    #           f" +/- {fi_result.importances_std[i]:.4f}")

    print('Done')
