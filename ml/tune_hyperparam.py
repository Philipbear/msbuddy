
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import ndcg_score
import joblib
from itertools import product
import concurrent.futures


def _train_test_split(X_arr, y_arr, group_arr, test_size=0.1, random_state=1):
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


def perform_cross_validation(X, y, group_sizes, params, n_splits=5):
    """
    Perform cross-validation.
    """
    X_1, X_test, y_1, y_test, groups_1, groups_test = _train_test_split(X, y, group_sizes, test_size=0.1)

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


def perform_cross_validation_parallel(X, y, groups, params, n_splits):
    # Assuming your original 'perform_cross_validation' function is defined elsewhere
    val_score, test_score = perform_cross_validation(X, y, groups, params, n_splits)
    return params, val_score, test_score


def grid_search_cv_parallel(X, y, groups, param_grid, n_splits=5):
    df_rows = []

    # Generate all combinations of hyperparameters
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in product(*values)]

    # Perform parallel cross-validation
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(perform_cross_validation_parallel, X, y, groups, params, n_splits)
                   for params in param_combinations]

        for future in concurrent.futures.as_completed(futures):
            params, val_score, test_score = future.result()
            row = [params.get(k) for k in keys] + [val_score, test_score]
            df_rows.append(row)
            print(f"Tested params: {params}, test NDCG@1 Score: {test_score}, val NDCG@1 Score: {val_score}")

    df_columns = list(keys) + ['val_score', 'test_score']
    df = pd.DataFrame(df_rows, columns=df_columns)
    return df


def grid_search_cv(X, y, groups, param_grid, n_splits=5):

    df_rows = []

    # Generate all combinations of hyperparameters
    keys, values = zip(*param_grid.items())
    for param_values in product(*values):
        params = dict(zip(keys, param_values))

        # Perform cross-validation
        val_score, test_score = perform_cross_validation(X, y, groups, params, n_splits=n_splits)

        # Save results, param_values, val_score, test_score
        df_rows.append([params['objective'], params['metric'], params['ndcg_at'], params['learning_rate'],
                        params['num_leaves'], params['max_depth'], params['min_data_in_leaf'], params['max_bin'],
                        params['bagging_fraction'], params['bagging_freq'], params['feature_fraction'],
                        params['lambda_l1'], params['lambda_l2'], params['seed'], params['verbose'],
                        val_score, test_score])

        print(f"Tested params: {params}, test NDCG@1 Score: {test_score}, val NDCG@1 Score: {val_score}")

    df = pd.DataFrame(df_rows, columns=['objective', 'metric', 'ndcg_at', 'learning_rate', 'num_leaves',
                                        'max_depth', 'min_data_in_leaf', 'max_bin', 'bagging_fraction',
                                        'bagging_freq', 'feature_fraction', 'lambda_l1', 'lambda_l2',
                                        'seed', 'verbose', 'val_score', 'test_score'])
    return df


def grid_search_single(X, y, groups, param_grid, n_splits=5):

    df_rows = []

    # Generate all combinations of hyperparameters
    keys, values = zip(*param_grid.items())
    for param_values in product(*values):
        params = dict(zip(keys, param_values))

        # Split training and testing data
        (X_train, X_val, X_test, y_train, y_val, y_test,
         groups_train, groups_val, groups_test) = _train_val_test_split(X, y, groups,
                                                                        val_size=0.1, test_size=0.1, random_state=24)

        # Create LightGBM datasets
        train_data = lgb.Dataset(data=X_train, label=y_train, group=groups_train)
        val_data = lgb.Dataset(data=X_val, label=y_val, group=groups_val)

        # Train the model
        gbm = lgb.train(params, train_data, valid_sets=[val_data], num_boost_round=1500,
                        callbacks=[lgb.early_stopping(stopping_rounds=30)])

        # Predict on test data
        test_preds = gbm.predict(X_test)

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
        test_score = np.mean(ndcg_scores)
        val_score = gbm.best_score['valid_0']['ndcg@1']

        # Save results, param_values, val_score, test_score
        df_rows.append([params['objective'], params['metric'], params['ndcg_at'], params['learning_rate'],
                        params['num_leaves'], params['max_depth'], params['min_data_in_leaf'], params['max_bin'],
                        params['bagging_fraction'], params['bagging_freq'], params['feature_fraction'],
                        params['lambda_l1'], params['lambda_l2'], params['seed'], params['verbose'],
                        val_score, test_score])

        print(f"Tested params: {params}, test NDCG@1 Score: {test_score}, val NDCG@1 Score: {val_score}")

    df = pd.DataFrame(df_rows, columns=['objective', 'metric', 'ndcg_at', 'learning_rate', 'num_leaves',
                                        'max_depth', 'min_data_in_leaf', 'max_bin', 'bagging_fraction',
                                        'bagging_freq', 'feature_fraction', 'lambda_l1', 'lambda_l2',
                                        'seed', 'verbose', 'val_score', 'test_score'])
    return df


def tune_hyperparams(ms1_iso, ms2_spec):
    group_arr = joblib.load('gnps_group_arr.joblib')[-40000:]
    total_cnt = int(np.sum(group_arr))

    # load training data
    X_arr = joblib.load('gnps_X_arr_filled.joblib')[-total_cnt:, :]
    y_arr = joblib.load('gnps_y_arr.joblib')[-total_cnt:]

    # # load training data
    # X_arr = joblib.load('gnps_X_arr_filled.joblib')
    # y_arr = joblib.load('gnps_y_arr.joblib')
    # group_arr = joblib.load('gnps_group_arr.joblib')

    # group arr as int
    group_arr = group_arr.astype(np.int32)
    assert np.sum(group_arr) == len(X_arr) == len(y_arr)

    if not ms1_iso:
        # discard the ms1 iso feature in X_arr
        X_arr = X_arr[:, 1:]
    if not ms2_spec:
        # discard the last 24 features in X_arr
        X_arr = X_arr[:, :-24]

    # Parameters for training
    param_grid = {
        'objective': ['lambdarank'],  # 'rank_xendcg', 'lambdarank'
        'metric': ['ndcg'],
        'ndcg_at': [[1]],
        'learning_rate': [0.005],
        'num_leaves': [2000, 2500],  # [1750, 2000]
        'max_depth': [-1],
        'min_data_in_leaf': [20, 25],  # [20, 30, 40]
        'max_bin': [200],  # [200, 300]
        'bagging_fraction': [0.8],  # [0.75, 0.8]
        'bagging_freq': [1],
        'feature_fraction': [0.7],  # [0.7, 0.75]
        'lambda_l1': [0],  # [0, 0.000001]
        'lambda_l2': [0],  # [0, 0.000001]
        'seed': [1],
        'verbose': [0]
    }

    print("Grid search for hyperparameter tuning...")
    df = grid_search_single(X_arr, y_arr, group_arr, param_grid)
    file_name = 'hyperparam_tuning_result'
    if ms1_iso:
        file_name += '_ms1'
    if ms2_spec:
        file_name += '_ms2'
    file_name += '.csv'
    df.to_csv(file_name, index=False)


if __name__ == "__main__":

    tune_hyperparams(ms1_iso=False, ms2_spec=True)
    # tune_hyperparams(ms1_iso=True, ms2_spec=False)
    # tune_hyperparams(ms1_iso=True, ms2_spec=True)
    # tune_hyperparams(ms1_iso=False, ms2_spec=False)
