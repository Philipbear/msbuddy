from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
import pandas as pd
import numpy as np
import joblib
import argparse


def train_cls_model(X_file, z_norm=True):
    """
    classification model training
    """
    X = pd.read_csv(X_file)

    # z-normalization
    if z_norm:
        # feature normalization using z-score, save mean and std
        mean_arr = np.array([])
        std_arr = np.array([])
        # do this for each column except the last two
        for col in X.columns[:-2]:
            mean = X[col].mean()
            std = X[col].std()
            mean_arr = np.append(mean_arr, mean)
            std_arr = np.append(std_arr, std)
            X[col] = (X[col] - mean) / std
        joblib.dump(mean_arr, "mean_arr.joblib")
        joblib.dump(std_arr, "std_arr.joblib")

    # first half is pos, second half is neg
    y = np.array([1] * int(X.shape[0] / 2))
    y = np.append(y, [0] * int(X.shape[0] / 2))

    print("train test split")
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        train_size=0.8,
                                                        random_state=2)

    print("cross validation & grid search...")
    # grid search
    param_grid = {'hidden_layer_sizes': [(512,), (256,), (128,),
                                         (128, 128), (128, 64), (64, 64), (64, 32),
                                         (128, 64, 64), (64, 64, 32), (64, 32, 32), (32, 32, 16), (32, 16, 16),
                                         (64, 32, 32, 16), (32, 16, 16, 8)],
                  'alpha': [1e-5, 1e-4, 1e-3, 1e-2],
                  'max_iter': [800]}

    mlp = MLPClassifier(random_state=1)
    clf = GridSearchCV(mlp, param_grid, cv=5, n_jobs=-1, verbose=1)
    clf.fit(X_train, y_train)

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
    if z_norm:
        joblib.dump(best_mlp, "model_a_z_norm.joblib")
    else:
        joblib.dump(best_mlp, "model_a.joblib")


def arg_parser():
    parser = argparse.ArgumentParser(description='ML-A training')
    parser.add_argument('--z_norm', action='store_true', help='Whether to z-normalize the features.')
    _args = parser.parse_args()
    return _args


if __name__ == '__main__':

    args = arg_parser()
    train_cls_model('formula_training_X.csv', z_norm=args.z_norm)

