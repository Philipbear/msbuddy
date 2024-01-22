from sklearn.linear_model import LogisticRegression
import numpy as np
import joblib

from ml_b.__train import _train_val_test_split


def platt_calibration(y_pred, y_gt):
    """
    Platt calibration for binary classification
    :param y_pred: predicted scores
    :param y_gt: ground truth
    :return: coefficients of the sigmoid function
    """
    # Reshape data
    prediction_scores = y_pred.reshape(-1, 1)
    ground_truths = y_gt.reshape(-1)

    # Train Platt scaling model (Logistic Regression)
    platt_scaler = LogisticRegression()
    platt_scaler.fit(prediction_scores, ground_truths)

    # Get the learned parameters
    a = platt_scaler.coef_[0][0]
    b = platt_scaler.intercept_[0]

    return a, b


def ml_pred(x, model_path):
    """
    ML b prediction
    :param x: X matrix
    :param model_path: path to the model
    :return: np.array
    """
    model = joblib.load(model_path)
    return model.predict(x)


if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    X_arr = joblib.load('gnps_X_arr_filled.joblib')
    y_arr = joblib.load('gnps_y_arr.joblib')
    group_arr = joblib.load('gnps_group_arr.joblib')

    (X_train, X_val, X_test, y_train, y_val, y_test,
     groups_train, groups_val, groups_test) = _train_val_test_split(X_arr, y_arr, group_arr,
                                                                    val_size=0.1, test_size=0.1, random_state=24)

    # # discard the ms1 iso feature in X_arr
    X_test = X_test[:, 1:]
    #
    # # discard the last 24 features in X_arr
    # X_test = X_test[:, :-24]

    y_prediction = ml_pred(X_test, 'model_ms2.joblib')

    a, b = platt_calibration(y_prediction, y_test)
    print(a, b)

