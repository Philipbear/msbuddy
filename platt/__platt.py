from sklearn.linear_model import LogisticRegression
import numpy as np
import joblib


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


def ml_b_pred(x, model_path):
    """
    ML b prediction
    :param x: X matrix
    :param model_path: path to the model
    :return: np.array
    """
    model = joblib.load(model_path)
    return model.predict_proba(x)[:, 1]



if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    X_arr = joblib.load('../ml_b_train/gnps_X_arr_SMOTE.joblib')
    y_gt = joblib.load('../ml_b_train/gnps_y_arr_SMOTE.joblib')

    X_train, X_test, y_train, y_test = train_test_split(X_arr, y_gt,
                                                        test_size=0.2, random_state=0)

    # discard the last 2 columns
    X_test = X_test[:, :-2]
    # discard the 2nd column
    X_test = np.delete(X_test, 1, 1)

    y_prediction = ml_b_pred(X_test,
                             '../msbuddy/data/model_b_noms1_ms2.joblib')

    a, b = platt_calibration(y_prediction, y_test)
    print(a, b)

