from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


# Optional: implement hyperparameter tuning.
def train_model(preprocessor_pipe, X_train, y_train, metric='precision'):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    lg = LogisticRegression(solver="liblinear")

    params = {
        "preprocessor__cat_transformer__drop": ["first", None],
        "logistic_regression__penalty": ["l1", "l2"],
        "logistic_regression__C": [0.1, 0.5, 1, 10]}

    pipe_model = Pipeline([
        ("preprocessor", preprocessor_pipe),
        ("logistic_regression", lg)
    ])

    grid_model = GridSearchCV(pipe_model,
                              param_grid=params,
                              cv=5,
                              scoring=metric)

    grid_model.fit(X_train, y_train)

    print(f"{metric}: {grid_model.cv_results_['mean_test_score'].mean()}")

    return grid_model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model
    using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : scikit-learn model
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds
