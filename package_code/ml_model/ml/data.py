import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


def load_data(path):
    """
    Load data from csv path

    Inputs
    ------
    path: Path to csv file.
    label: Name of y column to exclude from dataframe. Default value set to
           salary.

    Returns
    -------
    df_full: Complete dataframe for preprocessing.
    """
    df_full = pd.read_csv(path)
    df_full.columns = [col.strip() for col in df_full.columns]

    categorical_columns = df_full.select_dtypes(
            include=["object", "string"]).columns
    for col in categorical_columns:
        df_full[col] = df_full[col].apply(lambda x: x.strip())

    int64_columns = df_full.select_dtypes("int64").columns
    df_full[int64_columns] = df_full[int64_columns].astype(float)

    return df_full


def process_data(
        X, label=None):
    """ Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features
    and a label binarizer for the labels.
    This can be used in either training or inference/validation.

    Note: depending on the type of model used,
    you may want to add in functionality that
    scales the continuous data.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label.
        Columns in `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`.
        If None, then an empty array will be returned
        for y (default=None)
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True,
        otherwise returns the encoder passed
        in.
    pipe_features : sklearn.preprocessing.pipeline
        Pipeline designed to preprocess data
    """

    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    try:
        numeric_columns = X.select_dtypes(include=["int", "float"]).columns
        categorical_columns = X.select_dtypes(
            include=["object", "string"]).columns

        encoder = OneHotEncoder(sparse_output=False,
                                handle_unknown="ignore")

        pipe_features = ColumnTransformer([
            ("cat_transformer", encoder, categorical_columns),
            ("num_transformer", StandardScaler(), numeric_columns)
        ])

        lb = LabelBinarizer()
        y = lb.fit_transform(y).ravel()
    # Catch the case where y is None because we're doing inference.
    except AttributeError:
        pass

    return pipe_features, X, y


if __name__ == "__main__":
    # path="/home/dicanadu/code/udacity/ML_DevOps/ \
    # project_census/package_code/data/census.csv"
    # df = load_data(path)
    # print(dict(df.iloc[0,:]))
    pass
