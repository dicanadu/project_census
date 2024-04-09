from ml_model.train_model import train_and_save, performance_testing
from ml_model.ml.data import process_data, load_data
from sklearn.model_selection import train_test_split
import argparse
import os
import json


def deploy_model_slice(metric,
                       train_save,
                       slicing):
    """
    Load, process, trains and save a model
    Inputs
    ------
    metric: Choose metric to optimize model
    train_save: Set True to save the model on the folder package_code/model
                False to get last model on package_code/model
    slicing: Select a column to fix categories and get
             metrics for each category
    Output
    ------
    model: return a model fitted or last_available model
    """
    root = os.path.dirname(__file__)
    data_path = os.path.join(root, "package_code", "data", "census.csv")
    model_path = os.path.join(root, "package_code", "model")
    slicing_path = os.path.join(root, f"output_{slicing}_{metric}.json")

    df = load_data(data_path)
    preprocessor_pipe, X, y = process_data(df, "salary")
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.20,
                                                        stratify=y)
    if train_save is True:
        model = train_and_save(X_train,
                               y_train,
                               preprocessor_pipe,
                               path=model_path,
                               metric=metric)
    else:
        model = os.listdir(model_path)[-1]

    if slicing is not None:
        results = performance_testing(model, X_test, y_test, slicing)
        with open(slicing_path, "w") as file:
            json.dump(results, file, indent=2)
    else:
        pass

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", type=str,
                        default="accuracy", required=False)
    parser.add_argument("--slicing", type=str,
                        default=None, required=False)
    parser.add_argument("--save", type=bool,
                        default=True, required=False)
    args = parser.parse_args()

    deploy_model_slice(metric=args.metric,
                       slicing=args.slicing,
                       train_save=args.save)
