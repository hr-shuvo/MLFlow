import warnings
import argparse
import logging

from pathlib import Path
import os

import pandas as pd
import numpy as np
from mlflow import sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.dummy import DummyRegressor

import mlflow
import mlflow.sklearn
from mlflow.models import make_metric

import sklearn
import joblib
import cloudpickle
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# get arguments from command
parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, default=0.3, required=False)
parser.add_argument("--l1_ratio", type=float, default=0.6, required=False)
args = parser.parse_args()

# evaluation function
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from the url
    csv_url = (
        "red-wine-quality.csv"
    )

    data = pd.read_csv(csv_url)
    if not os.path.exists("data/"):
        os.mkdir("data/")
    data.to_csv("data/red-wine-quality.csv", index=False)

    # Split the data into training and test sets. (0.75, 0.25)
    train, test = train_test_split(data, random_state=42)

    data_dir = 'red-wine_data'
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    data.to_csv(data_dir + "/data.csv", index=False)
    train.to_csv(data_dir + "/train.csv", index=False)
    test.to_csv(data_dir + "/test.csv", index=False)

    # The predicted column is "quality" which is a scaler from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train["quality"]
    test_y = test["quality"]

    alpha = args.alpha
    l1_ratio = args.l1_ratio

    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
    # mlflow.set_tracking_uri(uri="")

    print("The set tracking uri is ", mlflow.get_tracking_uri())

    exp= mlflow.set_experiment(
        experiment_name="experiment_register_model_api"
    )
    get_exp = mlflow.get_experiment(exp.experiment_id)

    print("Exp id: ", get_exp.experiment_id)
    print(f"Name: {get_exp.name}")
    # print(f"Artifact Location: {get_exp.artifact_location}")
    # print(f"Tags: {get_exp.tags}")
    # print(f"Lifecycle_stage: {get_exp.lifecycle_stage}")
    # print(f"Creation timestamp: {get_exp.creation_time}")


    mlflow.start_run()

    # mlflow.set_tag('release.version', '0.1')
    tags={
        'engineering': 'ML platform',
        'release.candidate': 'RC1',
        'release.version': '2.0'
    }

    mlflow.set_tags(tags)

    # mlflow.sklearn.autolog(
    #     log_input_examples=False,
    #     log_model_signatures=False,
    #     log_models=False
    # )

    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_x, train_y)

    predicted_qualities = lr.predict(test_x)

    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    print(f"RMSE: {rmse}")
    print(f"MAE : {mae}")
    print(f"R2  : {r2}")
    print("ElasticNet model (alpha={:f}, l1_ratio={:f})".format(alpha, l1_ratio))

    mlflow.log_params({
        'alpha': alpha,
        'l1_ratio': l1_ratio
    })

    mlflow.log_metrics({
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    })


    sklearn_model_path = "sklearn_model.pkl"
    joblib.dump(lr, sklearn_model_path)
    artifacts = {
        'sklearn_model': sklearn_model_path,
        'data': data_dir
    }

    run = mlflow.last_active_run()
    mlflow.sklearn.log_model(lr, "model")

    mlflow.register_model(
        model_uri='runs:/{}/model'.format(run.info.run_id),
        name='elastic-api-2'
    )

    ld = mlflow.pyfunc.load_model(model_uri="models:/elastic-api-2/2")
    predicted_qualities = ld.predict(test_x)
    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)
    print("  RMSE_test: %s" % rmse)
    print("  MAE_test: %s" % mae)
    print("  R2_test: %s" % r2)
    mlflow.end_run()

    run = mlflow.last_active_run()
    print("Run ID  : ", run.info.run_id)
    print("Run name: ", run.info.run_name)





