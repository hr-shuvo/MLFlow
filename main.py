import warnings
import argparse
import logging

import pandas as pd
import numpy as np
from mlflow import sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import mlflow
import mlflow.sklearn

from pathlib import Path
import os
from mlflow.models.signature import ModelSignature, infer_signature
from mlflow.types.schema import Schema, ColSpec
import sklearn
import joblib
import cloudpickle

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# get arguments from command
parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, default=0.7, required=False)
parser.add_argument("--l1_ratio", type=float, default=0.5, required=False)
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
    # train.to_csv("data/train.csv", index=False)
    # test.to_csv("data/test.csv", index=False)

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

    # mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
    mlflow.set_tracking_uri(uri="")

    print("The set tracking uri is ", mlflow.get_tracking_uri())

    exp= mlflow.set_experiment(
        experiment_name="experiment_custom_sklearn"
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

    mlflow.sklearn.autolog(
        log_input_examples=False,
        log_model_signatures=False,
        log_models=False
    )

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


    class SklearnWrapper(mlflow.pyfunc.PythonModel):
        # def __init__(self):
        #     self.sklearn_model = None

        def load_context(self, context):
            self.sklearn_model = joblib.load(context.artifacts['sklearn_model'])

        def predict(self, context, model_input):
            return self.sklearn_model.predict(model_input.values)

    conda_env = {
        "channels": ["defaults"],
        "dependencies": [
            "python={}".format(3.10),
            "pip",
            {
                "pip": [
                    "mlflow=={}".format(mlflow.__version__),
                    "scikit-learn=={}".format(sklearn.__version__),
                    "cloudpickle=={}".format(cloudpickle.__version__),
                ],
            },
        ],
        "name": "sklearn_env",
    }



    mlflow.pyfunc.log_model(
        artifact_path="sklear_mlflow_pyfunc",
        python_model=SklearnWrapper(),
        artifacts=artifacts,
        code_path=['main.py'],
        conda_env=conda_env,

    )

    ld = mlflow.pyfunc.load_model(model_uri='runs:/5ec61b4e47c44990bbff990e011d185f/sklear_mlflow_pyfunc')
    predicted_qualities = ld.predict(test_x)

    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)
    print(f"RMSE: {rmse}")
    print(f"MAE : {mae}")
    print(f"R2  : {r2}")

    artifact_uri = mlflow.get_artifact_uri()
    print("The artifact path is ", artifact_uri)


    mlflow.end_run()

    run = mlflow.last_active_run()
    print("Run ID  : ", run.info.run_id)
    print("Run name: ", run.info.run_name)





