import warnings
import argparse
import logging

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import mlflow
import mlflow.sklearn

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
        "data/red-wine-quality.csv"
    )

    data = pd.read_csv(csv_url)

    # Split the data into training and test sets. (0.75, 0.25)
    train, test = train_test_split(data, random_state=42)

    # The predicted column is "quality" which is a scaler from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train["quality"]
    test_y = test["quality"]

    alpha = args.alpha
    l1_ratio = args.l1_ratio
    exp = mlflow.set_experiment(experiment_name="experiment_1")

    with mlflow.start_run(experiment_id=exp.experiment_id) as run:
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print(f"RMSE: {rmse}")
        print(f"MAE : {mae}")
        print(f"R2  : {r2}")
        print("ElasticNet model (alpha={:f}, l1_ratio={:f})".format(alpha, l1_ratio))

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        mlflow.sklearn.log_model(lr, "mymodel")





