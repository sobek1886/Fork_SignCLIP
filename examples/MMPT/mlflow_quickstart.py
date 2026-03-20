"""MLflow connectivity quickstart.

Verifies that the cluster can reach the company MLflow server and log a run.
Check results at https://mlflow.ai.mytkhgroup.com/ after running.

Requires ~/.mlflow/credentials on the machine running this script:
    [mlflow]
    mlflow_tracking_username = your.name@tkh.ai
    mlflow_tracking_password = ...
"""

import os

import mlflow
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("https://mlflow.ai.mytkhgroup.com/")
mlflow.set_experiment("signclip-connectivity-test")

print(f"Tracking URI : {mlflow.get_tracking_uri()}")
print(f"Compute node : {os.environ.get('SLURMD_NODENAME', 'unknown')}")
print(f"SLURM job    : {os.environ.get('SLURM_JOB_ID', 'unknown')}")

X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

params = {"solver": "lbfgs", "max_iter": 1000, "random_state": 8888}

with mlflow.start_run():
    mlflow.log_params(params)
    mlflow.set_tag("node", os.environ.get("SLURMD_NODENAME", "unknown"))
    mlflow.set_tag("job_id", os.environ.get("SLURM_JOB_ID", "unknown"))

    lr = LogisticRegression(**params)
    lr.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, lr.predict(X_test))
    mlflow.log_metric("accuracy", accuracy)

    print(f"Accuracy: {accuracy:.4f}")
    print("Run logged — check https://mlflow.ai.mytkhgroup.com/")
