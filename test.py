import numpy as np
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LinearRegression

def print_auto_logged_info(r):
  tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
  artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
  print(r)
  print("run_id: {}".format(r.info.run_id))
  print("artifacts: {}".format(artifacts))
  print("params: {}".format(r.data.params))
  print("metrics: {}".format(r.data.metrics))
  print("tags: {}".format(tags))

X = np.array([[1,1], [1,2], [2,2], [2,3]])
y = np.dot(X, np.array([1,2])) + 3

print(X, y)
mlflow.autolog()

model = LinearRegression()
with mlflow.start_run() as run:
  model.fit(X, y)
  print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))