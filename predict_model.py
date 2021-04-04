import mlflow.sklearn
import numpy as np

sk_model = mlflow.sklearn.load_model("svm_model")

predictions = sk_model.predict(np.array([[5.9, 3.,  5.1, 1.8]]))
print(predictions)
