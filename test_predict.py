import mlflow.sklearn
import numpy as np

sk_model = mlflow.sklearn.load_model("runs:/b935a8d67eb84731a9454b0fbf7f31b3/model")

predictions = sk_model.predict(np.array([[1, 1]]))
print(predictions)
