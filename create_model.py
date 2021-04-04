from sklearn import datasets
from sklearn import svm
import mlflow 

clf = svm.SVC(gamma="scale")
iris = datasets.load_iris()

x, y = iris.data, iris.target
clf.fit(x, y)
print(x)
mlflow.sklearn.log_model(clf, "svm_model")