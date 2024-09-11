import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
import sklearn.model_selection
from sklearn.neighbors import KNeighborsClassifier

Cancer = datasets.load_breast_cancer()
print(Cancer.feature_names)
print(Cancer.target_names)

x = Cancer.data
y= Cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.2)

clf = svm.SVC(kernel="linear")
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

acc = metrics.accuracy_score(y_test, y_pred)

print(acc)