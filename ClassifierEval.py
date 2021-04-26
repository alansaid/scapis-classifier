from sklearn.datasets import load_svmlight_file

X_data, y_data = load_svmlight_file("../scapis-parser/parsed-data/libsvmdata.txt")

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.20)

print("splits done")

print("svc classifier")
from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)

print('model trained')

y_pred = svclassifier.predict(X_test)

print("predictions done")

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
