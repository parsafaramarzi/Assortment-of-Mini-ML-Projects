import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn import decomposition

data = pd.read_csv("Datasets/breastcancer.csv")
data = data.iloc[:, :-1]
data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})

x = data.drop(columns=["id", "diagnosis"])
y = data["diagnosis"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

Best_model = "None"
Best_Accuracy = 0
Best_Prediction = None

clf = svm.SVC(kernel="linear")
clf.fit(x_train, y_train)
p = clf.predict(x_test)
print(f"Accuracy: {accuracy_score(y_test, p)}")
if accuracy_score(y_test, p) > Best_Accuracy:
    Best_Accuracy = accuracy_score(y_test, p)
    Best_model = "SVM Linear"
    Best_Prediction = p

clf = svm.SVC(kernel="poly")
clf.fit(x_train, y_train)
p = clf.predict(x_test)
print(f"Accuracy: {accuracy_score(y_test, p)}")
if accuracy_score(y_test, p) > Best_Accuracy:
    Best_Accuracy = accuracy_score(y_test, p)
    Best_model = "SVM Poly"
    Best_Prediction = p

clf = svm.SVC(kernel="rbf")
clf.fit(x_train, y_train)
p = clf.predict(x_test)
print(f"Accuracy: {accuracy_score(y_test, p)}")
if accuracy_score(y_test, p) > Best_Accuracy:
    Best_Accuracy = accuracy_score(y_test, p)
    Best_model = "SVM RBF"
    Best_Prediction = p

print(f"Best Model: {Best_model}")
print(f"Best Accuracy: {Best_Accuracy}")

pca_2d = decomposition.PCA(n_components=2)
pca_2d.fit(x_train)
x_pca_2d = pca_2d.transform(x_test)

plt.scatter(x_pca_2d[:, 0], x_pca_2d[:, 1], c=Best_Prediction, cmap="viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("SVM of Breast Cancer Dataset")
plt.show()

pca_3d = decomposition.PCA(n_components=3)
pca_3d.fit(x_train)
x_pca_3d = pca_3d.transform(x_test)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_pca_3d[:, 0], x_pca_3d[:, 1], x_pca_3d[:, 2], c=Best_Prediction, cmap="viridis")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_zlabel("Principal Component 3")
ax.set_title("SVM of Breast Cancer Dataset (3D)")
plt.show()