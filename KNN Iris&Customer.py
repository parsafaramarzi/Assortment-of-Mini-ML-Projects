import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets, preprocessing
from sklearn.metrics import accuracy_score

data_iris = datasets.load_iris()
data_customer = pd.read_csv("Datasets/Mall_Customers.csv")

le = preprocessing.LabelEncoder()
data_iris.target = le.fit_transform(data_iris.target)
x_iris = data_iris.data
y_iris = data_iris.target

data_customer["Gender"] = le.fit_transform(data_customer["Gender"])
x_customer = data_customer[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]
y_customer = data_customer["Gender"]

x_iris_train, x_iris_test, y_iris_train, y_iris_test = train_test_split(x_iris, y_iris, test_size=0.2, random_state=42)
x_customer_train, x_customer_test, y_customer_train, y_customer_test = train_test_split(x_customer, y_customer, test_size=0.2, random_state=42)

KNN = KNeighborsClassifier(n_neighbors=5)
KNN.fit(x_iris_train, y_iris_train)
y_iris_pred = KNN.predict(x_iris_test)
print("Iris Classification Accuracy:", accuracy_score(y_iris_test, y_iris_pred)*100)

KNN.fit(x_customer_train, y_customer_train)
y_customer_pred = KNN.predict(x_customer_test)
print("Customer Gender Classification Accuracy:", accuracy_score(y_customer_test, y_customer_pred)*100)

plt.subplot(1, 2, 1)
plt.scatter(x_iris_test[:, 0], x_iris_test[:, 1], c=y_iris_pred, cmap='viridis')
plt.title("Iris Classification")

plt.subplot(1, 2, 2)
plt.scatter(x_customer_test["Age"], x_customer_test["Annual Income (k$)"], c=y_customer_pred, cmap='viridis')
plt.title("Customer Gender Classification")
plt.show()