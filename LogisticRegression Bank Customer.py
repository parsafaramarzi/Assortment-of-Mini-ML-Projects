import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


data = pd.read_csv("Datasets/Bank Customer Churn Prediction.csv")
data = data.drop("customer_id", axis=1)
le = preprocessing.LabelEncoder()
data["country"] = le.fit_transform(data["country"])
data["gender"] = le.fit_transform(data["gender"])
x = data.drop("churn", axis=1)
y = data["churn"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy*100:.2f}%")