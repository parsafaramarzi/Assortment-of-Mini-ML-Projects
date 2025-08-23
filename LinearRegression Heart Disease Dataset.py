import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = pd.read_csv("heart_disease_dataset.csv")

plt.scatter(data['bmi'], data['chol'])
plt.xlabel('BMI')
plt.ylabel('Cholesterol')
plt.title('BMI vs Cholesterol')
plt.show()

X = data[['bmi']]
y = data[['chol']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("R^2 Score:", r2_score(y_test, y_pred))