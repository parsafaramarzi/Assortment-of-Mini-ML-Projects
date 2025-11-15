import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = pd.read_csv("Datasets/Automobile_data.csv")

data['horsepower'] = pd.to_numeric(data['horsepower'].replace('?', pd.NA))
data['price'] = pd.to_numeric(data['price'].replace('?', pd.NA))

data = data.dropna(subset=['horsepower', 'price'])

plt.scatter(data['horsepower'], data['price'])
plt.xlabel("Horsepower")
plt.ylabel("Price")
plt.title("Horsepower vs Price")
plt.savefig("output/linear_reg_automobile.png", dpi=300, bbox_inches='tight')
print(f"Saved scatter plot -> output/linear_reg_automobile.png")
plt.show()

X = data[['horsepower']]
y = data[['price']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("R^2 Score:", r2_score(y_test, y_pred))

plt.plot(X_test, y_test, 'o', label="Actual")
plt.plot(X_test, y_pred, 'r-', label="Predicted")
plt.xlabel("Horsepower")
plt.ylabel("Price")
plt.title(f"Actual vs Predicted Prices (R² = {r2_score(y_test, y_pred):.3f})")
plt.legend()
plt.savefig("output/linear_reg_automobile_pred.png", dpi=300, bbox_inches='tight')
print(f"Saved prediction plot -> output/linear_reg_automobile_pred.png")
plt.show()