import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
    
data = pd.read_csv("Datasets/Automobile_data.csv")

data['horsepower'] = pd.to_numeric(data['horsepower'].replace('?', pd.NA))
data['price'] = pd.to_numeric(data['price'].replace('?', pd.NA))

data = data.dropna(subset=['horsepower', 'price'])

plt.scatter(data['horsepower'], data['price'], alpha=0.6, edgecolor='k')
plt.xlabel('Horsepower')
plt.ylabel('Price')
plt.title('Horsepower vs Price')
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig("output/poly_reg_automobile_scatter.png", dpi=200, bbox_inches='tight')
plt.show()

horsepower = data['horsepower'].values
prices = data['price'].values

x_train, x_test, y_train, y_test = train_test_split(horsepower, prices, test_size=0.3, random_state=42)

bestdegree = 0
bestaccuracy = 0

for degree in range(1, 20):
    model = np.poly1d(np.polyfit(x_train, y_train, degree))
    p = model(x_test)
    accuracy = r2_score(y_test, p)
    if accuracy > bestaccuracy:
        bestaccuracy = accuracy
        bestdegree = degree

print(f"Best Degree: {bestdegree}, Best Accuracy: {bestaccuracy}")

bestmodel = np.poly1d(np.polyfit(x_train, y_train, bestdegree))
p = bestmodel(x_test)

sort_idx = np.argsort(x_test)
x_sorted = x_test[sort_idx]
p_sorted = p[sort_idx]

plt.scatter(x_test, y_test, color='blue', alpha=0.6, edgecolor='k', label='Actual Prices')
plt.plot(x_sorted, p_sorted, color='red', linewidth=2, label='Predicted Prices')
plt.xlabel('Horsepower')
plt.ylabel('Price')
plt.title(f'Polynomial Regression (Degree = {bestdegree}, R² = {bestaccuracy:.3f})')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig("output/poly_reg_automobile.png", dpi=200, bbox_inches='tight')
plt.show()