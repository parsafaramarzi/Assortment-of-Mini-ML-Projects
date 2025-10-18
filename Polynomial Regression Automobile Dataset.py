import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

    
data = pd.read_csv("Datasets/Automobile_data.csv")

# Replace '?' with NaN and convert columns to numeric
data['horsepower'] = pd.to_numeric(data['horsepower'].replace('?', pd.NA))
data['price'] = pd.to_numeric(data['price'].replace('?', pd.NA))

# Drop rows with missing values in relevant columns
data = data.dropna(subset=['horsepower', 'price'])

# Visualize the data
plt.scatter(data['horsepower'], data['price'])
plt.xlabel('Horsepower')
plt.ylabel('Price')
plt.title('Automobile Prices')
plt.show()

horsepower = data['horsepower'].values
prices = data['price'].values

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(horsepower, prices, test_size=0.3)

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

plt.scatter(x_test, y_test, color='blue', label='Actual Prices')
plt.scatter(x_test, p, color='red', label='Predicted Prices')
plt.xlabel('Horsepower')
plt.ylabel('Price')
plt.title('Automobile Prices - Polynomial Regression')
plt.legend()
plt.show()