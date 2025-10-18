import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Function to convert square footage to numeric
def convert_sqft_to_num(x):
    try:
        if '-' in str(x):
            tokens = x.split('-')
            return (float(tokens[0]) + float(tokens[1])) / 2
        return float(x)
    except:
        return np.nan

data = pd.read_csv("Datasets/bengaluru_house_prices.csv")
data['total_sqft'] = data['total_sqft'].apply(convert_sqft_to_num)
data = data.dropna(subset=['total_sqft', 'price'])

# Showing the data on graph
plt.scatter(data['total_sqft'], data['price'])
plt.xlabel("Total Square Feet")
plt.ylabel("Price")
plt.title("Bengaluru House Prices")
plt.show()

# Preparing and splitting the data
X = data[['total_sqft']]
y = data[['price']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,)

# Loading the model and fitting the dataset on model
model = LinearRegression()
model.fit(X_train, y_train)

# Calculating model performance
y_pred = model.predict(X_test)
print("R^2 Score:", r2_score(y_test, y_pred))

plt.plot(X_test, y_test, 'o', label='Actual')
plt.plot(X_test, y_pred, 'r-', label='Predicted')
plt.xlabel("Total Square Feet")
plt.ylabel("Price")
plt.title("Bengaluru House Prices - Actual vs Predicted")
plt.legend()
plt.show()