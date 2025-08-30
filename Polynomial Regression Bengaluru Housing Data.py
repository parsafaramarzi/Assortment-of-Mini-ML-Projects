import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def convert_sqft_to_num(x):
    try:
        if '-' in str(x):
            tokens = x.split('-')
            return (float(tokens[0]) + float(tokens[1])) / 2
        return float(x)
    except:
        return np.nan
    
# Prepare the data and drop the NA values
data = pd.read_csv("bengaluru_house_prices.csv")
data['total_sqft'] = data['total_sqft'].apply(convert_sqft_to_num)
data = data.dropna(subset=['total_sqft', 'price'])

# Visualize the data
plt.scatter(data['total_sqft'], data['price'])
plt.xlabel('Total Square Feet')
plt.ylabel('Price')
plt.title('Bengaluru Housing Prices')
plt.show()

total_squarefeet = data['total_sqft'].values
total_prices = data['price'].values

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(total_squarefeet, total_prices, test_size=0.3)

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
plt.xlabel('Total Square Feet')
plt.ylabel('Price')
plt.title('Bengaluru Housing Prices - Polynomial Regression')
plt.legend()
plt.show()