import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv("Datasets/drug200.csv")

# Encode categorical variables to numeric values
data["Sex"] = data["Sex"].map({"M": 1, "F": 0})
data["BP"] = data["BP"].map({"HIGH": 2, "NORMAL": 1, "LOW": 0})
data["Cholesterol"] = data["Cholesterol"].map({"NORMAL": 0, "HIGH": 1})
data["Drug"] = data["Drug"].map({"drugA": 0, "drugB": 1, "drugC": 2, "drugX": 3, "drugY": 4})

# Define features and target variable
x = data[["Sex", "Age", "BP", "Cholesterol", "Na_to_K"]]
y = data["Drug"]

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialize and train the Decision Tree Classifier
DT = DecisionTreeClassifier()
model = DT.fit(x_train, y_train)

# Predict on the test set
y_pred = model.predict(x_test)

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}%")

# Visualize the decision tree
fig = plt.figure(figsize=(15, 10))
tree.plot_tree(model, filled=True, feature_names=x.columns, class_names=["drugA", "drugB", "drugC", "drugX", "drugY"])
plt.show()