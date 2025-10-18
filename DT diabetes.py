import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv("Datasets/diabetes.csv")

# Define features and target variable
x = data[["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]]
y = data["Outcome"]

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
fig = plt.figure(figsize=(30, 20))
tree.plot_tree(model, filled=True, feature_names=x.columns, class_names=["No Diabetes", "Diabetes"], fontsize=3)
plt.show()