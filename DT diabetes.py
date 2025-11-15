import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("Datasets/diabetes.csv")

x = data[["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]]
y = data["Outcome"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

DT = DecisionTreeClassifier()
model = DT.fit(x_train, y_train)

importances = model.feature_importances_
feature_names = x.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}%")

fig = plt.figure(figsize=(30, 20))
tree.plot_tree(model, filled=True, feature_names=x.columns, class_names=["No Diabetes", "Diabetes"], fontsize=3)
fig.savefig("output/dt_diabetes_tree.png", dpi=300, bbox_inches='tight')
print(f"Tree saved -> output/dt_diabetes_tree.png")
plt.show()

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue', edgecolor='black')
plt.xlabel('Importance')
plt.title('Decision Tree - Feature Importance Ranking')
plt.gca().invert_yaxis()
plt.tight_layout()

plt.savefig("output/dt_diabetes_importance.png", dpi=300, bbox_inches='tight')
print(f"Importance plot saved -> output/dt_diabetes_importance.png")
plt.show()