import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("Datasets/diabetes.csv")

x = data[["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]]
y = data["Outcome"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

RF = RandomForestClassifier(n_estimators=100)
model = RF.fit(x_train, y_train)

importances = model.feature_importances_
feature_names = x.columns
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}%")

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], 
         color='forestgreen', edgecolor='black')
plt.xlabel('Importance')
plt.title('Random Forest - Feature Importance (Diabetes)')
plt.gca().invert_yaxis()  # Most important on top
plt.tight_layout()

plt.savefig("output/rf_diabetes_importance.png", dpi=300, bbox_inches='tight')
print(f"Importance plot saved -> output/rf_diabetes_importance.png")
plt.show()