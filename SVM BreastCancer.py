import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn import decomposition

data = pd.read_csv("Datasets/breastcancer.csv")
data = data.iloc[:, :-1]
data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})

x = data.drop(columns=["id", "diagnosis"])
y = data["diagnosis"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

Best_model = "None"
Best_Accuracy = 0
Best_Prediction = None

clf = svm.SVC(kernel="linear")
clf.fit(x_train, y_train)
p = clf.predict(x_test)
print(f"Accuracy: {accuracy_score(y_test, p)}")
if accuracy_score(y_test, p) > Best_Accuracy:
    Best_Accuracy = accuracy_score(y_test, p)
    Best_model = "SVM Linear"
    Best_Prediction = p

clf = svm.SVC(kernel="poly")
clf.fit(x_train, y_train)
p = clf.predict(x_test)
print(f"Accuracy: {accuracy_score(y_test, p)}")
if accuracy_score(y_test, p) > Best_Accuracy:
    Best_Accuracy = accuracy_score(y_test, p)
    Best_model = "SVM Poly"
    Best_Prediction = p

clf = svm.SVC(kernel="rbf")
clf.fit(x_train, y_train)
p = clf.predict(x_test)
print(f"Accuracy: {accuracy_score(y_test, p)}")
if accuracy_score(y_test, p) > Best_Accuracy:
    Best_Accuracy = accuracy_score(y_test, p)
    Best_model = "SVM RBF"
    Best_Prediction = p

print(f"Best Model: {Best_model}")
print(f"Best Accuracy: {Best_Accuracy}")

kernels = ['linear', 'poly', 'rbf']
accuracies = []
preds = []

for k in kernels:
    clf = svm.SVC(kernel=k, random_state=42)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    acc = accuracy_score(y_test, pred)
    accuracies.append(acc)
    preds.append(pred)
    print(f"SVM {k.upper():5} → {acc:.4f}")

best_idx = accuracies.index(max(accuracies))
best_kernel = kernels[best_idx]
best_pred = preds[best_idx]
print(f"\nBest: SVM {best_kernel.upper()} → {accuracies[best_idx]:.4f}")

plt.figure(figsize=(8, 5))
bars = plt.bar(kernels, accuracies, edgecolor='black')
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{acc:.3f}', ha='center')
plt.ylim(0, 1); plt.ylabel('Accuracy'); plt.title('SVM Kernel Performance')
plt.savefig("output/svm_breastcancer_kernels.png", dpi=300, bbox_inches='tight')
plt.show()

pca_2d = decomposition.PCA(n_components=2)
pca_2d.fit(x_train)
x_pca_2d = pca_2d.transform(x_test)

# --- 2D PCA Plot ---
pca = decomposition.PCA(n_components=2)
x_pca = pca.fit_transform(x_test)

plt.figure(figsize=(8, 6))
plt.scatter(x_pca[:, 0], x_pca[:, 1], c=best_pred, cmap='bwr', s=70, edgecolor='k')
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.title(f"SVM ({best_kernel.upper()}) – 2-D PCA")
plt.legend(handles=[
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', label='Benign'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',  label='Malignant')
], title="Prediction")
plt.grid(True, alpha=0.3)
plt.savefig("output/svm_breastcancer_2d.png", dpi=300, bbox_inches='tight')
plt.show()

# --- 3D PCA Plot ---
pca = decomposition.PCA(n_components=3)
x_pca = pca.fit_transform(x_test)

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_pca[:, 0], x_pca[:, 1], x_pca[:, 2],
           c=best_pred, cmap='bwr', s=60, edgecolor='k')
ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
ax.set_title(f"SVM ({best_kernel.upper()}) – 3-D PCA")
ax.legend(handles=[
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', label='Benign'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',  label='Malignant')
], title="Prediction")
plt.savefig("output/svm_breastcancer_3d.png", dpi=300, bbox_inches='tight')
plt.show()