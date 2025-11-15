import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn import decomposition

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

for kernel in kernels:
    clf = svm.SVC(kernel=kernel, random_state=42)
    clf.fit(x_train, y_train)
    acc = accuracy_score(y_test, clf.predict(x_test))
    accuracies.append(acc)
    print(f"SVM {kernel.upper():5} Accuracy: {acc:.4f}")
print(f"\nBest Model: {Best_model} | Accuracy: {Best_Accuracy:.4f}")

plt.figure(figsize=(8, 5))
bars = plt.bar(kernels, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c'], edgecolor='black')
plt.ylim(0, 1)
plt.ylabel('Accuracy')
plt.title('SVM Kernel Performance (Breast Cancer)')
for i, bar in enumerate(bars):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{accuracies[i]:.3f}', ha='center', fontsize=10)
plt.tight_layout()

kernel_path = os.path.join(OUTPUT_DIR, "svm_breastcancer_kernels.png")
plt.savefig(kernel_path, dpi=300, bbox_inches='tight')
print(f"Kernel comparison saved -> {kernel_path}")
plt.show()

pca_2d = decomposition.PCA(n_components=2)
pca_2d.fit(x_train)
x_pca_2d = pca_2d.transform(x_test)

plt.figure(figsize=(9, 7))
colors = ['#1f77b4', '#ff7f0e']  # Blue = Benign, Orange = Malignant
labels = ['Benign (0)', 'Malignant (1)']

for i, (label, color) in enumerate(zip(labels, colors)):
    mask = Best_Prediction == i
    plt.scatter(x_pca_2d[mask, 0], x_pca_2d[mask, 1],
                c=color, label=label, s=70, edgecolor='k', alpha=0.8)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title(f"SVM ({Best_model}) - 2D PCA Projection")
plt.legend(title="Prediction")
plt.grid(True, alpha=0.3)

pca2d_path = os.path.join(OUTPUT_DIR, "svm_breastcancer_2d.png")
plt.savefig(pca2d_path, dpi=300, bbox_inches='tight')
print(f"2D PCA saved -> {pca2d_path}")
plt.show()

pca_3d = decomposition.PCA(n_components=3)
pca_3d.fit(x_train)
x_pca_3d = pca_3d.transform(x_test)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

for i, (label, color) in enumerate(zip(labels, colors)):
    mask = Best_Prediction == i
    ax.scatter(x_pca_3d[mask, 0], x_pca_3d[mask, 1], x_pca_3d[mask, 2],
               c=color, label=label, s=60, edgecolor='k', depthshade=True)

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title(f"SVM ({Best_model}) - 3D PCA")
ax.legend(title="Prediction")

pca3d_path = os.path.join(OUTPUT_DIR, "svm_breastcancer_3d.png")
plt.savefig(pca3d_path, dpi=300, bbox_inches='tight')
print(f"3D PCA saved -> {pca3d_path}")
plt.show()