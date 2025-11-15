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

plt.savefig("output/svm_breastcancer_kernels.png", dpi=300, bbox_inches='tight')
print(f"Kernel comparison saved -> output/svm_breastcancer_kernels.png")
plt.show()

pca_2d = decomposition.PCA(n_components=2)
pca_2d.fit(x_train)
x_pca_2d = pca_2d.transform(x_test)

# --- 2D PCA Plot ---
pca = decomposition.PCA(n_components=2)
x_pca_2d = pca.fit_transform(x_test)

plt.figure(figsize=(8, 6))
plt.scatter(x_pca_2d[:, 0], x_pca_2d[:, 1], c=Best_Prediction, 
            cmap='bwr', s=70, edgecolor='k')  # Blue=Benign, Red=Malignant
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title(f"SVM ({Best_model}) - 2D PCA")
plt.legend(handles=[
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', label='Benign'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',  label='Malignant')
], title="Prediction")
plt.grid(True, alpha=0.3)
plt.savefig("output/svm_breastcancer_2d.png", dpi=300, bbox_inches='tight')
plt.show()

# --- 3D PCA Plot ---
pca = decomposition.PCA(n_components=3)
x_pca_3d = pca.fit_transform(x_test)

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_pca_3d[:, 0], x_pca_3d[:, 1], x_pca_3d[:, 2], 
           c=Best_Prediction, cmap='bwr', s=60, edgecolor='k')
ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
ax.set_title(f"SVM ({Best_model}) - 3D PCA")
ax.legend(handles=[
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', label='Benign'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',  label='Malignant')
], title="Prediction")
plt.savefig("output/svm_breastcancer_3d.png", dpi=300, bbox_inches='tight')
plt.show()