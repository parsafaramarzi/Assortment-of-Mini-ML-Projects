import pandas as pd
import matplotlib.pyplot as plt
from sklearn import decomposition

data = pd.read_csv("Datasets/WineQT.csv")
x = data.drop(columns=["quality", "Id"], axis=1)
y = data["quality"]

pca = decomposition.PCA(n_components=2)
x_pca = pca.fit_transform(x)
print(f"Number of features before: {x.shape[1]} vs number of features now: {x_pca.shape[1]}")

plt.scatter(x_pca[:, 0], x_pca[:, 1], c=y, cmap='viridis')
plt.title("PCA of Wine Quality")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label="Wine Quality")
plt.show()