import pandas as pd
import matplotlib.pyplot as plt
from sklearn import decomposition
import numpy as np

data = pd.read_csv("Datasets/WineQT.csv")
x = data.drop(columns=["quality", "Id"], axis=1)
y = data["quality"]

pca_full = decomposition.PCA()
pca_full.fit(x)
cum_var = np.cumsum(pca_full.explained_variance_ratio_)
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cum_var)+1), cum_var, 'bo-')
plt.axhline(0.9, color='r', linestyle='--', label='90%')
plt.xlabel('Components')
plt.ylabel('Cumulative Variance')
plt.title('PCA Explained Variance')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("output/pca_wine_variance.png", dpi=300, bbox_inches='tight')
plt.show()

pca = decomposition.PCA(n_components=2)
x_pca = pca.fit_transform(x)

plt.figure(figsize=(8, 6))
quality_levels = sorted(y.unique())
colors = plt.cm.Set1(np.linspace(0, 1, len(quality_levels)))

for q, color in zip(quality_levels, colors):
    mask = y == q
    plt.scatter(x_pca[mask, 0], x_pca[mask, 1], c=[color], label=f'Quality {q}', s=60, edgecolor='k')

plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
plt.title("Wine Quality - 2D PCA")
plt.legend(title="Quality")
plt.grid(True, alpha=0.3)
plt.savefig("output/pca_wine_2d.png", dpi=300, bbox_inches='tight')
plt.show()
print(f"Number of features before: {x.shape[1]} vs number of features now: {x_pca.shape[1]}")
print(f"2 PCs explain {sum(pca.explained_variance_ratio_):.1%} of variance")