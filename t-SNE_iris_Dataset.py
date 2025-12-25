import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
import itertools
import os
from sklearn.manifold import TSNE, trustworthiness
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
os.makedirs('output', exist_ok=True)

#----------------------------------------------------
# Setup Data and Global Constants
#----------------------------------------------------
data_iris = datasets.load_iris()
X_iris = data_iris.data
y_iris = data_iris.target
feature_names = data_iris.feature_names
target_names = data_iris.target_names

colors = ['red', 'blue', 'green']
cmap = plt.cm.colors.ListedColormap(colors)
legend_handles = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[0], label=target_names[0]),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[1], label=target_names[1]),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[2], label=target_names[2])
]

#----------------------------------------------------
# t-SNE Calculations
#----------------------------------------------------
tsne_3d = TSNE(n_components=3, random_state=42, init='pca', learning_rate='auto')
X_tsne_3d = tsne_3d.fit_transform(X_iris)

tsne_2d = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
X_tsne_2d = tsne_2d.fit_transform(X_iris)

tsne_1d = TSNE(n_components=1, random_state=42, init='pca', learning_rate='auto')
X_tsne_1d = tsne_1d.fit_transform(X_iris)


#----------------------------------------------------
# t-SNE Quality Metrics
#----------------------------------------------------
k = 5

trust_3d = trustworthiness(X_iris, X_tsne_3d, n_neighbors=k)
cont_3d = trustworthiness(X_tsne_3d, X_iris, n_neighbors=k)

trust_2d = trustworthiness(X_iris, X_tsne_2d, n_neighbors=k)
cont_2d = trustworthiness(X_tsne_2d, X_iris, n_neighbors=k)

trust_1d = trustworthiness(X_iris, X_tsne_1d, n_neighbors=k)
cont_1d = trustworthiness(X_tsne_1d, X_iris, n_neighbors=k)

print(f"3D t-SNE (k={k}): Trustworthiness (T)={trust_3d:.4f}, Continuity (C)={cont_3d:.4f}")
print(f"2D t-SNE (k={k}): Trustworthiness (T)={trust_2d:.4f}, Continuity (C)={cont_2d:.4f}")
print(f"1D t-SNE (k={k}): Trustworthiness (T)={trust_1d:.4f}, Continuity (C)={cont_1d:.4f}")

#----------------------------------------------------
# 3D VISUALIZATION: 3D Feature Combinations (4 plots) + 3D t-SNE (1 plot)
#----------------------------------------------------
feature_combinations_3d = list(itertools.combinations(range(X_iris.shape[1]), 3))
fig = plt.figure(figsize=(18, 12))
fig.suptitle('Iris Dataset: 3D Feature Combinations vs. 3D t-SNE', fontsize=18, y=1.02)

for i, (f1_idx, f2_idx, f3_idx) in enumerate(feature_combinations_3d):
    ax = fig.add_subplot(2, 3, i + 1, projection='3d')
    ax.scatter(
        X_iris[:, f1_idx], 
        X_iris[:, f2_idx], 
        X_iris[:, f3_idx], 
        c=y_iris, 
        cmap=cmap, 
        s=80, 
        edgecolor='k'
    )
    ax.set_xlabel(feature_names[f1_idx])
    ax.set_ylabel(feature_names[f2_idx])
    ax.set_zlabel(feature_names[f3_idx])
    ax.set_title(f'{feature_names[f1_idx]} vs {feature_names[f3_idx]} vs {feature_names[f2_idx]}', fontsize=10)

ax_tsne_3d = fig.add_subplot(2, 3, 5, projection='3d')
ax_tsne_3d.scatter(
    X_tsne_3d[:, 0], 
    X_tsne_3d[:, 1], 
    X_tsne_3d[:, 2], 
    c=y_iris, 
    cmap=cmap, 
    s=80, 
    edgecolor='k'
)
ax_tsne_3d.set_title(f'3D t-SNE Projection\nT: {trust_3d:.3f}, C: {cont_3d:.3f} (k={k})', fontsize=10)
ax_tsne_3d.set_xlabel('t-SNE 1')
ax_tsne_3d.set_ylabel('t-SNE 2')
ax_tsne_3d.set_zlabel('t-SNE 3')

fig.legend(
    handles=legend_handles, 
    title="Species", 
    loc='lower center', 
    ncol=3,
    fontsize=12,
    title_fontsize=14,
    bbox_to_anchor=(0.5, 0)
)
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig('output/tsne_iris_3d_comparison.png')
plt.show()

#----------------------------------------------------
# 2D VISUALIZATION: 2D Feature Combinations (6 plots) + 2D t-SNE (1 plot)
#----------------------------------------------------
feature_combinations_2d = list(itertools.combinations(range(X_iris.shape[1]), 2))
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('Iris Dataset: 2D Feature Combinations vs. 2D t-SNE', fontsize=18, y=1.02)
axes = axes.flatten()

for i, (f1_idx, f2_idx) in enumerate(feature_combinations_2d):
    ax = axes[i]
    ax.scatter(
        X_iris[:, f1_idx], 
        X_iris[:, f2_idx], 
        c=y_iris, 
        cmap=cmap, 
        s=80, 
        edgecolor='k'
    )
    ax.set_xlabel(feature_names[f1_idx])
    ax.set_ylabel(feature_names[f2_idx])
    ax.set_title(f'{feature_names[f1_idx]} vs {feature_names[f2_idx]}', fontsize=10)
    ax.grid(True, alpha=0.3)

ax_tsne_2d = axes[6]
ax_tsne_2d.scatter(
    X_tsne_2d[:, 0], 
    X_tsne_2d[:, 1], 
    c=y_iris, 
    cmap=cmap, 
    s=80, 
    edgecolor='k'
)
ax_tsne_2d.set_title(f'2D t-SNE Projection\nT: {trust_2d:.3f}, C: {cont_2d:.3f} (k={k})', fontsize=10)
ax_tsne_2d.set_xlabel('t-SNE 1')
ax_tsne_2d.set_ylabel('t-SNE 2')
ax_tsne_2d.grid(True, alpha=0.3)

axes[7].axis('off')

fig.legend(
    handles=legend_handles, 
    title="Species", 
    loc='lower center', 
    ncol=3,
    fontsize=12,
    title_fontsize=14,
    bbox_to_anchor=(0.5, -0.05)
)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('output/tsne_iris_2d_comparison.png')
plt.show()

#----------------------------------------------------
# 1D VISUALIZATION: 1D Feature Visualizations (4 plots) + 1D t-SNE (1 plot)
#----------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Iris Dataset: 1D Feature Visualizations vs. 1D t-SNE', fontsize=18, y=1.02)
axes = axes.flatten()

iris_df = pd.DataFrame(X_iris, columns=feature_names)
iris_df['species'] = data_iris.target_names[y_iris]

for i in range(X_iris.shape[1]):
    ax = axes[i]
    sns.stripplot(
        data=iris_df, 
        x=feature_names[i], 
        y='species', 
        hue='species', 
        palette=colors, 
        size=8, 
        jitter=True, 
        ax=ax,
        legend=False
    )
    ax.set_title(f'Distribution: {feature_names[i]}', fontsize=10)
    ax.set_ylabel('')
    ax.grid(True, axis='x', alpha=0.3)
    ax.set_xlabel(feature_names[i])

ax_tsne_1d = axes[4]
tsne_df = pd.DataFrame({
    't-SNE 1': X_tsne_1d.flatten(),
    'species': data_iris.target_names[y_iris]
})

sns.stripplot(
    data=tsne_df, 
    x='t-SNE 1', 
    y='species', 
    hue='species', 
    palette=colors, 
    size=8, 
    jitter=True, 
    ax=ax_tsne_1d,
    legend=False
)
ax_tsne_1d.set_title(f'1D t-SNE Projection\nT: {trust_1d:.3f}, C: {cont_1d:.3f} (k={k})', fontsize=10)
ax_tsne_1d.set_xlabel('t-SNE Component 1')
ax_tsne_1d.set_ylabel('')
ax_tsne_1d.grid(True, axis='x', alpha=0.3)

axes[5].axis('off')

fig.legend(
    handles=legend_handles, 
    title="Species", 
    loc='lower center', 
    ncol=3,
    fontsize=12,
    title_fontsize=14,
    bbox_to_anchor=(0.5, 0)
)
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig('output/tsne_iris_1d_comparison.png')
plt.show()