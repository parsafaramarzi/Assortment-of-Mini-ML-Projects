import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import seaborn as sns
import numpy as np

dataset = pd.read_csv('Datasets/Mall_Customers.csv') 

X = dataset.iloc[:, [2, 3, 4]].values
df_features = dataset.iloc[:, [2, 3, 4]]
df_features.columns = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
df_features.index = dataset['CustomerID']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
df_scaled = pd.DataFrame(X_scaled, columns=df_features.columns, index=df_features.index)

DIST_METRICS = ['euclidean', 'cityblock', 'cosine']
LINKAGE_METHODS_FOR_SEARCH = ['ward', 'single', 'complete', 'average'] 
LINKAGE_METHODS_FOR_GRID = ['single', 'complete', 'average'] 
N_CLUSTERS_RANGE = range(2, 11)


def find_optimal_clusters(data_matrix, k_range, dist_metrics, linkage_methods):
    score_data = []
    best_score = -1
    best_params = {}
    
    for k in k_range:
        for metric in dist_metrics:
            for linkage_method in linkage_methods:
                
                if linkage_method == 'ward' and metric != 'euclidean':
                    continue

                try:
                    model = AgglomerativeClustering(n_clusters=k, metric=metric, linkage=linkage_method)
                    labels = model.fit_predict(data_matrix)
                    
                    score = silhouette_score(data_matrix, labels, metric=metric)
                    
                    display_metric = 'Manhattan' if metric == 'cityblock' else metric.title()
                    display_linkage = linkage_method.title()
                    
                    score_data.append({
                        'Clusters (k)': k,
                        'Metric': display_metric,
                        'Linkage': display_linkage,
                        'Configuration': f'{display_metric} + {display_linkage}',
                        'Silhouette Score': score
                    })
                    
                    if score > best_score:
                        best_score = score
                        best_params = {'k': k, 'metric': metric, 'linkage': linkage_method, 'score': score}
                
                except ValueError:
                    pass
                
    
    print(f"\n--- OPTIMAL CHOICE (Found from all valid combinations) ---")
    print(f"  Best Clusters (k): {best_params['k']}")
    best_metric_display = 'Manhattan' if best_params['metric'] == 'cityblock' else best_params['metric'].title()
    print(f"  Best Distance Metric: {best_metric_display}")
    print(f"  Best Linkage Method: {best_params['linkage'].title()}")
    print(f"  Highest Score: {best_params['score']:.4f}")
    
    score_df = pd.DataFrame(score_data)
    return score_df, best_params

score_df, best_params = find_optimal_clusters(X_scaled, N_CLUSTERS_RANGE, DIST_METRICS, LINKAGE_METHODS_FOR_SEARCH)
optimal_k = best_params['k']
optimal_metric = best_params['metric']
optimal_linkage = best_params['linkage']
optimal_metric_display = 'Manhattan' if optimal_metric == 'cityblock' else optimal_metric.title()
optimal_linkage_display = optimal_linkage.title()


plt.figure(figsize=(16, 7))
plot_df = score_df[score_df['Linkage'].isin(['Single', 'Complete', 'Average'])].copy()
plot_df['Hue_Label'] = plot_df['Linkage'] + ' (' + plot_df['Metric'].str[:4] + ')'
barplot = sns.barplot(x='Clusters (k)', y='Silhouette Score', hue='Hue_Label', data=plot_df, palette='Spectral')

optimal_configuration = score_df[(score_df['Clusters (k)'] == optimal_k) & (score_df['Metric'] == optimal_metric_display) & (score_df['Linkage'] == optimal_linkage_display)].iloc[0]

barplot.axhline(optimal_configuration['Silhouette Score'], color='red', linestyle='--', linewidth=1)
barplot.text(2, optimal_configuration['Silhouette Score'] + 0.005, f'Overall Optimal: k={optimal_k} ({optimal_metric_display} + {optimal_linkage_display})', color='red', fontsize=10)

plt.title('Silhouette Score Comparison (9 Configurations)')
plt.ylabel('Silhouette Score (Higher is Better)')
plt.xlabel('Number of Clusters (k)')
plt.legend(title='Linkage (Metric)')
plt.tight_layout()
plt.savefig('output/hc_silhouette_scores_9_configs.png')
plt.show()
plt.close()


fig, axes = plt.subplots(3, 3, figsize=(24, 20))
fig.suptitle('Hierarchical Clustering: Dendrogram Comparison (3 x 3 Grid)', fontsize=18)
axes = axes.ravel() 

plot_index = 0
metric_map = {'euclidean': 'Euclidean', 'cityblock': 'Manhattan', 'cosine': 'Cosine'}
linkage_map = {'single': 'Single', 'complete': 'Complete', 'average': 'Average'}

for row_metric in DIST_METRICS:
    for col_linkage in LINKAGE_METHODS_FOR_GRID:
        ax = axes[plot_index]
        
        sch.dendrogram(sch.linkage(X_scaled, method=col_linkage, metric=row_metric), ax=ax, truncate_mode='none')
        
        title = f'{metric_map[row_metric]} / {linkage_map[col_linkage]}'
        ax.set_title(title, fontsize=12)
        
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min, y_max * 1.05) 
        
        if plot_index % 3 == 0:
            ax.set_ylabel(f'Row Metric: {metric_map[row_metric]}', fontsize=10)
        if plot_index < 3:
            ax.set_xlabel(f'Col Linkage: {linkage_map[col_linkage]}', fontsize=10)
            ax.xaxis.set_label_position('top')
            
        plot_index += 1

plt.tight_layout(rect=[0, 0.03, 1, 0.98])
plt.savefig('output/hc_dendrograms_9_comparison.png')
plt.show()
plt.close(fig)


fig_3d = plt.figure(figsize=(24, 24))
fig_3d.suptitle(f'3D Cluster Comparison (k={optimal_k} 3 x 3 Grid)', fontsize=18)

cmap = plt.colormaps.get_cmap('Spectral')
colors = [cmap(i) for i in np.linspace(0, 1, optimal_k)]

plot_index = 0
for row_metric in DIST_METRICS:
    for col_linkage in LINKAGE_METHODS_FOR_GRID:
        
        hc_model = AgglomerativeClustering(n_clusters=optimal_k, metric=row_metric, linkage=col_linkage)
        labels = hc_model.fit_predict(X_scaled)
        
        ax = fig_3d.add_subplot(3, 3, plot_index + 1, projection='3d')
        
        for j in range(optimal_k):
            ax.scatter(X[labels == j, 1],
                       X[labels == j, 2],
                       X[labels == j, 0],
                       s=30, c=colors[j], alpha=0.8) 

        title = f'{metric_map[row_metric]} / {linkage_map[col_linkage]}'
        ax.set_title(title, fontsize=12)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        if plot_index % 3 == 0:
            ax.set_ylabel(f'Row Metric: {metric_map[row_metric]}', fontsize=10)
        if plot_index < 3:
            ax.set_xlabel(f'Col Linkage: {linkage_map[col_linkage]}', fontsize=10)
            ax.xaxis.set_label_position('top')
        
        plot_index += 1

plt.tight_layout(rect=[0, 0.03, 1, 0.98])
plt.savefig('output/hc_clusters_3d_9_comparison.png')
plt.show()
plt.close(fig_3d)


for row_metric in DIST_METRICS:
    for col_linkage in LINKAGE_METHODS_FOR_GRID:
        
        metric_display = metric_map[row_metric]
        linkage_display = linkage_map[col_linkage]
        
        hc_plot = AgglomerativeClustering(n_clusters=optimal_k, metric=row_metric, linkage=col_linkage)
        hc_plot.fit(X_scaled)

        g = sns.clustermap(df_scaled,
                       method=col_linkage,
                       metric=row_metric,
                       cmap='magma',
                       figsize=(10, 10),
                       dendrogram_ratio=(.1, .3),
                       linecolor='gray',
                       linewidths=0.5,
                       cbar_pos=(0.02, 0.82, 0.03, 0.15))

        fig_heatmap = g.fig 
        title = f'Clustered Heatmap (k={optimal_k}, Metric: {metric_display}, Linkage: {linkage_display})'
        fig_heatmap.suptitle(title, y=1.02)
        
        g.ax_col_dendrogram.text(0.5, 0.85, f'{metric_display} / {linkage_display}', 
                                 transform=g.ax_col_dendrogram.transAxes, 
                                 fontsize=12, 
                                 color='red', 
                                 ha='center', va='center')
        
        filename = f'output/hc_clustered_heatmap_{row_metric}_{col_linkage}_k{optimal_k}.png'
        fig_heatmap.savefig(filename)

        plt.show()
        plt.close(fig_heatmap)