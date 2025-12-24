import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import itertools
import warnings
import os
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_wine

warnings.filterwarnings('ignore')
os.makedirs('output', exist_ok=True)

#===========================================================================================
# CUSTOM ELBOW DETECTION FUNCTION
#===========================================================================================
def find_elbow_epsilon_custom(k_distances_sorted):
    n_points = len(k_distances_sorted)
    
    if n_points < 3:
        return k_distances_sorted[-1] if n_points > 0 else 0.1 

    x = np.arange(n_points)
    y = k_distances_sorted
    
    p1 = np.array([x[0], y[0]])
    p2 = np.array([x[-1], y[-1]])
    
    line_vec = p2 - p1
    line_length_sq = np.sum(line_vec**2)
    
    if line_length_sq == 0:
        return y[-1]

    max_distance = -1
    elbow_y = y[-1] 

    for i in range(1, n_points - 1):
        p0 = np.array([x[i], y[i]])
        p1_to_p0_vec = p0 - p1
        t = np.dot(p1_to_p0_vec, line_vec) / line_length_sq
        closest_point = p1 + t * line_vec
        distance = np.linalg.norm(p0 - closest_point)
        
        if distance > max_distance:
            max_distance = distance
            elbow_y = y[i] 

    return elbow_y

#===========================================================================================
# CSV SAVING FUNCTION
#===========================================================================================
def save_ranking_to_csv(results_df, metric_col, filename, ascending=False):
    if ascending:
        filtered_df = results_df[results_df[metric_col] < np.inf].copy()
    else:
        filtered_df = results_df[results_df[metric_col] > -2.0].copy()
    ranking_df = filtered_df.sort_values(by=metric_col, ascending=ascending).copy()
    ranking_df['rank'] = np.arange(1, len(ranking_df) + 1)
    filepath = os.path.join('output', filename)
    ranking_df.to_csv(filepath, index=False)
    print(f"CSV file saved successfully to {filepath}")
    return ranking_df

#===========================================================================================
# COMPLEX TUNING FUNCTION WITH DB
#===========================================================================================
def run_dbscan_full_tuning(X_scaled, min_samples_range):
    results = []
    
    best_sil_score = -2.0
    best_sil_params = {'eps': 0.1, 'min_samples': min_samples_range[0]} 
    
    best_ch_score = -2.0
    best_ch_params = {'eps': 0.1, 'min_samples': min_samples_range[0]}
    
    best_db_score = np.inf
    best_db_params = {'eps': 0.1, 'min_samples': min_samples_range[0]}
    
    def calculate_k_distance_data(X_scaled, fixed_min_samples):
        k_distance_k = max(1, fixed_min_samples - 1) 
        neigh = NearestNeighbors(n_neighbors=k_distance_k)
        neigh.fit(X_scaled)
        distances, _ = neigh.kneighbors(X_scaled)
        k_distances_sorted = np.sort(distances[:, k_distance_k - 1], axis=0)
        return k_distances_sorted, k_distance_k
    
    elbows = []
    pre_min_samples = [5, 10, 15]
    for m in pre_min_samples:
        k_distances_sorted, k = calculate_k_distance_data(X_scaled, m)
        elbow_eps = find_elbow_epsilon_custom(k_distances_sorted)
        elbows.append(elbow_eps)
    avg_elbow = np.mean(elbows)
    eps_range = np.arange(max(0.1, avg_elbow * 0.5), avg_elbow * 1.5 + 0.1, 0.1)
    
    print(f"Dynamic eps_range based on pre-elbows: {eps_range}")
    print(f"Starting Exhaustive Triple-Metric Grid Search...")

    param_grid = list(itertools.product(eps_range, min_samples_range))
    found_any_valid = False

    for eps, min_samples in param_grid:
        if min_samples < 1: continue 
             
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_scaled)
        labels = db.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        sil_score = -1.0
        ch_score = -1.0
        db_score = np.inf
        
        if n_clusters >= 2 and len(set(labels)) > 1:
            try:
                sil_score = silhouette_score(X_scaled, labels)
                ch_score = calinski_harabasz_score(X_scaled, labels)
                db_score = davies_bouldin_score(X_scaled, labels)
                found_any_valid = True
            except Exception:
                pass
        
        results.append({
            'eps': eps,
            'min_samples': min_samples,
            'n_clusters': n_clusters,
            'silhouette_score': sil_score,
            'calinski_harabasz_score': ch_score,
            'davies_bouldin_score': db_score,
        })
        
        if sil_score > best_sil_score:
            best_sil_score = sil_score
            best_sil_params['eps'] = eps
            best_sil_params['min_samples'] = min_samples
            
        if ch_score > best_ch_score:
            best_ch_score = ch_score
            best_ch_params['eps'] = eps
            best_ch_params['min_samples'] = min_samples
            
        if db_score < best_db_score:
            best_db_score = db_score
            best_db_params['eps'] = eps
            best_db_params['min_samples'] = min_samples

    if not found_any_valid:
        print("\nWARNING: No valid clusters found. Defaulting to first parameter set.")

    results_df = pd.DataFrame(results)

    opt_sil = {
        'metric': 'Silhouette Max',
        'eps': best_sil_params['eps'],
        'min_samples': best_sil_params['min_samples'],
        'base_score': best_sil_score
    }
    
    opt_ch = {
        'metric': 'Calinski-Harabasz Max',
        'eps': best_ch_params['eps'],
        'min_samples': best_ch_params['min_samples'],
        'base_score': best_ch_score
    }
    
    opt_db = {
        'metric': 'Davies-Bouldin Min',
        'eps': best_db_params['eps'],
        'min_samples': best_db_params['min_samples'],
        'base_score': best_db_score
    }
    
    print(f"\nMax Silhouette Score ({best_sil_score:.4f}) found at $\\epsilon$={opt_sil['eps']:.3f}, M={opt_sil['min_samples']}.")
    print(f"Max Calinski-Harabasz Score ({best_ch_score:.1f}) found at $\\epsilon$={opt_ch['eps']:.3f}, M={opt_ch['min_samples']}.")
    print(f"Min Davies-Bouldin Score ({best_db_score:.4f}) found at $\\epsilon$={opt_db['eps']:.3f}, M={opt_db['min_samples']}.");

    k_dist_sil, k_sil = calculate_k_distance_data(X_scaled, opt_sil['min_samples'])
    eps_elbow_sil = find_elbow_epsilon_custom(k_dist_sil)
    if eps_elbow_sil <= 0: eps_elbow_sil = opt_sil['eps']

    opt_elbow_sil = {
        'metric': 'Elbow (Sil M)',
        'eps': eps_elbow_sil,
        'min_samples': opt_sil['min_samples'],
        'base_score': -1
    }
    
    k_dist_ch, k_ch = calculate_k_distance_data(X_scaled, opt_ch['min_samples'])
    eps_elbow_ch = find_elbow_epsilon_custom(k_dist_ch)
    if eps_elbow_ch <= 0: eps_elbow_ch = opt_ch['eps']

    opt_elbow_ch = {
        'metric': 'Elbow (CH M)',
        'eps': eps_elbow_ch,
        'min_samples': opt_ch['min_samples'],
        'base_score': -1
    }
    
    k_dist_db, k_db = calculate_k_distance_data(X_scaled, opt_db['min_samples'])
    eps_elbow_db = find_elbow_epsilon_custom(k_dist_db)
    if eps_elbow_db <= 0: eps_elbow_db = opt_db['eps']

    opt_elbow_db = {
        'metric': 'Elbow (DB M)',
        'eps': eps_elbow_db,
        'min_samples': opt_db['min_samples'],
        'base_score': -1
    }
    
    print(f"Custom Elbow Heuristic found $\\epsilon$={eps_elbow_sil:.3f} using M={opt_sil['min_samples']}.")
    print(f"Custom Elbow Heuristic found $\\epsilon$={eps_elbow_ch:.3f} using M={opt_ch['min_samples']}.")
    print(f"Custom Elbow Heuristic found $\\epsilon$={eps_elbow_db:.3f} using M={opt_db['min_samples']}.")

    all_eps = [opt_sil['eps'], opt_ch['eps'], opt_db['eps'], opt_elbow_sil['eps'], opt_elbow_ch['eps'], opt_elbow_db['eps']]
    avg_eps = np.mean(all_eps)
    
    opt_final = {
        'metric': 'Final Consensus (Avg \u03B5)', 
        'eps': avg_eps,
        'min_samples': opt_sil['min_samples'], 
        'base_score': -1
    }
    
    all_combos = [opt_sil, opt_ch, opt_db, opt_elbow_sil, opt_elbow_ch, opt_elbow_db, opt_final]
    
    db_final = DBSCAN(eps=opt_final['eps'], min_samples=opt_final['min_samples']).fit(X_scaled)
    labels_final = db_final.labels_
    n_clusters_final = len(set(labels_final)) - (1 if -1 in labels_final else 0)
    
    if n_clusters_final >= 2 and len(set(labels_final)) > 1:
        opt_final['base_score_sil'] = silhouette_score(X_scaled, labels_final)
        opt_final['base_score_ch'] = calinski_harabasz_score(X_scaled, labels_final)
        opt_final['base_score_db'] = davies_bouldin_score(X_scaled, labels_final)
    else:
        opt_final['base_score_sil'] = -1
        opt_final['base_score_ch'] = -1
        opt_final['base_score_db'] = np.inf
        
    print(f"\nFinal Consensus $\\epsilon$={opt_final['eps']:.3f} using M={opt_final['min_samples']}.")
    
    return all_combos, results_df, k_dist_sil, k_sil, k_dist_ch, k_ch, k_dist_db, k_db, calculate_k_distance_data

#===========================================================================================
# MAIN EXECUTION BLOCK
#===========================================================================================
dataset_path = 'Datasets/wine-clustering.csv'
user_path = input("Enter CSV file location (default: Datasets/wine-clustering.csv): ").strip()
if user_path:
    dataset_path = user_path

try:
    dataset = pd.read_csv(dataset_path)
    all_columns = dataset.columns.tolist()
    print(f"Available columns: {all_columns}")
    
    exclude_input = input("Enter columns to exclude from features (comma-separated, e.g., 'col1','col2'): ").strip()
    
    if exclude_input:
        exclude_cols = [col.strip().strip("'\"") for col in exclude_input.split(',')]
        feature_cols = [col for col in all_columns if col not in exclude_cols]
    else:
        feature_cols = all_columns
    
    print(f"Using columns as features: {feature_cols}")
    X = dataset[feature_cols].values
except FileNotFoundError:
    print(f"\n*** File not found. Falling back to sklearn load_wine ***\n")
    data = load_wine()
    dataset = pd.DataFrame(data.data, columns=data.feature_names)
    X = dataset.values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca_full = PCA().fit(X_scaled)
explained_variance_ratios = pca_full.explained_variance_ratio_

pca_3d = PCA(n_components=3)
X_pca = pca_3d.fit_transform(X_scaled)

min_samples_search_range = np.arange(5, 20, 1)

all_combos, results_df, k_dist_sil, k_sil, k_dist_ch, k_ch, k_dist_db, k_db, calculate_k_distance_data = run_dbscan_full_tuning(
    X_scaled, 
    min_samples_search_range
)

combos_df = pd.DataFrame(all_combos)
combos_df = combos_df.set_index('metric')

sil_ranking_df = save_ranking_to_csv(results_df, 'silhouette_score', 'dbscan_wine_silhouette_ranking.csv', ascending=False)
ch_ranking_df = save_ranking_to_csv(results_df, 'calinski_harabasz_score', 'dbscan_wine_calinski_harabasz_ranking.csv', ascending=False)
db_ranking_df = save_ranking_to_csv(results_df, 'davies_bouldin_score', 'dbscan_wine_davies_bouldin_ranking.csv', ascending=True)

print("\n\n###################################################################")
print("                  SIX-COMBO DBSCAN TUNING REPORT")
print("###################################################################")
print("\n---> SIX OPTIMAL COMBINATIONS:")
print(combos_df[['eps', 'min_samples']].round(4))

#===========================================================================================
# BAR PLOTS: ALL COMBOS SIDE-BY-SIDE
#===========================================================================================
sil_full = sil_ranking_df.copy()
ch_full = ch_ranking_df.copy()
db_full = db_ranking_df.copy()

sil_full['params'] = sil_full.apply(lambda row: f"E={row['eps']:.2f}, M={int(row['min_samples'])}, R={int(row['rank'])}", axis=1)
ch_full['params'] = ch_full.apply(lambda row: f"E={row['eps']:.2f}, M={int(row['min_samples'])}, R={int(row['rank'])}", axis=1)
db_full['params'] = db_full.apply(lambda row: f"E={row['eps']:.2f}, M={int(row['min_samples'])}, R={int(row['rank'])}", axis=1)

fig, axes = plt.subplots(1, 3, figsize=(30, 40)) 

sns.barplot(data=sil_full, y='params', x='silhouette_score', color='blue', ax=axes[0])
axes[0].set_title(f'All {len(sil_full)} DBSCAN Combos Ranked by Silhouette Score (Higher Better)')
axes[0].set_xlabel('Silhouette Score')
axes[0].set_ylabel('Parameters (Epsilon, Min_Samples, Rank)')
axes[0].tick_params(axis='y', labelsize=6) 

sns.barplot(data=ch_full, y='params', x='calinski_harabasz_score', color='orange', ax=axes[1])
axes[1].set_title(f'All {len(ch_full)} DBSCAN Combos Ranked by Calinski-Harabasz Score (Higher Better)')
axes[1].set_xlabel('Calinski-Harabasz Score')
axes[1].set_ylabel('Parameters (Epsilon, Min_Samples, Rank)')
axes[1].tick_params(axis='y', labelsize=6) 

sns.barplot(data=db_full, y='params', x='davies_bouldin_score', color='green', ax=axes[2])
axes[2].set_title(f'All {len(db_full)} DBSCAN Combos Ranked by Davies-Bouldin Score (Lower Better)')
axes[2].set_xlabel('Davies-Bouldin Score')
axes[2].set_ylabel('Parameters (Epsilon, Min_Samples, Rank)')
axes[2].tick_params(axis='y', labelsize=6) 

plt.tight_layout()
plt.savefig('output/dbscan_all_combo_ranking_barplots.png')
plt.show()  
plt.close()

#===========================================================================================
# HEATMAP GENERATION FUNCTION
#===========================================================================================
def generate_triple_heatmaps(heatmap_sil, heatmap_ch, heatmap_db, combo_indices, combo_map, filename_suffix):
    fig, axes = plt.subplots(1, 3, figsize=(30, 8))

    sns.heatmap(heatmap_sil, cmap="viridis", linewidths=0.5, linecolor='white', cbar_kws={'label': 'Silhouette Score (Max 1.0)'}, ax=axes[0])
    axes[0].set_title('DBSCAN Performance: Silhouette Score')
    axes[0].set_xlabel('Epsilon (ε)')
    axes[0].set_ylabel('Min_Samples')

    sns.heatmap(heatmap_ch, cmap="plasma", linewidths=0.5, linecolor='white', cbar_kws={'label': 'Calinski-Harabasz Score (Higher is Better)'}, ax=axes[1])
    axes[1].set_title('DBSCAN Performance: Calinski-Harabasz Score')
    axes[1].set_xlabel('Epsilon (ε)')
    axes[1].set_ylabel('Min_Samples')

    sns.heatmap(heatmap_db, cmap="inferno", linewidths=0.5, linecolor='white', cbar_kws={'label': 'Davies-Bouldin Score (Lower is Better)'}, ax=axes[2])
    axes[2].set_title('DBSCAN Performance: Davies-Bouldin Score')
    axes[2].set_xlabel('Epsilon (ε)')
    axes[2].set_ylabel('Min_Samples')

    eps_labels = [f'{e:.2f}' for e in heatmap_sil.columns]
    for ax in axes:
        ax.set_xticklabels(eps_labels, rotation=90)

    handles = []
    legend_labels = []
    for name, indices in combo_indices.items():
        props = combo_map.get(name)
        if not props: continue 
        
        for ax in axes:
            ax.add_patch(plt.Rectangle((indices['eps_idx'], indices['min_samples_idx']), 1, 1, fill=False, edgecolor=props['color'], linestyle=props['style'], linewidth=3))
        
        handles.append(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor=props['color'], linestyle=props['style'], linewidth=3)) 
        legend_labels.append(props['label'])

    fig.legend(handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=4, title="Optimal Combos", fontsize=10, framealpha=0.95)

    plt.tight_layout(rect=[0, 0.03, 1, 1]) 
    plt.savefig(f'output/dbscan_tuning_triple_heatmaps{filename_suffix}.png', bbox_inches='tight')
    plt.show()  
    plt.close()

#===========================================================================================
# TRIPLE HEATMAPS: OUTLINES ADDED, LEGEND FIXED
#===========================================================================================
heatmap_sil = results_df.pivot(index='min_samples', columns='eps', values='silhouette_score')
heatmap_ch = results_df.pivot(index='min_samples', columns='eps', values='calinski_harabasz_score')
heatmap_db = results_df.pivot(index='min_samples', columns='eps', values='davies_bouldin_score')

def get_idx(val, list_):
    return np.abs(np.array(list_) - val).argmin()

combo_indices = {}
for name, row in combos_df.iterrows():
    combo_indices[name] = {
        'eps_idx': get_idx(row['eps'], heatmap_sil.columns.tolist()),
        'min_samples_idx': get_idx(row['min_samples'], heatmap_sil.index.tolist())
    }

combo_map = {
    'Silhouette Max': {'color': 'red', 'style': '-', 'label': f"Sil Max ($\epsilon$={combos_df.loc['Silhouette Max', 'eps']:.2f}, M={int(combos_df.loc['Silhouette Max', 'min_samples'])})"},
    'Calinski-Harabasz Max': {'color': 'blue', 'style': '-', 'label': f"CH Max ($\epsilon$={combos_df.loc['Calinski-Harabasz Max', 'eps']:.2f}, M={int(combos_df.loc['Calinski-Harabasz Max', 'min_samples'])})"},
    'Davies-Bouldin Min': {'color': 'purple', 'style': '-', 'label': f"DB Min ($\epsilon$={combos_df.loc['Davies-Bouldin Min', 'eps']:.2f}, M={int(combos_df.loc['Davies-Bouldin Min', 'min_samples'])})"},
    'Elbow (Sil M)': {'color': 'orange', 'style': '--', 'label': f"Elbow (Sil M) ($\epsilon$={combos_df.loc['Elbow (Sil M)', 'eps']:.2f}, M={int(combos_df.loc['Elbow (Sil M)', 'min_samples'])})"},
    'Elbow (CH M)': {'color': 'green', 'style': '--', 'label': f"Elbow (CH M) ($\epsilon$={combos_df.loc['Elbow (CH M)', 'eps']:.2f}, M={int(combos_df.loc['Elbow (CH M)', 'min_samples'])})"},
    'Elbow (DB M)': {'color': 'cyan', 'style': '--', 'label': f"Elbow (DB M) ($\epsilon$={combos_df.loc['Elbow (DB M)', 'eps']:.2f}, M={int(combos_df.loc['Elbow (DB M)', 'min_samples'])})"},
    'Final Consensus (Avg \u03B5)': {'color': 'yellow', 'style': ':', 'label': f"Final Consensus (Avg $\epsilon$) ($\epsilon$={combos_df.loc['Final Consensus (Avg \u03B5)', 'eps']:.2f}, M={int(combos_df.loc['Final Consensus (Avg \u03B5)', 'min_samples'])})"}
}

print("\n=== TRIPLE HEATMAPS VISUALIZATION ===")
print("Heatmap dimensions and combo positions:")
for name, indices in combo_indices.items():
    props = combo_map.get(name)
    if props:
        print(f"  {name}: eps_idx={indices['eps_idx']}, min_samples_idx={indices['min_samples_idx']}")

generate_triple_heatmaps(heatmap_sil, heatmap_ch, heatmap_db, combo_indices, combo_map, "_6_combos")
print("✓ Initial triple heatmaps visualization saved")

#===========================================================================================
# CUSTOM COMBO INPUT
#===========================================================================================
print("\n=== CUSTOM COMBO SECTION ===")
custom_input = input("Enter your custom combo as (eps, min_samples), e.g. (1.34, 5), or press Enter to skip: ").strip()

custom_combo_added = False
if custom_input:
    try:
        clean_input = custom_input.strip('() ').replace(' ', '')
        custom_eps_str, custom_m_str = clean_input.split(',')
        custom_eps = float(custom_eps_str)
        custom_m = int(custom_m_str)
        custom_combo_added = True
        
        custom_combo = {'metric': 'Custom Combo', 'eps': custom_eps, 'min_samples': custom_m, 'base_score': -1}
        
        combos_df = combos_df.reset_index()
        combos_df = pd.concat([combos_df, pd.DataFrame([custom_combo])], ignore_index=True)
        combos_df = combos_df.set_index('metric')
        
        combo_map['Custom Combo'] = {'color': 'magenta', 'style': '-.', 'label': f"Custom ($\epsilon$={custom_eps:.2f}, M={custom_m})"}
        
        combo_indices['Custom Combo'] = {
            'eps_idx': get_idx(custom_eps, heatmap_sil.columns.tolist()),
            'min_samples_idx': get_idx(custom_m, heatmap_sil.index.tolist())
        }
        
        print(f"Custom combo added: eps={custom_eps:.3f}, min_samples={custom_m}")
        print(f"  Position in heatmap: eps_idx={combo_indices['Custom Combo']['eps_idx']}, min_samples_idx={combo_indices['Custom Combo']['min_samples_idx']}")
        
        generate_triple_heatmaps(heatmap_sil, heatmap_ch, heatmap_db, combo_indices, combo_map, "")
        print("✓ Updated triple heatmaps with custom combo saved")
    except Exception as e:
        print("Invalid format. Skipping custom combo.")
        custom_combo_added = False
else:
    print("No custom combo provided. Continuing with original 7 combos.")

#===========================================================================================
# K-DISTANCE ELBOW PLOT
#===========================================================================================
print("\n=== K-DISTANCE ELBOW PLOT ===")
plt.figure(figsize=(10, 6))
plt.plot(k_dist_sil, label=f'k-Distance (k={k_sil}) Sil Base', color='gray', linestyle='-')
plt.plot(k_dist_ch, label=f'k-Distance (k={k_ch}) CH Base', color='black', linestyle='--')
plt.plot(k_dist_db, label=f'k-Distance (k={k_db}) DB Base', color='darkgray', linestyle=':')

print(f"K-distance plots:")
print(f"  Silhouette: k={k_sil}, min distance={k_dist_sil.min():.4f}, max distance={k_dist_sil.max():.4f}")
print(f"  Calinski-Harabasz: k={k_ch}, min distance={k_dist_ch.min():.4f}, max distance={k_dist_ch.max():.4f}")
print(f"  Davies-Bouldin: k={k_db}, min distance={k_dist_db.min():.4f}, max distance={k_dist_db.max():.4f}")

if custom_combo_added:
    k_dist_custom, k_custom = calculate_k_distance_data(X_scaled, custom_m)
    plt.plot(k_dist_custom, label=f'k-Distance (k={k_custom}) Custom Base', color='magenta', linestyle='-.')
    print(f"  Custom: k={k_custom}, min distance={k_dist_custom.min():.4f}, max distance={k_dist_custom.max():.4f}")

print("\nEpsilon threshold lines:")
for name, props in combo_map.items():
    if name in combos_df.index:
        eps_val = combos_df.loc[name, 'eps']
        plt.axhline(y=eps_val, color=props['color'], linestyle=props['style'], linewidth=2, label=props['label'])
        print(f"  {name}: ε={eps_val:.4f}")

plt.xlabel('Points Sorted by Distance')
plt.ylabel('Distance to Nearest Neighbor')
plt.title('DBSCAN K-Distance Graphs with Optimal Epsilon Lines')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('output/dbscan_k_distance_elbow_plot_final.png')
print("✓ K-Distance elbow plot saved")
plt.show()  
plt.close()

#===========================================================================================
# EXPLAINED VARIANCE PLOT (SCREE PLOT)
#===========================================================================================
print("\n=== PCA SCREE PLOT ===")
cumulative_explained_variance = np.cumsum(explained_variance_ratios)
n_components = len(explained_variance_ratios)
components_labels = np.arange(1, n_components + 1)

plt.figure(figsize=(12, 7))

plt.bar(components_labels, explained_variance_ratios, alpha=0.4, align='center', 
        label='Individual Explained Variance', color='steelblue', width=0.6)

plt.plot(components_labels, cumulative_explained_variance, 
         marker='o', markersize=6, linewidth=3, color='darkred', 
         label='Cumulative Explained Variance', zorder=5)

target_pcs = [3, 5, 10]
colors_map = {3: 'green', 5: 'dodgerblue', 10: 'darkorange'}

print("Individual and Cumulative Explained Variance:")
print(f"{'PC':<5} {'Individual':<15} {'Cumulative':<15}")
print("-" * 35)
for pc in components_labels:
    ind_var = explained_variance_ratios[pc - 1]
    cum_var = cumulative_explained_variance[pc - 1]
    marker = " *" if pc in target_pcs else ""
    print(f"{pc:<5} {ind_var:>13.1%} {cum_var:>13.1%}{marker}")

for pc in target_pcs:
    if pc <= len(cumulative_explained_variance):
        cum_var = cumulative_explained_variance[pc - 1]
        
        plt.vlines(pc, 0, cum_var, linestyle='--', color=colors_map[pc], linewidth=2.5, alpha=0.7)
        plt.scatter([pc], [cum_var], s=150, color=colors_map[pc], zorder=6, 
                   edgecolors='black', linewidth=2, marker='D')
        plt.text(pc + 0.15, cum_var + 0.025, f'PCA-{pc}\n{cum_var:.1%}', 
                fontsize=11, color=colors_map[pc], fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor=colors_map[pc], linewidth=1.5))

plt.ylabel('Explained Variance Ratio', fontsize=12, fontweight='bold')
plt.xlabel('Principal Component Index', fontsize=12, fontweight='bold')
plt.title('PCA Cumulative Explained Variance (Scree Plot)', fontsize=14, fontweight='bold', pad=20)
plt.legend(loc='lower right', fontsize=11, framealpha=0.95)
plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)
plt.xticks(components_labels, fontsize=10)
plt.yticks(fontsize=10)
plt.ylim([0, 1.05])

plt.tight_layout()
plt.savefig('output/dbscan_pca_cumulative_explained_variance_scree_plot.png', dpi=300)
print("\n✓ PCA Scree plot saved")
plt.show() 
plt.close()

#===========================================================================================
# PCA ANALYSIS AND CLUSTER VISUALIZATION
#===========================================================================================
print("\n=== PCA 3D CLUSTER VISUALIZATION ===")
n_combos = len(combos_df)
n_cols = min(3, n_combos)
n_rows = (n_combos + n_cols - 1) // n_cols
fig_width = 14 * n_cols
fig_height = 6 * n_rows

fig = plt.figure(figsize=(fig_width, fig_height))

print(f"\nCluster Results ({n_rows}x{n_cols} grid):")
print(f"{'Combo':<35} {'Clusters':<10} {'Noise Points':<15}")
print("-" * 60)

for i, (name, params) in enumerate(combos_df.iterrows()):
    
    db = DBSCAN(eps=params['eps'], min_samples=int(params['min_samples'])).fit(X_scaled)
    labels = db.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    ax = fig.add_subplot(n_rows, n_cols, i + 1, projection='3d')

    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], 
                         c=labels, cmap='Spectral', alpha=0.8, s=50)

    label_map = {k: f"Cluster {k+1}" for k in range(n_clusters)}
    label_map[-1] = "Noise (-1)"
    unique_labels = sorted(set(labels))
    legend_labels = [label_map[l] for l in unique_labels]

    plot_handles = scatter.legend_elements()[0]
    legend1 = ax.legend(plot_handles, 
                         legend_labels, 
                         loc="lower left", title="Clusters", ncol=1)
    ax.add_artist(legend1)

    title_text = f'{name}\n$\\epsilon$={params["eps"]:.3f}, M={int(params["min_samples"])}\nClusters: {n_clusters}, Noise: {n_noise}'
    ax.set_title(title_text, fontsize=10, fontweight='bold', pad=20)
    ax.set_xlabel(f'PC 1 ({explained_variance_ratios[0]*100:.1f}%)')
    ax.set_ylabel(f'PC 2 ({explained_variance_ratios[1]*100:.1f}%)')
    ax.set_zlabel(f'PC 3 ({explained_variance_ratios[2]*100:.1f}%)') 
    
    print(f"{name:<35} {n_clusters:<10} {n_noise:<15}")

plt.tight_layout()
plt.savefig('output/dbscan_pca_3d_clusters_comparison_final.png')
print("\n✓ PCA 3D cluster visualization saved")
plt.show()  
plt.close()

print("\n" + "="*60)
print("Script completed successfully. All data analysis and visualization steps are included.")
print("="*60)