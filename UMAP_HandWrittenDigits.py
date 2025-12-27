import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler
from umap import UMAP
import os
import warnings

# ----------------------------------------------
# 1. Configuration and Setup
# ----------------------------------------------
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

os.makedirs('output', exist_ok=True)

# ----------------------------------------------
# 2. Visualization Functions
# ----------------------------------------------

def plot_images(flattened_data, targets, num_images, feature_dim, title_prefix, filename):
    
    rows = 4
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(8, 10),
                             subplot_kw={'xticks':[], 'yticks':[]},
                             gridspec_kw=dict(hspace=0.2, wspace=0.2))
    
    fig.suptitle(f'{title_prefix} MNIST Images ({feature_dim} x {feature_dim} Pixels)', fontsize=14)
    
    for i, ax in enumerate(axes.flat):
        if i < num_images:
            image = flattened_data[i].reshape(feature_dim, feature_dim)
            ax.imshow(image, cmap='gray', interpolation='nearest')
            ax.set_title(f'Label: {targets[i]}', fontsize=10)
        else:
            fig.delaxes(ax)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filename)
    plt.show()
    plt.close()

def plot_2d_embedding(X_embedded, targets, num_samples, title_prefix, filename):
    
    plt.figure(figsize=(10, 8))
    
    scatter = plt.scatter(X_embedded[:num_samples, 0], X_embedded[:num_samples, 1], 
                          c=targets[:num_samples].astype(int),
                          cmap=plt.colormaps['Spectral'], 
                          alpha=0.8, s=20)
    
    plt.colorbar(scatter, ticks=range(10), label='True Digit Label', boundaries=np.arange(11)-0.5)
    
    plt.title(f'{title_prefix}: 2D UMAP Embedding of MNIST Digits', fontsize=14)
    plt.xlabel('UMAP Feature 1')
    plt.ylabel('UMAP Feature 2')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    plt.close()

# ----------------------------------------------
# 3. Data Loading and Preprocessing
# ----------------------------------------------

NUM_SAMPLES = 1000
PLOT_SAMPLES_2D = 1000
PLOT_SAMPLES_GRID = 10
ORIGINAL_DIM_SIZE = 28

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X_mnist = mnist.data.astype(np.float32)[:NUM_SAMPLES]
y_mnist = mnist.target[:NUM_SAMPLES]

# --- Select One of Each Digit (0-9) and build arrays in order ---
unique_digits_str = [str(i) for i in range(10)]
X_unique_list = []
y_unique_list = []

for digit in unique_digits_str:
    index = np.where(y_mnist == digit)[0][0]
    
    X_unique_list.append(X_mnist[index])
    y_unique_list.append(y_mnist[index])

X_unique = np.array(X_unique_list)
y_unique = np.array(y_unique_list)

# --- Scale Data (All 1000 samples for UMAP) ---
scaler = MinMaxScaler()
X_scaled_all = scaler.fit_transform(X_mnist)

# Scale the 10 unique samples for plotting
X_unique_scaled = scaler.transform(X_unique) 

# ----------------------------------------------
# 4. Visualization of Original Images
# ----------------------------------------------

plot_images(
    flattened_data=X_unique_scaled,
    targets=y_unique,
    num_images=PLOT_SAMPLES_GRID,
    feature_dim=ORIGINAL_DIM_SIZE,
    title_prefix='Original MNIST (One of Each Digit, Sorted)',
    filename='output/UMAP_original_28x28_mnist_unique_sorted.png'
)

# ----------------------------------------------
# 5. UMAP Execution
# ----------------------------------------------

TARGET_FEATURES = 2

umap_reducer = UMAP(
    n_components=TARGET_FEATURES,
    random_state=42, 
    n_neighbors=15, 
    min_dist=0.1,
    metric='euclidean'
)
X_embedded = umap_reducer.fit_transform(X_scaled_all)

# ----------------------------------------------
# 6. Visualization of UMAP Embedding
# ----------------------------------------------

plot_2d_embedding(
    X_embedded=X_embedded,
    targets=y_mnist,
    num_samples=PLOT_SAMPLES_2D,
    title_prefix='UMAP Visualization',
    filename='output/UMAP_2d_visualization_mnist.png'
)