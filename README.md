# Assortment of Mini ML Projects
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Last Updated](https://img.shields.io/badge/last_updated-December_2025-blue)

A growing collection of **mini machine learning projects**, each focused on a key ML algorithm or concept — **regression, classification, clustering, and dimensionality reduction** — using real-world datasets.

Each project is **self-contained**, **beginner-friendly**, and ideal for **hands-on learning** and **portfolio building**.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1hb5CcUgi0QXym9RUFIm1-3BWEvNqPp26?usp=sharing)

---

## Repository Overview
### Folders
- **`Datasets/`** – All datasets used across projects. 
- **`output/`** – Saved plots and models.

---

## Regression Projects

### 1. Linear Regression
| Project | Feature | Target | Goal |
|--------|---------|--------|------|
| `LinearRegression Automobile Dataset.py` | `horsepower` | `price` | Predict car price |
| `LinearRegression Bengaluru Housing Data.py` | `total_sqft` | `price` | Predict house price |
| `LinearRegression Heart Disease Dataset.py` | `bmi` | `chol` | Predict cholesterol from BMI |

<img src="output/linear_reg_automobile_pred.png" width="600" alt="Linear Reg – Horsepower vs Price" />
*Scatter + fitted line (Automobile dataset)*

---

### 2. Polynomial Regression
| Project | Feature | Target | Goal |
|--------|---------|--------|------|
| `Polynomial Regression Automobile Dataset.py` | `horsepower` | `price` | Capture non-linear price trends |
| `Polynomial Regression Bengaluru Housing Data.py` | `total_sqft` | `price` | Model non-linear housing trends |
| `Polynomial Regression Heart Disease Dataset.py` | `bmi` | `chol` | Fit non-linear BMI-cholesterol |

<img src="output/poly_reg_automobile.png" width="600"/>
*Degree-14 polynomial curve on horsepower vs price*

---

### 3. Multiple Linear Regression — *Concrete Compressive Strength*
- **`MultipleLinearRegression Concrete Dataset.py`**
  - **Dataset:** `concrete_data.csv`
  - **Goal:** Predict **concrete compressive strength** from 8 material components.

<img src="output/concrete_corr_heatmap.png" width="600"/>
<img src="output/concrete_feature_scatter.png" width="600"/>
<img src="output/concrete_residuals_before.png" width="600"/>
<img src="output/concrete_residuals.png" width="600"/>

> **Before vs After:** Polynomial features + removing weak predictors (`blast_furnace_slag`, `fly_ash`) 
> **improved R² from 62.8% → 76.3% (+13.5%)** and **reduced MSE from 96 → 61 (-36%)**.

---

## Classification Projects

### 4. Decision Trees (DT)
| Project | Goal |
|--------|------|
| `DT Drug200.py` | Classify drug type |
| `DT diabetes.py` | Predict diabetes |

**Decision Tree**

<img src="output/dt_diabetes_tree.png" width="700"/>

**Feature Importance Ranking** (most → least important) 

<img src="output/dt_diabetes_importance.png" width="500"/>

> **Key insight:** `Glucose` is the dominant splitter, followed by `BMI`, `Age`, etc.

---

### 5. Random Forest (RF)
| Project | Goal |
|--------|------|
| `RF Drug200.py` | Ensemble drug classification |
| `RF diabetes.py` | Ensemble diabetes prediction |

**Feature Importance (100 trees)**

<img src="output/rf_diabetes_importance.png" width="500"/>

> **Top 3:** `Glucose` > `BMI` > `Age` — same pattern as DT, but more stable

---

### 6. Support Vector Machine (SVM)
- **`SVM BreastCancer.py`** – kernels `linear`, `poly`, `rbf`

**Kernel Comparison**

<img src="output/svm_breastcancer_kernels.png" width="600"/>

**2D PCA (Best Model)**

<img src="output/svm_breastcancer_2d.png" width="600"/>

**3D PCA View**

<img src="output/svm_breastcancer_3d.png" width="600"/>

> **Legend:** Blue = Benign | Orange = Malignant 
> **Best:** `SVM Linear` → ~95% accuracy

---

### 7. K-Nearest Neighbors (KNN)
- **`KNN Iris&Customer.py`**

**Iris Classification**

<img src="output/knn_iris.png" width="500"/>

**Customer Gender Prediction**

<img src="output/knn_customer.png" width="500"/>

> **Legend:** Blue = Setosa/Male | Orange = Versicolor/Female | Green = Virginica 
> **Iris:** ~100% | **Gender:** ~60% (challenging overlap)

---

### 8. Logistic Regression — *Bank Customer Churn*
- **`LogisticRegression Bank Customer Churn.py`**

**Confusion Matrix**

<img src="output/logreg_churn_confmat.png" width="500"/>

**ROC Curve**

<img src="output/logreg_churn_roc.png" width="500"/>

**Feature Importance (Coefficients)**

<img src="output/logreg_churn_importance.png" width="600"/>

> **Accuracy:** `81.60%` | **AUC:** `0.744` 
> **Top churn driver:** `country` (strongest positive coefficient) 
> **Strongest protector:** `active_member` (biggest negative coefficient)

---

## Clustering Projects

### 9. K-Means — *Mall Customer Segmentation*
- **`Kmeans Mall Customers.py`**

**Elbow Plot (k = 1–99)** <img src="output/kmeans_elbow.png" width="600"/>

**Customer Clusters (k = 8)** <img src="output/kmeans_clusters.png" width="650"/>

> **Detected optimal k = 8** via automated elbow method 
> 8 distinct customer groups identified 
> Centroids shown as **black X**

### 10. Hierarchical Clustering (HC) — *Mall Customer Segmentation*
- **`HierarchicalClustering Mall Customers.py`**

This project provides a comprehensive analysis of customer segmentation using **Hierarchical Clustering (HC)**. A rigorous hyperparameter search was performed, comparing 3 different distance metrics and 4 linkage methods to determine the optimal clustering configuration.

#### 📌 Optimal Configuration Summary

A systematic search across $k \in [2, 10]$, 3 distance metrics, and 4 linkage methods was performed. The configuration yielding the highest Silhouette Score was chosen as the optimal model.

| Parameter | Best Result | Configuration | Score |
| :--- | :--- | :--- | :--- |
| **Optimal Clusters ($k$)** | **6** | **Cosine + Average** | **0.7033** |
| **Best Distance Metric** | Cosine | | |
| **Best Linkage Method** | Average | | |

#### 📊 Performance Comparison: Silhouette Score

The bar plot below illustrates the Silhouette Score for the three primary linkage methods (Single, Complete, Average) combined with three distance metrics across the range of cluster counts ($k=2$ to $k=10$).

**Visualizing the Optimal $k$ Search:**
![Silhouette Score Comparison (9 Configurations)](output/hc_silhouette_scores_9_configs.png)

---

#### 📈 Comprehensive $3 \times 3$ Cluster Comparison ($k=6$)

To visualize the dramatic effect of hyperparameters, we fixed the number of clusters at the optimal value ($k=6$) and generated a $3 \times 3$ grid comparing all combinations of Distance Metrics (Rows: Euclidean, Manhattan, Cosine) and Linkage Methods (Columns: Single, Complete, Average).

##### 1. Dendrogram Analysis (3x3 Grid)

These plots show the hierarchy of cluster formation. Note the differences in cluster height and shape, particularly the "chaining" behavior evident in Single Linkage plots.

![Hierarchical Clustering: Dendrogram Comparison (3 x 3 Grid)](output/hc_dendrograms_9_comparison.png)

##### 2. 3D Cluster Visualization (3x3 Grid)

These scatter plots show the **six** clusters in the 3D feature space (Age, Annual Income, Spending Score) as grouped by each parameter combination.

![3D Cluster Comparison (k=6 3 x 3 Grid)](output/hc_clusters_3d_9_comparison.png)

##### 3. Heatmap Visualization (9 Separate Clustermaps)

The clustered heatmaps show the scaled customer features. Customers are reordered based on the dendrogram structure for each specific metric/linkage combination, allowing for visual inspection of the feature values within the resulting clusters.

| Euclidean / Single | Euclidean / Complete | Euclidean / Average |
| :---: | :---: | :---: |
| ![Heatmap Euclidean/Single](output/hc_clustered_heatmap_euclidean_single_k6.png) | ![Heatmap Euclidean/Complete](output/hc_clustered_heatmap_euclidean_complete_k6.png) | ![Heatmap Euclidean/Average](output/hc_clustered_heatmap_euclidean_average_k6.png) |
| **Manhattan / Single** | **Manhattan / Complete** | **Manhattan / Average** |
| ![Heatmap Manhattan/Single](output/hc_clustered_heatmap_cityblock_single_k6.png) | ![Heatmap Manhattan/Complete](output/hc_clustered_heatmap_cityblock_complete_k6.png) | ![Heatmap Manhattan/Average](output/hc_clustered_heatmap_cityblock_average_k6.png) |
| **Cosine / Single** | **Cosine / Complete** | **Cosine / Average** |
| ![Heatmap Cosine/Single](output/hc_clustered_heatmap_cosine_single_k6.png) | ![Heatmap Cosine/Complete](output/hc_clustered_heatmap_cosine_complete_k6.png) | ![Heatmap Cosine/Average](output/hc_clustered_heatmap_cosine_average_k6.png) |

---

## Dimensionality Reduction

### 11. Principal Component Analysis (PCA)
- **`PCA Wine.py`**

**Explained Variance** <img src="output/pca_wine_variance.png" width="550"/>

**2D PCA Projection** <img src="output/pca_wine_2d.png" width="600"/>

> Reduced **11 → 2** features 
> Captures **99.5%** of total variance 
> Excellent compression with minimal information loss

---

## Getting Started

### Requirements
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
