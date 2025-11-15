# Assortment of Mini ML Projects
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Last Updated](https://img.shields.io/badge/last_updated-November_2025-blue)

A growing collection of **mini machine learning projects**, each focused on a key ML algorithm or concept — **regression, classification, clustering, dimensionality reduction, and computer vision** — using real-world datasets.

Each project is **self-contained**, **beginner-friendly**, and ideal for **hands-on learning** and **portfolio building**.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1hb5CcUgi0QXym9RUFIm1-3BWEvNqPp26?usp=sharing)

---

## Repository Overview
### Folders
- **`Datasets/`** – All datasets used across projects.  
- **`output/`** – Saved plots, models, or videos.

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

![Decision Tree – Diabetes](output/dt_diabetes_tree.png)  
*Visualised tree (depth ≤ 4)*

---

### 5. Random Forest (RF)
| Project | Goal |
|--------|------|
| `RF Drug200.py` | Ensemble drug classification |
| `RF diabetes.py` | Ensemble diabetes prediction |

![Feature Importance – RF Diabetes](output/rf_diabetes_importance.png)  
*Top 5 important features*

---

### 6. Support Vector Machine (SVM)
- **`SVM BreastCancer.py`** – kernels `linear`, `poly`, `rbf`

![SVM Decision Boundaries](output/svm_breastcancer_boundaries.png)  
*Linear vs RBF kernels (2-D projection)*

---

### 7. K-Nearest Neighbors (KNN)
- **`KNN Iris&Customer.py`**

![KNN – Iris Decision Regions](output/knn_iris_regions.png)  
*Decision regions for the Iris dataset (k=5)*

---

### 8. Logistic Regression — *Bank Customer Churn*
- **`LogisticRegression Bank Customer Churn.py`**

![Confusion Matrix Heatmap](output/logreg_churn_confmat.png)  
*Confusion matrix (seaborn heatmap)*

![ROC Curve + AUC](output/logreg_churn_roc.png)  
*ROC curve with AUC = 0.86*

---

## Clustering Project

### 9. K-Means — *Mall Customer Segmentation*
- **`Kmeans Mall Customers.py`**

![Elbow Plot](output/kmeans_elbow.png)  
*Elbow method to choose k*

![Customer Clusters](output/kmeans_clusters.png)  
*2-D scatter of Income vs Spending Score (k=5)*

---

## Dimensionality Reduction

### 10. Principal Component Analysis (PCA)
- **`PCA Wine.py`**

![Explained Variance Ratio](output/pca_wine_variance.png)  
*Cumulative explained variance*

![PCA 2-D Projection](output/pca_wine_2d.png)  
*Wine classes in the first two principal components*

---

## Computer Vision — Object Detection

### 11. YOLOv8 Real-Time Object Tracking
- **`YOLOv8 Car Traffic Detection.py`**
  - **Model:** `yolov8n.pt`
  - **Input:** `Datasets/cartraffic03.mp4`

![YOLOv8 Traffic Demo](output/yolov8_traffic_frame.jpg)  
*Sample frame with custom coloured bounding boxes (person=red, car=blue, bus=cyan, etc.)*

---

## Getting Started

### Requirements
```bash
pip install pandas numpy scikit-learn matplotlib seaborn opencv-python ultralytics
