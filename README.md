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
- **`notebooks/`** *(optional)* – Jupyter notebooks for interactive exploration.
- **`output/`** *(optional)* – Saved plots, models, or videos.

---

## Regression Projects

### 1. Linear Regression
| Project | Feature | Target | Goal |
|-------|--------|--------|------|
| `LinearRegression Automobile Dataset.py` | `horsepower` | `price` | Predict car price |
| `LinearRegression Bengaluru Housing Data.py` | `total_sqft` | `price` | Predict house price |
| `LinearRegression Heart Disease Dataset.py` | `bmi` | `chol` | Predict cholesterol from BMI |

### 2. Polynomial Regression
| Project | Feature | Target | Goal |
|-------|--------|--------|------|
| `Polynomial Regression Automobile Dataset.py` | `horsepower` | `price` | Capture non-linear price trends |
| `Polynomial Regression Bengaluru Housing Data.py` | `total_sqft` | `price` | Model non-linear housing trends |
| `Polynomial Regression Heart Disease Dataset.py` | `bmi` | `chol` | Fit non-linear BMI-cholesterol |

### 3. Multiple Linear Regression — *Concrete Compressive Strength*
- **`MultipleLinearRegression Concrete Dataset.py`**
  - **Dataset:** `concrete_data.csv`
  - **Goal:** Predict **concrete compressive strength** from 8 material components.
  - **Features:** Cement, Slag, Fly Ash, Water, Superplasticizer, Aggregates, Age
  - **Highlights:**
    - EDA (scatter plots, correlation heatmap)
    - R², MSE evaluation
    - PCA visualization
    - Polynomial feature enhancement
    - Residual analysis

---

## Classification Projects

### 4. Decision Trees (DT)
| Project | Goal |
|--------|------|
| `DT Drug200.py` | Classify drug type from patient features |
| `DT diabetes.py` | Predict diabetes (0/1) from health metrics |

### 5. Random Forest (RF)
| Project | Goal |
|--------|------|
| `RF Drug200.py` | Improved drug classification with ensemble |
| `RF diabetes.py` | Enhanced diabetes prediction |

### 6. Support Vector Machine (SVM)
- **`SVM BreastCancer.py`**
  - **Kernels:** `linear`, `poly`, `rbf`
  - Compares kernel performance on breast cancer diagnosis.

### 7. K-Nearest Neighbors (KNN)
- **`KNN Iris&Customer.py`**
  - **Iris:** Species classification
  - **Customer:** Predict `Gender` from `Age`, `Income`, `Spending Score`

### 8. Logistic Regression — *Bank Customer Churn*
- **`LogisticRegression Bank Customer Churn.py`**
  - **Dataset:** `Bank Customer Churn Prediction.csv`
  - **Goal:** Predict customer churn
  - **Highlights:**
    - Label encoding
    - Accuracy, classification report
    - Confusion matrix heatmap
    - ROC curve + AUC
    - Feature importance

---

## Clustering Project

### 9. K-Means — *Mall Customer Segmentation*
- **`Kmeans Mall Customers.py`**
  - **Features:** `Age`, `Income`, `Spending Score`
  - **Goal:** Segment customers into clusters
  - **Includes:** Elbow method, 2D cluster visualization

---

## Dimensionality Reduction

### 10. Principal Component Analysis (PCA)
- **`PCA Wine.py`**
  - Apply PCA to Wine dataset
  - Visualize class separation
  - Plot explained variance

---

## Computer Vision — Object Detection

### 11. YOLOv8 Real-Time Object Tracking
- **`YOLOv8 Car Traffic Detection.py`**
  - **Model:** `yolov8n.pt` (Ultralytics)
  - **Input:** `Datasets/cartraffic03.mp4`
  - **Features:**
    - Real-time object detection & tracking
    - **Custom class-specific BGR colors** (80+ classes)
    - Aspect-ratio-preserving resize
    - Live OpenCV display
  - **Goal:** Detect and track cars, people, buses, etc. in traffic video

---

## Getting Started

### Requirements
```bash
pip install pandas numpy scikit-learn matplotlib seaborn opencv-python ultralytics
