# Assortment of Mini ML Projects

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A growing collection of **mini machine learning projects**, each focused on a core ML algorithm or concept — regression, classification, clustering, dimensionality reduction, and ensemble methods — using real datasets.  
Each project is self-contained and written to be easy to run and learn from.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1hb5CcUgi0QXym9RUFIm1-3BWEvNqPp26?usp=sharing)

---

## 🗂 Repository Overview

### 📁 Folder
- **Datasets/** – contains the CSV files used by the projects.

---

## 🧮 Regression Projects

### 1. Linear Regression
- **`LinearRegression Automobile Dataset.py`**  
  - **Features:** `horsepower`  
  - **Target:** `price`  
  - **Goal:** Predict automobile price from horsepower using linear regression.

- **`LinearRegression Bengaluru Housing Data.py`**  
  - **Features:** `total_sqft` (total square feet)  
  - **Target:** `price`  
  - **Goal:** Predict Bengaluru house prices from total square feet using linear regression.

- **`LinearRegression Heart Disease Dataset.py`**  
  - **Features:** `bmi`  
  - **Target:** `chol` (cholesterol)  
  - **Goal:** Predict cholesterol level from BMI using linear regression.

### 2. Polynomial Regression
- **`Polynomial Regression Automobile Dataset.py`**  
  - **Features:** `horsepower`  
  - **Target:** `price`  
  - **Goal:** Capture non-linear relationship between horsepower and price using polynomial regression.

- **`Polynomial Regression Bengaluru Housing Data.py`**  
  - **Features:** `total_sqft`  
  - **Target:** `price`  
  - **Goal:** Model non-linear house-price relationships with polynomial features.

- **`Polynomial Regression Heart Disease Dataset.py`** *(if present)*  
  - **Features:** `bmi`  
  - **Target:** `chol`  
  - **Goal:** Fit non-linear trends between BMI and cholesterol using polynomial regression.

---

## 🧠 Classification Projects

### 3. Decision Trees (DT)
- **`DT Drug200.py`**  
  - Classification of drug type from patient features.

- **`DT diabetes.py`**  
  - Predict diabetes presence from medical features.

### 4. Random Forest (RF)
- **`RF Drug200.py`**  
  - Ensemble version of Drug200 classification.

- **`RF diabetes.py`**  
  - Random Forest for diabetes prediction.

### 5. Support Vector Machine (SVM)
- **`SVM BreastCancer.py`**  
  - **Kernels used:** `linear`, `poly`, `rbf`  
  - Classify breast cancer cases and compare kernels.

### 6. K-Nearest Neighbors (KNN)
- **`KNN Iris&Customer.py`**  
  - **Iris:** standard species classification.  
  - **Customer subtask:**  
    - **Features:** `Age`, `Annual Income (k$)`, `Spending Score (1-100)`  
    - **Target:** `Gender`  
    - **Goal:** Predict customer gender using KNN on the three features.

---

## 🌀 Clustering Project

### 7. K-Means Mall Customers
- **`Kmeans Mall Customers.py`**  
  - **Features:** `Age`, `Annual Income (k$)`, `Spending Score (1-100)`  
  - **Goal:** Unsupervised customer segmentation using K-Means.  
  - **Includes:** elbow method for k selection and 2D/3D cluster visualizations.

---

## 🔮 Dimensionality Reduction

### 8. Principal Component Analysis (PCA)
- **`PCA Wine.py`**  
  - Apply PCA to the Wine dataset to visualize separation and explained variance.

---

## ⚙️ Getting Started

### Requirements
Install dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
