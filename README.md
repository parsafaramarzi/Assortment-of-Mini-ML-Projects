# Assortment of Mini ML Projects

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A growing collection of **mini machine learning projects**, each focused on a key ML algorithm or concept — regression, classification, clustering, and dimensionality reduction — using real-world datasets.

Each project is self-contained, beginner-friendly, and ideal for hands-on ML learning and portfolio building.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1hb5CcUgi0QXym9RUFIm1-3BWEvNqPp26?usp=sharing)

---

## 🗂 Repository Overview

### 📁 Folder
- **Datasets/** – Contains all datasets used by the projects.

---

## 🧮 Regression Projects

### 1. Linear Regression
- **`LinearRegression Automobile Dataset.py`**  
  - **Features:** `horsepower`  
  - **Target:** `price`  
  - **Goal:** Predict automobile price from horsepower using linear regression.

- **`LinearRegression Bengaluru Housing Data.py`**  
  - **Features:** `total_sqft`  
  - **Target:** `price`  
  - **Goal:** Predict Bengaluru house prices based on total square feet.

- **`LinearRegression Heart Disease Dataset.py`**  
  - **Features:** `bmi`  
  - **Target:** `chol`  
  - **Goal:** Predict cholesterol level from BMI using linear regression.

### 2. Polynomial Regression
- **`Polynomial Regression Automobile Dataset.py`**  
  - **Features:** `horsepower`  
  - **Target:** `price`  
  - **Goal:** Capture non-linear relationship between horsepower and price.

- **`Polynomial Regression Bengaluru Housing Data.py`**  
  - **Features:** `total_sqft`  
  - **Target:** `price`  
  - **Goal:** Model non-linear house-price relationships with polynomial features.

- **`Polynomial Regression Heart Disease Dataset.py`** *(if present)*  
  - **Features:** `bmi`  
  - **Target:** `chol`  
  - **Goal:** Fit non-linear trends between BMI and cholesterol using polynomial regression.

### 3. Multiple Linear Regression — *Concrete Compressive Strength*
- **`MultipleLinearRegression Concrete Dataset.py`**  
  - **Dataset:** `concrete_data.csv`  
  - **Goal:** Predict **compressive strength of concrete** from its material components.  
  - **Features:**  
    Cement, Blast furnace slag, Fly ash, Water, Superplasticizer, Coarse aggregate, Fine aggregate, Age.  
  - **Target:** `concrete_compressive_strength`  
  - **Highlights:**  
    - Multi-feature Linear Regression  
    - EDA (scatter, histograms, correlation heatmap)  
    - Model evaluation (R², MSE)  
    - PCA visualization  
    - Polynomial features for model improvement  
    - Residual analysis (before and after improvements)

---

## 🧠 Classification Projects

### 4. Decision Trees (DT)
- **`DT Drug200.py`**  
  - Classify drug type based on patient features.  
- **`DT diabetes.py`**  
  - Predict diabetes presence using medical data.

### 5. Random Forest (RF)
- **`RF Drug200.py`**  
  - Ensemble classifier improving accuracy on Drug200 dataset.  
- **`RF diabetes.py`**  
  - Random Forest applied to diabetes prediction.

### 6. Support Vector Machine (SVM)
- **`SVM BreastCancer.py`**  
  - **Kernels used:** `linear`, `poly`, `rbf`  
  - Classify breast cancer cases and compare kernel performance.

### 7. K-Nearest Neighbors (KNN)
- **`KNN Iris&Customer.py`**  
  - **Iris:** species classification.  
  - **Customer:**  
    - **Features:** `Age`, `Annual Income (k$)`, `Spending Score (1-100)`  
    - **Target:** `Gender`  
    - **Goal:** Predict customer gender using KNN.

### 8. Logistic Regression — *Bank Customer Churn Prediction*
- **`LogisticRegression Bank Customer Churn.py`**  
  - **Dataset:** `Bank Customer Churn Prediction.csv`  
  - **Goal:** Predict whether a customer will leave the bank (churn).  
  - **Features:** `credit_score`, `country`, `gender`, `age`, `tenure`, `balance`, `products_number`, `credit_card`, `active_member`, `estimated_salary`.  
  - **Target:** `churn`  
  - **Highlights:**  
    - Label encoding for categorical data  
    - Model accuracy and classification report  
    - Confusion matrix heatmap  
    - ROC curve and AUC visualization  
    - Feature importance via coefficients

---

## 🌀 Clustering Project

### 9. K-Means Mall Customers
- **`Kmeans Mall Customers.py`**  
  - **Features:** `Age`, `Annual Income (k$)`, `Spending Score (1-100)`  
  - **Goal:** Customer segmentation with K-Means clustering.  
  - **Includes:** Elbow method and 2D cluster visualization.

---

## 🔮 Dimensionality Reduction

### 10. Principal Component Analysis (PCA)
- **`PCA Wine.py`**  
  - Apply PCA to the Wine dataset to visualize feature separation and explained variance.

---

## ⚙️ Getting Started

### Requirements
Install dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
