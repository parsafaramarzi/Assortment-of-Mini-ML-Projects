# Assortment of Mini ML Projects

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A growing collection of **mini machine learning projects**, each focused on a key ML algorithm or concept — regression, classification, clustering, dimensionality reduction, and ensemble methods — using real-world datasets.  

Each project is self-contained, beginner-friendly, and ideal for learning or portfolio showcasing.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1hb5CcUgi0QXym9RUFIm1-3BWEvNqPp26?usp=sharing)

---

## 🗂 Repository Overview

### 📁 Folder
- **Datasets/** – Contains all datasets used by the projects.

---

## 🧮 Regression Projects

### 1. Linear Regression
- **`LinearRegression Automobile Dataset.py`**  
  Predict automobile prices from features like horsepower, weight, and engine size.  
- **`LinearRegression Bengaluru Housing Data.py`**  
  Estimate Bengaluru housing prices using location, area, and number of rooms.  
- **`LinearRegression Heart Disease Dataset.py`**  
  Predict cholesterol levels (`chol`) based on BMI and other health metrics.

### 2. Polynomial Regression
- **`Polynomial Regression Automobile Dataset.py`**  
  Extend the automobile dataset with polynomial features to capture non-linear trends.  
- **`Polynomial Regression Bengaluru Housing Data.py`**  
  Explore polynomial fits for complex housing price relationships.

---

## 🧠 Classification Projects

### 3. Decision Trees (DT)
- **`DT Drug200.py`**  
  Classify drug prescriptions based on patient features (age, sex, BP, cholesterol).  
- **`DT diabetes.py`**  
  Predict diabetes presence from patient medical data.

### 4. Random Forest (RF)
- **`RF Drug200.py`**  
  Ensemble classifier improving accuracy over single decision trees on Drug200.  
- **`RF diabetes.py`**  
  Random Forest applied to diabetes prediction for better generalization.

### 5. Support Vector Machine (SVM)
- **`SVM BreastCancer.py`**  
  Classify breast cancer cases using linear and RBF kernels.

### 6. K-Nearest Neighbors (KNN)
- **`KNN Iris&Customer.py`**  
  Apply KNN for:
  - Iris dataset species classification  
  - Customer segmentation based on spending and income data.

---

## 🌀 Clustering Project

### 7. K-Means Mall Customers
- **`Kmeans Mall Customers.py`**  
  Perform unsupervised **customer segmentation** using K-Means clustering.  
  Groups mall customers into clusters based on:
  - Annual income  
  - Spending score  
  - Purchasing behavior  

  Includes:
  - Elbow method to find the optimal number of clusters  
  - Cluster visualization in 2D space  

---

## 🔮 Dimensionality Reduction

### 8. Principal Component Analysis (PCA)
- **`PCA Wine.py`**  
  Apply PCA to the **Wine dataset** to visualize class separation in lower dimensions.  
  Shows explained variance and feature compression for interpretability.

---

## ⚙️ Getting Started

### Requirements
Install dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
