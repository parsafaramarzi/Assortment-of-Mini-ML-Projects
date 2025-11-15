import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn import decomposition

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

data = pd.read_csv("Datasets/concrete_data.csv")
x = data.drop(columns=["concrete_compressive_strength"])
y = data["concrete_compressive_strength"]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
feature_names = data.columns.drop("concrete_compressive_strength")
target_name = "concrete_compressive_strength"

fig = plt.figure(figsize=(12, 12))
for i in range(x_train.shape[1]):
    feature_name = feature_names[i]
    ax1 = fig.add_subplot(4,2,i+1)
    ax1.scatter(data[feature_name], data[target_name])
    ax1.set_title(f"{feature_name} vs {target_name}")
fig.tight_layout(pad=3.0)
plt.savefig(os.path.join(OUTPUT_DIR, "concrete_feature_scatter.png"), dpi=200, bbox_inches='tight')
plt.show()

fig = plt.figure(figsize=(12, 12))
for i in range(x_train.shape[1]):
    feature_name = feature_names[i]
    ax1 = fig.add_subplot(4,2,i+1)
    ax1.hist(data[feature_name], bins=20, color='skyblue', edgecolor='black')
    ax1.set_title(f"{feature_name} Distribution")
fig.tight_layout(pad=3.0)
plt.savefig(os.path.join(OUTPUT_DIR, "concrete_histograms.png"), dpi=200, bbox_inches='tight')
plt.show()

cor_matrix = data.corr()
sns.heatmap(cor_matrix,annot=True,cmap="coolwarm")
plt.savefig(os.path.join(OUTPUT_DIR, "concrete_corr_heatmap.png"), dpi=200, bbox_inches='tight')
plt.show()

LR_Model = LinearRegression()
Bestmodel = "None"
BestAcc = 0
BestMSE = 0
BestPred = None

LR_Model.fit(x_train,y_train)
y_pred = LR_Model.predict(x_test)
acc = r2_score(y_test,y_pred)
MSE = mean_squared_error(y_test, y_pred)
print(f"MultiLinear accuracy is {acc*100:.2f}% and MSE is {MSE:.2f}")
if BestAcc < acc:
    Bestmodel = "MultiLinear"
    BestAcc = acc
    BestMSE = MSE
    BestPred = y_pred
pca = decomposition.PCA(n_components=1)
pca.fit(x_train)
x_test_pca = pca.transform(x_test)
plt.scatter(x_test_pca, y_test)
plt.scatter(x_test_pca, y_pred, color='red')
plt.title("Multilinear All features vs Strength")
plt.xlabel("Features")
plt.ylabel("Strength")
plt.legend(["Actual", "Predicted"])
plt.savefig(os.path.join(OUTPUT_DIR, "concrete_pca_1d.png"), dpi=200, bbox_inches='tight')
plt.show()

fig = plt.figure(figsize=(12, 12))
for i in range(x_train.shape[1]):

    feature_name = feature_names[i]
    X_train_single = x_train[[feature_name]]
    X_test_single = x_test[[feature_name]]
    LR_Model.fit(X_train_single, y_train)
    y_pred = LR_Model.predict(X_test_single)
    acc = r2_score(y_test, y_pred)
    MSE = mean_squared_error(y_test, y_pred)
    print(f"Linear R² accuracy with '{feature_name}' as feature is {acc*100:.2f}% and MSE is {MSE:.2f}")
    if BestAcc < acc:
        Bestmodel = f"Linear {feature_name}"
        BestAcc = acc
        BestMSE = MSE
        BestPred = y_pred
    ax1 = fig.add_subplot(4,2,i+1)
    ax1.scatter(x_test[feature_name], y_test)
    ax1.scatter(x_test[feature_name], y_pred,c='red')
    ax1.set_title(f"{feature_name} vs Strength")
fig.tight_layout(pad=3.0)
plt.savefig(os.path.join(OUTPUT_DIR, "concrete_single_feature_fits.png"), dpi=200, bbox_inches='tight')
plt.show()    
print(f"\nOverall Best Model: {Bestmodel} with R² = {BestAcc*100:.2f}% and MSE = {BestMSE:.2f}")

LR_Model.fit(x_train, y_train)
y_pred_multi = LR_Model.predict(x_test)
residuals = y_test - y_pred_multi
min_val = min(y_test.min(), y_pred_multi.min())
max_val = max(y_test.max(), y_pred_multi.max())

fig = plt.figure(figsize=(12, 12))
#Predicted vs. Actual Strength
ax1 = fig.add_subplot(2, 2, 1)
ax1.scatter(y_test, y_pred_multi, alpha=0.6)
ax1.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Fit (y=x)')
ax1.set_title('Predicted vs. Actual Strength (Multi-Linear Before Improvements)')
ax1.set_xlabel('Actual Compressive Strength (y_test)')
ax1.set_ylabel('Predicted Compressive Strength (y_pred_multi)')
ax1.legend()
ax1.grid(True, linestyle=':', alpha=0.7)

# Residuals vs. Predicted Values
ax2 = fig.add_subplot(2, 2, 2)
ax2.scatter(y_pred_multi, residuals, alpha=0.6)
ax2.hlines(y=0, xmin=y_pred_multi.min(), xmax=y_pred_multi.max(), color='red', linestyle='--')
ax2.set_title('Residuals vs. Predicted Values (Before Improvements)')
ax2.set_xlabel('Predicted Compressive Strength (y_pred_multi)')
ax2.set_ylabel('Residuals (y_test - y_pred_multi)')
ax2.grid(True, linestyle=':', alpha=0.7)

# Distribution of Residuals
ax3 = fig.add_subplot(2, 2, 3)
sns.histplot(residuals, kde=True, bins=30, color='skyblue', edgecolor='black', ax=ax3)
ax3.set_title('Distribution of Residuals (Before Improvements)')
ax3.set_xlabel('Residuals')
ax3.set_ylabel('Frequency')
fig.tight_layout(pad=3.0)
plt.savefig(os.path.join(OUTPUT_DIR, "concrete_residuals_before.png"), dpi=200, bbox_inches='tight')
plt.show()

x_train_fs = x_train.drop(columns=["blast_furnace_slag","fly_ash"])
x_test_fs = x_test.drop(columns=["blast_furnace_slag","fly_ash"])

poly = PolynomialFeatures(degree=2, include_bias=False)
x_train_poly = poly.fit_transform(x_train_fs)
x_test_poly = poly.transform(x_test_fs)

LR_Model.fit(x_train, y_train)
y_pred = LR_Model.predict(x_test)

LR_Model_New = LinearRegression()
LR_Model_New.fit(x_train_poly, y_train)
y_pred_poly = LR_Model_New.predict(x_test_poly)
acc_poly = r2_score(y_test, y_pred_poly)
MSE_poly = mean_squared_error(y_test, y_pred_poly)
print(f"MultiLinear accuracy after improvements is {acc_poly*100:.2f}% and MSE is {MSE_poly:.2f}")

residuals_New = y_test - y_pred_poly
min_val = min(y_test.min(), y_pred_poly.min())
max_val = max(y_test.max(), y_pred_poly.max())

#Predicted vs. Actual Strength
fig = plt.figure(figsize=(12, 12))
ax1 = fig.add_subplot(3, 2, 1)
ax1.scatter(y_test, y_pred_poly, alpha=0.6)
ax1.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Fit (y=x)')
ax1.set_title('Predicted vs. Actual Strength (Multi-Linear After Improvements)')
ax1.set_xlabel('Actual Compressive Strength (y_test)')
ax1.set_ylabel('Predicted Compressive Strength (y_pred_poly)')
ax1.legend()
ax1.grid(True, linestyle=':', alpha=0.7)

ax1 = fig.add_subplot(3, 2, 2)
ax1.scatter(y_test, y_pred, alpha=0.6)
ax1.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Fit (y=x)')
ax1.set_title('Predicted vs. Actual Strength (Multi-Linear Before Improvements)')
ax1.set_xlabel('Actual Compressive Strength (y_test)')
ax1.set_ylabel('Predicted Compressive Strength (y_pred)')
ax1.legend()
ax1.grid(True, linestyle=':', alpha=0.7)

# Residuals vs. Predicted Values
ax2 = fig.add_subplot(3, 2, 3)
ax2.scatter(y_pred_poly, residuals_New, alpha=0.6)
ax2.hlines(y=0, xmin=y_pred_poly.min(), xmax=y_pred_poly.max(), color='red', linestyle='--')
ax2.set_title('Residuals vs. Predicted Values (After Improvements)')
ax2.set_xlabel('Predicted Compressive Strength (y_pred_poly)')
ax2.set_ylabel('Residuals (y_test - y_pred_poly)')
ax2.grid(True, linestyle=':', alpha=0.7)

ax2 = fig.add_subplot(3, 2, 4)
ax2.scatter(y_pred, residuals, alpha=0.6)
ax2.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), color='red', linestyle='--')
ax2.set_title('Residuals vs. Predicted Values (Before Improvements)')
ax2.set_xlabel('Predicted Compressive Strength (y_pred)')
ax2.set_ylabel('Residuals (y_test - y_pred)')
ax2.grid(True, linestyle=':', alpha=0.7)

# Distribution of Residuals
ax3 = fig.add_subplot(3, 2, 5)
sns.histplot(residuals_New, kde=True, bins=30, color='skyblue', edgecolor='black', ax=ax3)
ax3.set_title('Distribution of Residuals (After Improvements)')
ax3.set_xlabel('Residuals')
ax3.set_ylabel('Frequency')

ax3 = fig.add_subplot(3, 2, 6)
sns.histplot(residuals, kde=True, bins=30, color='skyblue', edgecolor='black', ax=ax3)
ax3.set_title('Distribution of Residuals (Before Improvements)')
ax3.set_xlabel('Residuals')
ax3.set_ylabel('Frequency')
fig.tight_layout(pad=3.0)
plt.savefig(os.path.join(OUTPUT_DIR, "concrete_residuals.png"), dpi=200, bbox_inches='tight')
plt.show()