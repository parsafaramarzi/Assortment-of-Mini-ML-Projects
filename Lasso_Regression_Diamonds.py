import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import os

# ---------------------------------------------
# 1. Configuration Constants
# ---------------------------------------------
ALPHA_VALUES = np.logspace(-4, 2, 100)
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_ITER = 10000
OUTPUT_FOLDER = 'output'

# ---------------------------------------------
# 2. Data Module: Loading and Preprocessing
# ---------------------------------------------
def load_and_preprocess_data(test_size, random_state):
    """Loads, encodes, splits, and scales the Diamonds dataset."""
    print("--- Data Loading and Preprocessing ---")
    
    data = sns.load_dataset('diamonds')
    
    X = data.drop(columns=['price'])
    Y = data['price']

    X_encoded = pd.get_dummies(X, drop_first=True)
    
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_encoded, Y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    feature_names = X_encoded.columns
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_names)
    
    print("Features successfully encoded and scaled.")
    
    return X_train_scaled_df, X_test_scaled_df, Y_train, Y_test, feature_names

# ---------------------------------------------
# 3. Training Module: Tuning and Model Training
# ---------------------------------------------
def tune_and_train_model(X_train, X_test, Y_train, Y_test, alpha_values, max_iter, random_state):
    """Iterates through alpha values to find the optimal Lasso model."""
    print("\n--- Model Tuning and Training ---")
    
    rmse_scores = []
    r2_scores = []
    coefficients = []

    for alpha in alpha_values:
        lasso = Lasso(alpha=alpha, max_iter=max_iter, random_state=random_state)
        lasso.fit(X_train, Y_train)

        Y_pred = lasso.predict(X_test)
        rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
        r2 = r2_score(Y_test, Y_pred)

        rmse_scores.append(rmse)
        r2_scores.append(r2)
        coefficients.append(lasso.coef_)
        
    coef_df = pd.DataFrame(coefficients, columns=X_train.columns, index=alpha_values)

    optimal_alpha_index = np.argmin(rmse_scores)
    optimal_alpha = alpha_values[optimal_alpha_index]
    
    best_lasso_model = Lasso(alpha=optimal_alpha, max_iter=max_iter, random_state=random_state)
    best_lasso_model.fit(X_train, Y_train)
    
    return best_lasso_model, optimal_alpha, rmse_scores, r2_scores, coef_df

# ---------------------------------------------
# 4. Analysis Module: Reporting Results
# ---------------------------------------------
def analyze_and_report_results(best_model, optimal_alpha, rmse_scores, r2_scores, feature_names, X_test, Y_test):
    """Prints performance metrics and feature selection results."""
    
    best_rmse = rmse_scores[np.argmin(rmse_scores)]
    best_r2 = r2_scores[np.argmin(rmse_scores)]
    
    print("\n--- Model Performance Report ---")
    print(f"Optimal Alpha (Lowest RMSE): {optimal_alpha:.6f}")
    print(f"Best RMSE on Test Set: {best_rmse:.2f}")
    print(f"Best R2 Score on Test Set: {best_r2:.4f}")
    
    final_coefs = pd.Series(best_model.coef_, index=feature_names)

    eliminated_features = final_coefs[final_coefs.abs() < 1e-4] 
    remaining_features = final_coefs[final_coefs.abs() >= 1e-4].sort_values(ascending=False)

    print(f"\nFeature Selection Results (Best Model, Alpha={optimal_alpha:.6f}):")
    print(f"\nFeatures ELIMINATED by Lasso ({len(eliminated_features)} features):")
    print(eliminated_features.index.tolist())

    print(f"\nTop 10 Remaining Features (Sorted by Magnitude):")
    print(remaining_features.head(10).to_string())

# ---------------------------------------------
# 5. Visualization Module: Plotting
# ---------------------------------------------
def generate_visualizations(alpha_values, rmse_scores, coef_df, best_model, X_test, Y_test, optimal_alpha):
    """
    Generates the three required plots for the Lasso project and 
    saves them to the defined output folder.
    """
    
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"\nCreated output folder: {OUTPUT_FOLDER}/")

    Y_pred = best_model.predict(X_test)
    residuals = Y_test - Y_pred

    plt.figure(figsize=(10, 6))
    plt.plot(alpha_values, rmse_scores, color='blue', label='RMSE on Test Set')
    plt.axvline(optimal_alpha, color='r', linestyle='--', label=f'Optimal Alpha ({optimal_alpha:.6f})')

    plt.xscale('log')
    plt.title('Plot 1: Lasso Regression Performance (RMSE) vs. Alpha ($\\alpha$)')
    plt.xlabel('Alpha ($\\alpha$) - Log Scale')
    plt.ylabel('Root Mean Squared Error (RMSE)')
    plt.legend()
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'LassoReg_Performance_vs_Alpha.png'))
    plt.show()

    plt.figure(figsize=(12, 8))
    plt.plot(coef_df.index, coef_df.values) 

    plt.xscale('log')
    plt.axvline(optimal_alpha, color='k', linestyle='--', label=f'Optimal Alpha')
    plt.title('Plot 2: Lasso Coefficient Path (Demonstrating Sparsity)')
    plt.xlabel('Alpha ($\\alpha$) - Log Scale')
    plt.ylabel('Coefficient Value')
    plt.legend(coef_df.columns, loc='upper right', ncol=2, fontsize='small')
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'LassoReg_Coefficient_Path.png'))
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.scatter(Y_pred, residuals, alpha=0.5)
    plt.hlines(0, Y_pred.min(), Y_pred.max(), color='r', linestyle='--')
    plt.xlabel('Predicted Prices')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.title('Plot 3: Residual Plot (Best Lasso Model)')
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'LassoReg_Residual_Plot.png'))
    plt.show()
    
# ---------------------------------------------
# 6. Main Execution Block
# ---------------------------------------------
if __name__ == "__main__":
    X_train, X_test, Y_train, Y_test, feature_names = load_and_preprocess_data(
        TEST_SIZE, RANDOM_STATE
    )

    best_lasso_model, optimal_alpha, rmse_scores, r2_scores, coef_df = tune_and_train_model(
        X_train, X_test, Y_train, Y_test, ALPHA_VALUES, MAX_ITER, RANDOM_STATE
    )
    
    analyze_and_report_results(
        best_lasso_model, optimal_alpha, rmse_scores, r2_scores, feature_names, X_test, Y_test
    )
    
    generate_visualizations(
        ALPHA_VALUES, rmse_scores, coef_df, best_lasso_model, X_test, Y_test, optimal_alpha
    )

    print("\n--- Lasso Regression Project Execution Complete ---")