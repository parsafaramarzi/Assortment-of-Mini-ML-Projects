#----------------------------------------------------
# Imports and Libraries
#----------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
import os
import matplotlib.cm as cm
from matplotlib.lines import Line2D

#----------------------------------------------------
# Configuration Constants
#----------------------------------------------------
ALPHA_VALUES = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
RANDOM_STATE = 42
TEST_SIZE = 0.2
OUTPUT_DIR = 'output'
DATA_PATH = 'Datasets\\concrete_data.csv'
FILE_PREFIX = 'RidgeRegression_'
PCA_PLOT_FILENAME = FILE_PREFIX + 'pca_1d_regression_comparison.png'
PERF_PLOT_FILENAME = FILE_PREFIX + 'ridge_performance_vs_alpha.png'
CSV_OUTPUT_FILENAME = FILE_PREFIX + 'model_performance_metrics.csv'
COEF_OUTPUT_FILENAME = FILE_PREFIX + 'coefficient_comparison.csv'

#----------------------------------------------------
# Data Handling Functions
#----------------------------------------------------
def load_data(path):
    try:
        data = pd.read_csv(path)
        x_names = data.columns[:-1].tolist()
        y_name = data.columns[-1]
        X = data[x_names]
        Y = data[y_name]
        return X, Y
    except FileNotFoundError:
        print(f"Error: The file '{path}' was not found.")
        print("Please ensure the path is correct or place the file in the correct directory.")
        exit()

def prepare_data(X, Y, test_size, random_state):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    return X_train_scaled_df, X_test_scaled_df, Y_train, Y_test

#----------------------------------------------------
# Model Evaluation Function (8 Features)
#----------------------------------------------------
def evaluate_ridge_models_8D(X_train, X_test, Y_train, Y_test, alpha_values):
    results = []
    
    ols_model = LinearRegression()
    ols_model.fit(X_train, Y_train)
    Y_pred_ols = ols_model.predict(X_test)
    
    results.append({
        'Alpha': 0.0,
        'MSE': mean_squared_error(Y_test, Y_pred_ols),
        'R2': r2_score(Y_test, Y_pred_ols),
        'Model': ols_model
    })
    
    for alpha in alpha_values:
        ridge_model = Ridge(alpha=alpha, random_state=RANDOM_STATE)
        ridge_model.fit(X_train, Y_train)
        Y_pred = ridge_model.predict(X_test)

        results.append({
            'Alpha': alpha,
            'MSE': mean_squared_error(Y_test, Y_pred),
            'R2': r2_score(Y_test, Y_pred),
            'Model': ridge_model
        })
            
    return pd.DataFrame(results)

#----------------------------------------------------
# Coefficient Analysis Function
#----------------------------------------------------
def analyze_coefficients(results_df, feature_names, output_dir, filename):
    
    # Identify key models
    ols_row = results_df[results_df['Alpha'] == 0.0].iloc[0]
    best_row = results_df.loc[results_df['MSE'].idxmin()]
    max_alpha = max(results_df['Alpha'])
    max_penalty_row = results_df[results_df['Alpha'] == max_alpha].iloc[0]
    
    # Extract coefficients
    ols_coef = np.insert(ols_row['Model'].coef_, 0, ols_row['Model'].intercept_)
    best_coef = np.insert(best_row['Model'].coef_, 0, best_row['Model'].intercept_)
    max_penalty_coef = np.insert(max_penalty_row['Model'].coef_, 0, max_penalty_row['Model'].intercept_)

    # Create comparison table
    coef_df = pd.DataFrame({
        'Feature': ['Intercept'] + feature_names,
        'OLS (Alpha=0.0)': ols_coef,
        f'Best Ridge (Alpha={best_row["Alpha"]:.3f})': best_coef,
        f'Max Penalty (Alpha={max_alpha:.0f})': max_penalty_coef
    })

    print("\n----------------------------------------------------")
    print("--- Coefficient Comparison (Scaled Data) ---")
    print("----------------------------------------------------")
    print(coef_df.round(4).to_string(index=False))
    
    # Save to CSV
    output_path = os.path.join(output_dir, filename)
    coef_df.round(4).to_csv(output_path, index=False)
    print(f"\nCoefficient data saved to {output_path}")

#----------------------------------------------------
# CSV Output Function
#----------------------------------------------------
def save_results_to_csv(results_df, output_dir, filename):
    output_path = os.path.join(output_dir, filename)
    results_df.drop(columns=['Model']).to_csv(output_path, index=False)
    print(f"Performance data saved to {output_path}")

#----------------------------------------------------
# Visualization Function - Plot 1 (1D PCA with Regression Lines)
#----------------------------------------------------
def plot_1d_pca_with_lines(X_train_scaled, X_test_scaled, Y_train, Y_test, alpha_values, output_dir, filename):
    
    pca = PCA(n_components=1, random_state=RANDOM_STATE)
    Z_train = pca.fit_transform(X_train_scaled.values)
    Z_test = pca.transform(X_test_scaled.values)
    
    Z_train_reshaped = Z_train.reshape(-1, 1)
    Z_test_reshaped = Z_test.reshape(-1, 1)
    
    all_models = {}
    all_mse = {}
    best_1d_mse = float('inf')
    best_1d_alpha = 0.0
    all_alpha_keys = [0.0] + alpha_values

    ols_model = LinearRegression()
    ols_model.fit(Z_train_reshaped, Y_train)
    all_models[0.0] = ols_model
    all_mse[0.0] = mean_squared_error(Y_test, ols_model.predict(Z_test_reshaped))
    best_1d_mse = all_mse[0.0]
    best_1d_alpha = 0.0

    for alpha in alpha_values:
        ridge_model = Ridge(alpha=alpha, random_state=RANDOM_STATE)
        ridge_model.fit(Z_train_reshaped, Y_train)
        all_models[alpha] = ridge_model
        
        mse = mean_squared_error(Y_test, ridge_model.predict(Z_test_reshaped))
        all_mse[alpha] = mse
        
        if mse < best_1d_mse:
            best_1d_mse = mse
            best_1d_alpha = alpha
            
    plt.figure(figsize=(14, 8))

    plt.scatter(Z_train, Y_train, color='darkgreen', alpha=0.6, s=50, label='Train Data')
    plt.scatter(Z_test, Y_test, color='darkorange', alpha=0.8, s=50, label='Test Data')

    Z_plot = np.linspace(min(Z_train.min(), Z_test.min()), max(Z_train.max(), Z_test.max()), 100).reshape(-1, 1)

    colors = cm.plasma(np.linspace(0.2, 0.9, len(alpha_values)))
    
    ridge_alphas = sorted(alpha_values)

    for i, alpha in enumerate(all_alpha_keys):
        model = all_models[alpha]
        y_plot = model.predict(Z_plot)
        
        if alpha == 0.0:
            plt.plot(Z_plot, y_plot, 
                     label='OLS ($\mathbf{\\alpha}$=0)', 
                     color='black', 
                     linestyle='-', 
                     linewidth=2.5)
        elif alpha == best_1d_alpha:
            plt.plot(Z_plot, y_plot, 
                     label=f'Best Ridge ($\mathbf{{\\alpha}}$={alpha})', 
                     color='red', 
                     linestyle='--', 
                     linewidth=3.0)
        else:
            idx = ridge_alphas.index(alpha)
            plt.plot(Z_plot, y_plot, 
                     label=f'Ridge $\mathbf{{\\alpha}}$={alpha}', 
                     color=colors[idx],
                     linestyle='-', 
                     linewidth=1.0,
                     alpha=0.6)

    plt.title('Plot 1: 1D PCA Projection with Regression Lines', fontsize=18)
    plt.xlabel('First Principal Component ($\mathbf{Z_1}$ of Scaled Features)', fontsize=14)
    plt.ylabel('Concrete Compressive Strength (MPa)', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    print(f"Plot 1 saved to {os.path.join(output_dir, filename)}")
    plt.show()

#----------------------------------------------------
# Visualization Function - Plot 2 (Performance Metrics)
#----------------------------------------------------
def plot_performance_metrics(results_df, output_dir, filename):
    
    plot_data = results_df[results_df['Alpha'] > 0].copy()
    baseline_mse = results_df[results_df['Alpha'] == 0]['MSE'].iloc[0]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Alpha Value (Log Scale)', fontsize=12)
    ax1.set_ylabel('Mean Squared Error (MSE)', color=color, fontsize=12)
    ax1.plot(plot_data['Alpha'], plot_data['MSE'], color=color, marker='o', label='Ridge MSE')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xscale('log')

    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('R-squared (R2) Score', color=color, fontsize=12)
    ax2.plot(plot_data['Alpha'], plot_data['R2'], color=color, marker='x', linestyle='--', label='Ridge R2')
    ax2.tick_params(axis='y', labelcolor=color)

    ax1.axhline(baseline_mse, color='k', linestyle=':', label=f'OLS Baseline MSE ({baseline_mse:.4f})')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.title('Plot 2: Ridge Regression Performance vs. Alpha (8 Features)', fontsize=14)
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    print(f"Plot 2 saved to {os.path.join(output_dir, filename)}")
    plt.show()

#----------------------------------------------------
# Main Execution Block
#----------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    X, Y = load_data(DATA_PATH)
    X_train_scaled_df, X_test_scaled_df, Y_train, Y_test = prepare_data(
        X, Y, TEST_SIZE, RANDOM_STATE
    )
    feature_names = X.columns.tolist()

    # 1. PLOT 1: 1D Visualization
    print("\n----------------------------------------------------")
    print("--- Generating Plot 1: 1D PCA with Regression Lines ---")
    print("----------------------------------------------------")
    plot_1d_pca_with_lines(
        X_train_scaled_df, 
        X_test_scaled_df,  
        Y_train,           
        Y_test,            
        ALPHA_VALUES, 
        OUTPUT_DIR,
        PCA_PLOT_FILENAME
    )
    
    # 2. Model Evaluation (8D)
    print("\n----------------------------------------------------")
    print("--- Evaluating 8D Ridge Models ---")
    print("----------------------------------------------------")
    results_df = evaluate_ridge_models_8D(
        X_train_scaled_df, X_test_scaled_df, Y_train, Y_test, ALPHA_VALUES
    )
    
    # 3. Coefficient Analysis (Print and Save to CSV)
    analyze_coefficients(results_df, feature_names, OUTPUT_DIR, COEF_OUTPUT_FILENAME)
    
    # 4. Save Performance Metrics to CSV
    print("\n----------------------------------------------------")
    print("--- Saving Performance Metrics to CSV ---")
    print("----------------------------------------------------")
    save_results_to_csv(results_df, OUTPUT_DIR, CSV_OUTPUT_FILENAME)
    
    # 5. PLOT 2: Performance Metrics
    print("\n----------------------------------------------------")
    print("--- Generating Plot 2: Performance Metrics ---")
    print("----------------------------------------------------")
    plot_performance_metrics(results_df, OUTPUT_DIR, PERF_PLOT_FILENAME)

#----------------------------------------------------
# Script Entry Point
#----------------------------------------------------
if __name__ == '__main__':
    main()