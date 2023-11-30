from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import time
import matplotlib.pyplot as plt
import os

# Get the directory path of the current Python script
current_directory = os.path.dirname(__file__)

# Create a folder to store results in the current directory
result_folder = os.path.join(current_directory, "results")
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

start_time = time.time()
# Load the dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
data = pd.read_csv(url)
data['Month'] = pd.to_datetime(data['Month'])
data.set_index('Month', inplace=True)

# Prepare features and target variable
X = data.index.values.astype(float).reshape(-1, 1)
y = data['Passengers'].values

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the hyperparameters grid to search
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500, 600],
    'max_depth': [3, 4, 5, 6, 7, 8],
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3, 0.4],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5]
}

# Initialize XGBoost regressor
model = xgb.XGBRegressor(objective='reg:squarederror')

# Try GridSearchCV instead of RandomizedSearchCV for exhaustive search
grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error',
                           cv=5, verbose=2, n_jobs=-1)

# Fit the grid search to find the best parameters
grid_search.fit(X_train, y_train)

end_time = time.time()
total_time = end_time - start_time
print(f"\nTotal time for hyperparameter optimization: {total_time:.4f} seconds")

# Save configurations and errors to CSV file
results_df = pd.DataFrame(grid_search.cv_results_)
results_df.to_csv(os.path.join(result_folder, 'grid_search_results.csv'), index=False)

# Get the best parameters and the best estimator
best_params = grid_search.best_params_
best_estimator = grid_search.best_estimator_

# Use the best estimator to make predictions
y_pred_test_best = best_estimator.predict(X_test)
test_rmse_best = np.sqrt(mean_squared_error(y_test, y_pred_test_best))
r2_best = r2_score(y_test, y_pred_test_best)
print(f"\nTest RMSE with best estimator: {test_rmse_best}")
print(f"R-squared with best estimator: {r2_best}")

# Save errors to a CSV file
errors_df = pd.DataFrame({'True Values': y_test, 'Predicted Values': y_pred_test_best})
errors_df.to_csv(os.path.join(result_folder, 'test_errors.csv'), index=False)

# Create a plot of predicted vs true data for XGBoost
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_test_best, color='blue', label='Predicted vs True (XGBoost)')
plt.plot(y_test, y_test, linestyle='--', color='red', label='x=y')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs True Values with Best Estimator (XGBoost)')
plt.legend()
plt.grid(True)
# Add text for R-squared value in the plot
plt.text(0.5, 0.9, f'R-squared: {r2_best:.4f}', ha='center', va='center', transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', alpha=0.5))
# Save the plot as a .png file in the result_folder
plt.savefig(os.path.join(result_folder, 'predicted_vs_true_best_estimator_grid.png'))
plt.show()

# Plot predicted values by XGBoost against true values
plt.figure(figsize=(10, 6))
plt.plot(y_test, '--o', label='True Values', color='blue')
plt.plot(y_pred_test_best, '--o', label='Predicted (XGBoost)', color='green')
plt.xlabel('Index')
plt.ylabel('Passengers')
plt.title('Predicted (XGBoost) vs True Values (Test Set)')
plt.legend()
plt.grid(True)
plt.tight_layout()
# Save the plot as a .png file in the result_folder
plt.savefig(os.path.join(result_folder, 'predicted_and_true_xgboost.png'))
plt.show()
