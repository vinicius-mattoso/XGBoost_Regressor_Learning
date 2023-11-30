import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
import xgboost as xgb
import matplotlib.pyplot as plt

# Create a folder to store results
result_folder = "results"
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

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

# Initialize XGBoost regressor
model = xgb.XGBRegressor(objective='reg:squarederror')

# Create KFold cross-validation
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1],
    # Add more hyperparameters as needed
}

# Define the scoring method
scorer = make_scorer(mean_squared_error, greater_is_better=False)

# Perform GridSearchCV with KFold cross-validation
grid_search = GridSearchCV(model, param_grid, scoring=scorer, cv=kf, verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best parameters and the best estimator
best_params = grid_search.best_params_
best_estimator = grid_search.best_estimator_

# Save configurations and errors to CSV file
results_df = pd.DataFrame(grid_search.cv_results_)
results_df.to_csv(os.path.join(result_folder, 'grid_search_results.csv'), index=False)

# Use the best estimator to make predictions on the test set
y_pred_test = best_estimator.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2_test = r2_score(y_test, y_pred_test)
print(f"Test RMSE with best estimator: {test_rmse}")
print(f"R-squared with best estimator: {r2_test}")

# Save errors to a CSV file
errors_df = pd.DataFrame({'True Values': y_test, 'Predicted Values': y_pred_test})
errors_df.to_csv(os.path.join(result_folder, 'test_errors.csv'), index=False)

# Create a plot of predicted vs true data for the test set and save it in the result_folder
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_test, color='blue', label='Predicted vs True')
plt.plot(y_test, y_test, linestyle='--', color='red', label='x=y')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs True Values\nR-squared: {:.4f}'.format(r2_test))
plt.legend()
plt.grid(True)
plt.text(0.5, 0.9, f'R-squared: {r2_test:.4f}', ha='center', va='center', transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', alpha=0.5))
plt.savefig(os.path.join(result_folder, 'predicted_vs_true_with_r2_combined.png'))
plt.show()

# Plot predicted values against true values in the index for the test set and save it in the result_folder
plt.figure(figsize=(10, 6))
plt.plot(data.index[len(data) - len(X_test):], y_test, '--o', label='True Values', color='blue')
plt.plot(data.index[len(data) - len(X_test):], y_pred_test, '--o', label='Predicted Values', color='green')
plt.xlabel('Index')
plt.ylabel('Passengers')
plt.title('Predicted vs True Values in Index (Test Set)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(result_folder, 'predicted_and_true_values_index_test_set.png'))
plt.show()
