import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import time
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt

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

# Initialize XGBoost regressor
model = xgb.XGBRegressor(objective='reg:squarederror')

# Fit the model on training data
model.fit(X_train, y_train)

# Predict on test set
y_pred_test = model.predict(X_test)

# Calculate test RMSE and R-squared
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2_test = r2_score(y_test, y_pred_test)
print(f"Test RMSE: {test_rmse}")
print(f"R-squared: {r2_test}")

end_time = time.time()
total_time = end_time - start_time
print(f"Total time: {total_time:.4f} seconds")

# Create a plot of predicted vs true data for the test set
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_test, color='blue', label='Predicted vs True')
plt.plot(y_test, y_test, linestyle='--', color='red', label='x=y')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs True Values (Test Set)\nR-squared: {:.4f}'.format(r2_test))
plt.legend()
plt.grid(True)

# Add text for R-squared value in the plot
plt.text(0.5, 0.9, f'R-squared: {r2_test:.4f}', ha='center', va='center', transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', alpha=0.5))

# Save the plot as a .png file
plt.savefig('predicted_vs_true_test_set.png')
plt.show()

# Plot predicted values against true values in the index for the test set
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
plt.savefig('predicted_and_true_values_index_test_set.png')
plt.show()
