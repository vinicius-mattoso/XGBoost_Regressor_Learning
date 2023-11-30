import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
data = pd.read_csv(url)
data['Month'] = pd.to_datetime(data['Month'])
data.set_index('Month', inplace=True)

# Create a Naive Forecast
def naive_forecast(data):
    # Shift the test data by one time step to get the next value
    forecast = data.shift(1)
    return forecast

# Prepare features and target variable
X = data.index.values.astype(float).reshape(-1, 1)
y = data['Passengers'].values

# Splitting the data into train and test sets
train_size = int(len(data) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Fit the Naive Forecast on the training set
y_train_naive = naive_forecast(pd.Series(y_train, index=data.index[:train_size]))

# Predict using Naive Forecast for the test set
y_test_naive = naive_forecast(pd.Series(y, index=data.index))[train_size:]

# Calculate RMSE for Naive Forecast on the test set
test_rmse_naive = np.sqrt(mean_squared_error(y_test[1:], y_test_naive[1:]))
print(f"Naive Forecast RMSE on Test Set: {test_rmse_naive}")

# Calculate R-squared for Naive Forecast on the test set
r2_naive_test = r2_score(y_test[1:], y_test_naive[1:])
print(f"Naive Forecast R-squared on Test Set: {r2_naive_test}")

# Create a plot for Naive Forecast against true values on the test set
plt.figure(figsize=(8, 6))
plt.scatter(y_test[1:], y_test_naive[1:], color='blue', label='Predicted vs True')
plt.plot(y_test[1:], y_test[1:], linestyle='--', color='red', label='x=y')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Naive Forecast Predicted vs True Values (Test Set)')
plt.text(0.5, 0.9, f'R-squared: {r2_naive_test:.4f}', ha='center', va='center', transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', alpha=0.5))
plt.legend()
plt.grid(True)
plt.savefig('predicted_vs_true_best_naive.png')
plt.show()


# Plot Naive Forecast values against true values along the index
plt.figure(figsize=(10, 6))
plt.plot(data.index[train_size + 1:], y_test[1:],'--o', label='True Values', color='blue')
plt.plot(data.index[train_size + 1:], y_test_naive[1:],'--o', label='Naive Forecast', color='red')
plt.xlabel('Index')
plt.ylabel('Passengers')
plt.title('Naive Forecast vs True Values (Test Set)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('predicted_and_true_best_naive.png')
plt.show()
