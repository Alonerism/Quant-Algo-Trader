import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima.arima import auto_arima, ndiffs
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  # Add these imports

# Load the dataset
print("Loading dataset...")
file_path = 'FILL IN DATA'
data = pd.read_csv(file_path)

# Parse the timestamp column as datetime
print("Parsing timestamp column...")
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Sort data by timestamp
print("Sorting data by timestamp...")
data = data.sort_values('timestamp')

# Drop rows with missing values
print("Dropping rows with missing values...")
data.dropna(inplace=True)

# Determine the optimal number of differences
print("Determining the optimal number of differences using ndiffs...")
n_diffs = ndiffs(data['close'], test='adf')
print(f'Optimal number of differences: {n_diffs}')

# Apply the differencing
print(f"Applying differencing with n_diffs = {n_diffs}...")
data['close_diff'] = data['close'].diff(n_diffs)

# Remove NaN values resulting from differencing
data.dropna(subset=['close_diff'], inplace=True)

# Check the data after differencing
print("First few rows of the original data:")
print(data[['timestamp', 'close']].head())
print("First few rows of the differenced data:")
print(data[['timestamp', 'close_diff']].head())

# Split the data into train and test sets
print("Splitting data into train and test sets...")
train_size = int(len(data) * 0.8)
train, test = data.iloc[:train_size], data.iloc[train_size:]

# Perform grid search for the best p, d, q parameters using auto_arima
print("Performing grid search for the best p, d, q parameters using auto_arima...")
model = auto_arima(train['close'],
                   start_p=1, max_p=10,  # Expanded range
                   start_q=1, max_q=10,  # Expanded range
                   d=n_diffs,
                   start_P=0, max_P=3,  # Seasonal components
                   start_Q=0, max_Q=3,  # Seasonal components
                   D=1, max_D=2,  # Seasonal differencing
                   m=24,  # Seasonal period (assuming daily seasonality in hourly data)
                   seasonal=True,
                   trace=True,
                   error_action='ignore',
                   suppress_warnings=True,
                   stepwise=True)

# Fit the best model
print("Fitting the best model...")
model_fit = model.fit(train['close'])

# Print the best parameters
print("Best model summary:")
print(model.summary())

# Make predictions
print("Making predictions on the test set...")
predictions = model.predict(n_periods=len(test))

# Evaluate the model using RMSE
print("Evaluating the model using RMSE...")
rmse = np.sqrt(mean_squared_error(test['close'], predictions))
print(f'Root Mean Squared Error: {rmse}')

# Merge predictions with test set for comparison
print("Merging predictions with the test set for comparison...")
test = test.copy()
test['predicted_close'] = predictions

# Plot actual vs predicted closing prices
print("Plotting actual vs predicted closing prices...")
plt.figure(figsize=(14, 7))
plt.plot(test['timestamp'], test['close'], label='Actual Close Price')
plt.plot(test['timestamp'], test['predicted_close'], label='Predicted Close Price', linestyle='--')
plt.xlabel('Timestamp')
plt.ylabel('Close Price')
plt.title('Actual vs Predicted Close Prices')
plt.legend()
plt.show()

# Checking for stationarity using Augmented Dickey-Fuller test
print("Checking for stationarity using Augmented Dickey-Fuller test...")
result = adfuller(data['close_diff'].dropna())
print('ADF Statistic (Differenced):', result[0])
print('p-value (Differenced):', result[1])
for key, value in result[4].items():
    print(f'Critical Values (Differenced): {key}, {value}')

# Residual Analysis
print("Performing residual analysis...")
residuals = model_fit.resid()

# Plot residuals
plt.figure(figsize=(14, 7))
plt.plot(residuals)
plt.title('Residuals of the ARIMA Model')
plt.show()

# Plot ACF of residuals
plot_acf(residuals, lags=30)
plt.title('ACF of Residuals')
plt.show()

# Plot PACF of residuals
plot_pacf(residuals, lags=30)
plt.title('PACF of Residuals')
plt.show()

# Ljung-Box test
ljung_box_result = acorr_ljungbox(residuals, lags=[10], return_df=True)
print("Ljung-Box test results:")
print(ljung_box_result)