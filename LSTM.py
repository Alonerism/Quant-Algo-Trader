# This script trains an LSTM model to predict future stock prices based on historical data.
# It performs hyperparameter tuning to find the best model, predicts closing prices over different 
# time horizons (e.g., 1, 2, 5, 10 days), and evaluates the model using RMSE.
# The predicted values are saved alongside actual closing prices and volume, 
# enhancing the dataset with future price predictions for better trading insights.

#FILL IN WHAT THE MODEL READS IN AND WHERE IT OUTPUTS

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import matplotlib.pyplot as plt

# Load the data
file_path = 'FILL IN'
data = pd.read_csv(file_path)

# Convert timestamp to datetime
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Sort data by timestamp
data = data.sort_values('timestamp')

# Select the 'close' column
close_prices = data['close'].values.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_close_prices = scaler.fit_transform(close_prices)

# Prepare sequences
def create_sequences(data, seq_length, prediction_steps):
    xs, ys = [], []
    for i in range(len(data) - seq_length - prediction_steps):
        x = data[i:i+seq_length]
        y = data[i+seq_length:i+seq_length+prediction_steps]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 50
prediction_steps = 10
X, y = create_sequences(scaled_close_prices, seq_length, prediction_steps)

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).squeeze()

# Split into training and testing sets
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Create DataLoaders
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"CUDA is available. GPU: {torch.cuda.get_device_name(0)}")

# Define a smaller hyperparameter grid around the best parameters
param_grid = {
    'hidden_dim': [80, 90, 100],
    'num_layers': [1, 2, 3],
    'learning_rate': [0.0005, 0.001, 0.0015],
    'batch_size': [16, 32, 64]
}

# Grid search
best_model_avg_rmse = None
best_model_next_day_rmse = None
best_params_avg_rmse = None
best_params_next_day_rmse = None
best_avg_rmse = float('inf')
best_next_day_rmse = float('inf')

for params in ParameterGrid(param_grid):
    print(f'Training with params: {params}')
    
    # Create the model
    model = LSTMModel(input_dim=1, hidden_dim=params['hidden_dim'], output_dim=prediction_steps, num_layers=params['num_layers']).to(device)
    
    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    
    # Create DataLoader with batch_size
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)
    
    # Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Evaluate the model
    model.eval()
    with torch.no_grad():
        y_pred = []
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            y_pred.extend(outputs.cpu().numpy())

    # Calculate RMSE for each prediction horizon
    y_pred = np.array(y_pred)
    y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(y_pred.shape)
    y_test_scaled = scaler.inverse_transform(y_test.cpu().numpy().reshape(-1, 1)).reshape(y_test.shape)

    rmses = [np.sqrt(mean_squared_error(y_test_scaled[:, i], y_pred[:, i])) for i in range(prediction_steps)]
    avg_rmse = np.mean(rmses)
    next_day_rmse = rmses[0]
    print(f'RMSEs: {rmses}, Avg RMSE: {avg_rmse:.4f}, Next Day RMSE: {next_day_rmse:.4f}')
    
    if avg_rmse < best_avg_rmse:
        best_avg_rmse = avg_rmse
        best_model_avg_rmse = model
        best_params_avg_rmse = params

    if next_day_rmse < best_next_day_rmse:
        best_next_day_rmse = next_day_rmse
        best_model_next_day_rmse = model
        best_params_next_day_rmse = params

print(f'Best params for Avg RMSE: {best_params_avg_rmse}, Best Avg RMSE: {best_avg_rmse:.4f}')
print(f'Best params for Next Day RMSE: {best_params_next_day_rmse}, Best Next Day RMSE: {best_next_day_rmse:.4f}')

# Plot predicted vs actual values for each prediction horizon for the best model by Avg RMSE
best_model_avg_rmse.eval()
with torch.no_grad():
    y_pred = []
    for inputs, _ in test_loader:
        inputs = inputs.to(device)
        outputs = best_model_avg_rmse(inputs)
        y_pred.extend(outputs.cpu().numpy())

# Inverse transform the predictions to get the actual values
y_pred = np.array(y_pred)
y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(y_pred.shape)
y_test_scaled = scaler.inverse_transform(y_test.cpu().numpy().reshape(-1, 1)).reshape(y_test.shape)

# Print the head of the predictions next to the closing price
print("Predictions vs Actuals:")
print(pd.DataFrame({'Actual': y_test_scaled[:, 0], 'Predicted': y_pred[:, 0]}).head())

# Save the data containing only: timestamp, closing price, the 1, 2, 5, 10 day predictions, and volume
predicted_columns = ['Predicted_1_day', 'Predicted_2_day', 'Predicted_5_day', 'Predicted_10_day']
predictions_df = pd.DataFrame(y_pred[:, [0, 1, 4, 9]], columns=predicted_columns)

result_df = data[['timestamp', 'close', 'volume']].iloc[seq_length+prediction_steps:].reset_index(drop=True)
result_df = pd.concat([result_df, predictions_df.reset_index(drop=True)], axis=1)

# Save the resulting DataFrame to a CSV file
result_df.to_csv('FILL IN', index=False)

print("Data saved to 'predicted_stock_prices.csv'")