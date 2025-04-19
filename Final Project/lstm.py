import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import itertools

# Load dataset
def load_data():
    """Load and preprocess the dataset."""

    df = pd.read_csv("hf://datasets/misikoff/SPX/^SPX.csv")
    df['Date'] = pd.to_datetime(df['Date'])

    df['Date_Original'] = df['Date']
    df.set_index('Date', inplace=True)
    df = df.sort_index()

    df = df[['Date_Original', 'Close']]
    return df

# Perform EDA and Handle Missing Data
def perform_eda(df):
    """Perform exploratory data analysis on the dataset."""

    print("Dataset Overview:\n", df.head())
    print("\n\nSummary Statistics:\n", df.describe())

    # Plot Closing Price over Time
    plt.figure(figsize=(12,6))
    plt.plot(df.index, df['Close'], label='Closing Price')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.title('SPX Closing Price Over Time')
    plt.legend()
    plt.show()

    return df

# Train-Test Split
def train_test_split(df, test_ratio=0.2):
    """Split the dataset into training and testing sets."""

    split_idx = int(len(df) * (1 - test_ratio))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    return train_df, test_df

# Prepare Data for PyTorch
def prepare_data(df, seq_length=50, scaler=None, fit=True):
    """Prepare data for PyTorch model."""

    if scaler is None:
        scaler = MinMaxScaler()

    if fit:
        data_scaled = scaler.fit_transform(df[['Close']])
    else:
        data_scaled = scaler.transform(df[['Close']])

    # Create sequences
    X, y = [], []
    for i in range(len(data_scaled) - seq_length):
        X.append(data_scaled[i:i+seq_length])
        y.append(data_scaled[i+seq_length])
    
    # Convert to tensors
    X, y = np.array(X), np.array(y)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # Create DataLoader for batching
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=32, shuffle=False), scaler

# Define LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, dropout=0.2):
        """Initialize LSTM model."""
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        """Forward pass for the model."""
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
    
    # Train Model
def train_model(model, train_loader, num_epochs=30, lr=0.001, device='cpu'):
    """Train the RNN model."""

    criterion = nn.MSELoss() # Mean Squared Error Loss
    optimizer = optim.Adam(model.parameters(), lr=lr) # Adam Optimizer
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)
    epoch_losses = []
    best_loss = float('inf')
    patience_counter = 0
    patience_limit = 15

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        total_loss = 0.0  # Initialize total loss for the epoch

        # Iterate over batches
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            # Forward pass
            outputs = model(X_batch)
            # Compute loss
            loss = criterion(outputs, y_batch)
            # Backward pass and optimization
            loss.backward()
            # Update weights
            optimizer.step()
            total_loss += loss.item()  # Accumulate batch loss

        avg_loss = total_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print("Early stopping triggered.")
                break
        scheduler.step(avg_loss)

    # Plot training loss
    plt.figure()
    plt.plot(epoch_losses)
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

def evaluate_model(model, test_loader, scaler, device='cpu'):
    """Evaluate the model on the test set."""

    model.to(device)
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            output = model(X_batch)
            predictions.append(output.cpu().numpy())
            actuals.append(y_batch.cpu().numpy())

    predictions = np.concatenate(predictions).reshape(-1, 1)
    actuals = np.concatenate(actuals).reshape(-1, 1)

    predictions = scaler.inverse_transform(predictions)
    actuals = scaler.inverse_transform(actuals)

    mse = np.mean((predictions - actuals) ** 2)
    mae = np.mean(np.abs(predictions - actuals))
    print(f"Test MSE: {mse:.4f}")
    print(f"Test MAE: {mae:.4f}")

    plt.figure(figsize=(12,6))
    plt.plot(actuals, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.title('Actual vs Predicted Closing Prices on Test Set')
    plt.xlabel('Time Step')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def hyperparameter_tuning(train_df, device = 'cpu'):
    """Tune hyperparameters for the model."""

    param_grid = {
        'hidden_size': [256, 512],
        'num_layers': [1, 2],
        'dropout': [0.3, 0.5],
        'lr': [1e-3, 5e-4, 1e-4],
        'seq_length': [50, 70]
    }

    best_loss = float('inf')
    best_params = None

    # Split the training data into a smaller training set and validation set
    train_sub, val_sub = train_test_split(train_df, test_ratio=0.2)

    # Iterate over all combinations of hyperparameters
    for hidden_size, num_layers, dropout, lr, seq_length in itertools.product(
        param_grid['hidden_size'], param_grid['num_layers'],
        param_grid['dropout'], param_grid['lr'], param_grid['seq_length']
    ):
        print(f"Trying config: hidden_size={hidden_size}, num_layers={num_layers}, dropout={dropout}, lr={lr}, seq_length={seq_length}")
        train_loader, scaler = prepare_data(train_sub, seq_length, fit=True)
        val_loader, _ = prepare_data(val_sub, seq_length, scaler=scaler, fit=False)

        model = LSTMModel(hidden_size=hidden_size, num_layers=num_layers, dropout=dropout).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        for epoch in range(10):  # Short training during tuning
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
        
        # Evaluate on validation set
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        avg_loss = val_loss / len(val_loader)
        print(f"Avg Validation Loss: {avg_loss:.4f}")

        # Check if this is the best configuration
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_params = (hidden_size, num_layers, dropout, lr, seq_length)

    print("\nBest config:")
    print(f"hidden_size={best_params[0]}, num_layers={best_params[1]}, dropout={best_params[2]}, lr={best_params[3]}, seq_length={best_params[4]}")
    # Return the best parameters
    return best_params

# Main function to run the LSTM model
if __name__ == "__main__":
    # Set device to use GPU if available, otherwise fallback to CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load and preprocess data
    df = load_data()
    df = perform_eda(df)
    train_df, test_df = train_test_split(df)

    # Hyperparameter tuning to find the best configuration
    best_params = hyperparameter_tuning(train_df, device)

    # Train the model with the best parameters
    train_loader, scaler = prepare_data(train_df, seq_length=best_params[4])
    model = LSTMModel(hidden_size=best_params[0], num_layers=best_params[1], dropout=best_params[2]).to(device)
    train_model(model, train_loader, lr=best_params[3], device=device)

    print("\nModel Type: Standard LSTM")
    print(f"Hidden Size: {best_params[0]}")
    print(f"Num Layers: {best_params[1]}")
    print(f"Dropout: {best_params[2]}")
    print(f"Learning Rate: {best_params[3]}")
    print(f"Window Length: {best_params[4]}")

    # Prepare test data for evaluation
    test_loader, _ = prepare_data(test_df, seq_length=best_params[4], scaler=scaler, fit=False)
    # Evaluate the model on the test set
    evaluate_model(model, test_loader, scaler)