import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from sklearn.metrics import mean_squared_error, r2_score
import os

# This script trains a Deep Learning model using PyTorch.
# PyTorch is a professional library for building custom Neural Networks.
# Here we build a "Feed-Forward" network with 3 layers.

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'training_data.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODELS_DIR, 'valuation_model_dl.pth')

# Define the Neural Network Architecture
class ValuationNet(nn.Module):
    def __init__(self, input_size):
        super(ValuationNet, self).__init__()
        # Layer 1: Input -> 128 neurons
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128), # Helps training stability
            nn.ReLU(),           # Activation function
            nn.Dropout(0.2)      # Prevents overfitting
        )
        # Layer 2: 128 -> 64 neurons
        self.layer2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        # Layer 3: 64 -> 32 neurons
        self.layer3 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU()
        )
        # Output Layer: 32 -> 1 (The Price Prediction)
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.output(x)
        return x

def train():
    print("Training PyTorch Deep Learning Model...")
    
    if not os.path.exists(DATA_PATH):
        print("Error: Data file not found.")
        return

    df = pd.read_csv(DATA_PATH)
    
    features = [
        'EPS', 'BookValue', 'SalesPerShare', 'MarketCap', 'AvgVolume', 
        'TotalDebt', 'FreeCashFlow', 'OperatingCashFlow', 'ROE', 
        'ProfitMargin', 'RevenueGrowth', 'ForwardPE', 'PriceToBook', 'DebtToEquity'
    ]
    target = 'Price'
    
    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)
    
    # Use 99th percentile as threshold - adapts automatically to any dataset.
    price_threshold = np.percentile(df[target], 99)
    original_count = len(df)
    df = df[df[target] <= price_threshold]
    filtered_count = len(df)
    print(f"Filtered data: {original_count} -> {filtered_count} samples (removed top 1% outliers > ${price_threshold:.2f})")
    
    X = df[features].values
    y = df[target].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale Features using PowerTransformer (handles skewed financial data)
    scaler_x = PowerTransformer()
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_test_scaled = scaler_x.transform(X_test)
    
    # Scale Target using QuantileTransformer (robust to remaining outliers via ranking)
    scaler_y = QuantileTransformer(output_distribution='normal', n_quantiles=min(len(y_train), 1000))
    y_train_transformed = scaler_y.fit_transform(y_train.reshape(-1, 1))
    
    # Convert to PyTorch Tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train_transformed)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    
    # Create Data Loader for batching
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize Model
    model = ValuationNet(input_size=len(features))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training Loop
    epochs = 200
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.4f}")
            
    # Evaluation
    model.eval()
    with torch.no_grad():
        prediction_z_scores = model(X_test_tensor).numpy()
        predictions = scaler_y.inverse_transform(prediction_z_scores)
        
    predictions = predictions.flatten()
    
    # Clamp to non-negative (prices can't be negative)
    predictions = np.clip(predictions, 0, None)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f"Model Trained! MSE: {mse:.2f}, R2 Score: {r2:.2f}")
    
    # Save Model, Scalers, and Price Threshold
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'scaler_x': scaler_x,
        'scaler_y': scaler_y,
        'input_size': len(features),
        'price_threshold': price_threshold
    }
    torch.save(checkpoint, MODEL_PATH)
    print(f"Saved PyTorch model to {MODEL_PATH}")

if __name__ == "__main__":
    train()
