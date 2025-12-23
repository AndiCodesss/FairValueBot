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
    
    # HANDLE MISSING DATA PROPERLY
    # 0. Replace Infinity with NaN (SimpleImputer doesn't handle Inf)
    df = df.replace([np.inf, -np.inf], np.nan)

    # 1. Drop rows where we have absolutely NO valid info or target
    df = df.dropna(subset=['MarketCap', 'Price'])
    
    X = df[features].values
    y = df[target].values
    
    # FILTER OUTLIERS
    price_threshold = np.percentile(y, 99)
    valid_indices = y <= price_threshold
    X = X[valid_indices]
    y = y[valid_indices]
    print(f"Filtered outliers > ${price_threshold:.2f}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Impute Missing Values (Learn from Train, Apply to Test)
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    # Scale Features using PowerTransformer (handles skewed financial data)
    scaler_x = PowerTransformer()
    X_train_scaled = scaler_x.fit_transform(X_train_imputed)
    X_test_scaled = scaler_x.transform(X_test_imputed)
    
    # Scale Target using QuantileTransformer (robust to remaining outliers via ranking)
    scaler_y = QuantileTransformer(output_distribution='normal', n_quantiles=min(len(y_train), 1000))
    y_train_transformed = scaler_y.fit_transform(y_train.reshape(-1, 1))
    
    # Convert to PyTorch Tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train_transformed)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test) # Keep original price for evaluation

    # Initialize Model, Loss, Optimizer
    model = ValuationNet(X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train Loop
    epochs = 200
    batch_size = 32
    
    for epoch in range(epochs):
        model.train()
        
        # Mini-batch training
        permutation = torch.randperm(X_train_tensor.size()[0])
        
        epoch_loss = 0
        for i in range(0, X_train_tensor.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train_tensor[indices], y_train_tensor[indices]
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(X_train_tensor):.4f}")

    # Evaluate
    model.eval()
    with torch.no_grad():
        z_score_predictions = model(X_test_tensor).numpy()
        # Inverse transform to get real prices
        predictions = scaler_y.inverse_transform(z_score_predictions)
        
        # Ensure non-negative
        predictions = np.clip(predictions, 0, None)
        
        # Flatten for metric calculation
        predictions = predictions.flatten()
        
        # Print samples
        print("\nLast 20 Predictions (Cleaned):")
        for i in range(20):
             print(f"Real: {y_test[i]:8.2f} vs Pred: {predictions[i]:8.2f}")
    
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f"Model Trained! MSE: {mse:.2f}, R2 Score: {r2:.2f}")
    
    # Save Model, Scalers, Imputer, and Price Threshold
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'scaler_x': scaler_x,
        'scaler_y': scaler_y,
        'imputer': imputer,
        'input_size': len(features),
        'price_threshold': price_threshold
    }
    torch.save(checkpoint, MODEL_PATH)
    print(f"Saved PyTorch model to {MODEL_PATH}")

if __name__ == "__main__":
    train()
