import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
    
    X = df[features].values
    y = df[target].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale Data (Required for PyTorch too)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to PyTorch Tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)
    
    # Create Data Loader for batching
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize Model
    model = ValuationNet(input_size=len(features))
    criterion = nn.MSELoss() # Mean Squared Error Loss
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
            
    # Evaluate
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor).numpy()
        
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f"Model Trained! MSE: {mse:.2f}, R2 Score: {r2:.2f}")
    
    # Save Model and Scaler
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'input_size': len(features)
    }
    torch.save(checkpoint, MODEL_PATH)
    print(f"Saved PyTorch model to {MODEL_PATH}")

if __name__ == "__main__":
    train()
