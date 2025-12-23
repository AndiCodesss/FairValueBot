import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# This script trains a Neural Network (MLP).
# Neural Networks are inspired by the human brain and can learn very complex patterns.
# Neural Networks need data to be "Scaled" (numbers between -1 and 1 usually).

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'training_data.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'valuation_model_ml.joblib')
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

def train():
    print("Training Neural Network (MLP)...")
    
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
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the data! This is crucial for Neural Nets.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Scale the Target (Log Transform) to handle large price ranges
    y_train_log = np.log1p(y_train)

    # Create a network with 2 hidden layers (100 neurons, then 50 neurons)
    model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train_log)
    
    # Predict in Log Space, then convert back
    log_predictions = model.predict(X_test_scaled)
    predictions = np.expm1(log_predictions)
    
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f"Model Trained! MSE: {mse:.2f}, R2 Score: {r2:.2f}")
    
    # Save model, scaler, and price threshold for inference
    joblib.dump({'model': model, 'scaler': scaler, 'price_threshold': price_threshold}, MODEL_PATH)
    print(f"Saved model and scaler to {MODEL_PATH}")

if __name__ == "__main__":
    train()
