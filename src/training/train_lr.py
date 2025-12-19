import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# This script trains a Linear Regression model.
# Linear Regression tries to find a straight line that best fits the data.

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'training_data.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'valuation_model_lr.joblib')
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

def train():
    print("Training Linear Regression Model...")
    
    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        print("Error: Data file not found. Please run fetch_data.py first.")
        return

    df = pd.read_csv(DATA_PATH)
    
    # 2. Define Features (X) and Target (y)
    # We use these 14 financial metrics to predict the Price
    features = [
        'EPS', 'BookValue', 'SalesPerShare', 'MarketCap', 'AvgVolume', 
        'TotalDebt', 'FreeCashFlow', 'OperatingCashFlow', 'ROE', 
        'ProfitMargin', 'RevenueGrowth', 'ForwardPE', 'PriceToBook', 'DebtToEquity'
    ]
    target = 'Price'
    
    # Fill missing values with 0
    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)
    
    X = df[features]
    y = df[target]
    
    # 3. Split into Training and Testing sets
    # We train on 80% of the data and test on the other 20%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Train the Model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 5. Evaluate
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f"Model Trained! MSE: {mse:.2f}, R2 Score: {r2:.2f}")
    
    # 6. Save the Model
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

if __name__ == "__main__":
    train()
