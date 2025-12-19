import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# This script trains an XGBoost model.
# XGBoost (Extreme Gradient Boosting) is one of the most powerful models for tabular data.
# It builds trees one by one, where each new tree tries to fix the errors of the previous ones.

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'training_data.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'valuation_model_xgb.joblib')
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

def train():
    print("Training XGBoost Model...")
    
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
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize XGBoost
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f"Model Trained! MSE: {mse:.2f}, R2 Score: {r2:.2f}")
    
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

if __name__ == "__main__":
    train()
