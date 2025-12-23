import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# This script trains a Random Forest model.
# Random Forest uses many "Decision Trees" to make a more accurate prediction.


# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'training_data.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'valuation_model_rf.joblib')
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

def train():
    print("Training Random Forest Model...")
    
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
    
    # HANDLE MISSING DATA
    # 0. Replace Infinity with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # 1. Drop rows where we have absolutely NO valid info or target
    df = df.dropna(subset=['MarketCap', 'Price'])
    
    X = df[features]
    y = df[target]
    
    # FILTER OUTLIERS
    price_threshold = np.percentile(y, 99)
    valid_indices = y <= price_threshold
    X = X[valid_indices]
    y = y[valid_indices]
    print(f"Filtered outliers > ${price_threshold:.2f}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Impute with Median
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    # 3. Scale Features (StandardScaler - optional for RF but good for consistency)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    # Train the model with 100 trees
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    predictions = model.predict(X_test_scaled)
    
    from sklearn.metrics import mean_absolute_error
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f"\nLast 20 Predictions:")
    for i in range(20):
        print(f"Real: {y_test.values[i]:8.2f} vs Pred: {predictions[i]:8.2f} (Diff: {predictions[i]-y_test.values[i]:.2f})")
        
    print(f"\nModel Trained!")
    print(f"MSE: {mse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R2 Score: {r2:.2f}")
    
    # Save Dictionary
    joblib.dump({
        'model': model, 
        'scaler': scaler, 
        'imputer': imputer, 
        'price_threshold': price_threshold
    }, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

if __name__ == "__main__":
    train()
