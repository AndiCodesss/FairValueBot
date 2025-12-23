import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
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
    
    # HANDLE MISSING DATA PROPERLY
    # 0. Replace Infinity with NaN (SimpleImputer doesn't handle Inf)
    df = df.replace([np.inf, -np.inf], np.nan)

    # 1. Drop rows where we have absolutely NO valid info or target
    # MarketCap is critical context; Price is the target.
    df = df.dropna(subset=['MarketCap', 'Price'])
    
    # 2. Impute missing feature values with the MEDIAN
    # We use the median to minimize the impact of outliers on the central tendency.
    from sklearn.impute import SimpleImputer
    
    X = df[features]
    y = df[target]
    
    # filter outliers
    # Use 99th percentile as threshold - adapts automatically to any dataset.
    price_threshold = np.percentile(y, 99)
    # Get indices of valid data
    valid_indices = y <= price_threshold
    X = X[valid_indices]
    y = y[valid_indices]
    print(f"Filtered outliers > ${price_threshold:.2f}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Impute Missing Values (Learn from Train, Apply to Test)
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    # Scale Features (Robust to outliers)
    from sklearn.preprocessing import PowerTransformer, QuantileTransformer
    
    scaler_x = PowerTransformer()
    X_train_scaled = scaler_x.fit_transform(X_train_imputed)
    X_test_scaled = scaler_x.transform(X_test_imputed)
    
    # Scale Target (Robust to outliers)
    # Using normal distribution output helps the neural net significantly
    scaler_y = QuantileTransformer(output_distribution='normal', n_quantiles=min(len(y_train), 1000))
    y_train_transformed = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()

    # Create a network with 2 hidden layers (128 neurons, then 64 neurons)
    # Increased size slightly to match DL model capacity
    model = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=2000, random_state=42, learning_rate_init=0.001)
    model.fit(X_train_scaled, y_train_transformed)
    
    # Predict (outputs Z-score)
    z_score_predictions = model.predict(X_test_scaled)
    
    # Inverse transform to get real prices
    # Reshape is needed for inverse_transform
    predictions = scaler_y.inverse_transform(z_score_predictions.reshape(-1, 1)).flatten()
    
    # Ensure non-negative
    predictions = np.clip(predictions, 0, None)
    
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f"Model Trained! MSE: {mse:.2f}, R2 Score: {r2:.2f}")
    
    # Save model, scalers, IMPUTER, and price threshold for inference
    # Note: We save scaler_x as 'scaler' for compatibility with some predict logic, 
    # but we ALSO save scaler_y so we can inverse transform.
    joblib.dump({
        'model': model, 
        'scaler': scaler_x,      # Feature scaler (compatible name)
        'scaler_y': scaler_y,    # Target scaler (new)
        'imputer': imputer, 
        'price_threshold': price_threshold
    }, MODEL_PATH)
    print(f"Saved model, scalers, and imputer to {MODEL_PATH}")

if __name__ == "__main__":
    train()
