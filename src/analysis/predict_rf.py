import sys
import pandas as pd
import joblib
import yfinance as yf
import os
import warnings
import numpy as np

# This script uses the trained Random Forest model to predict stock prices.

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'valuation_model_rf.joblib')
REPORT_PATH = os.path.join(BASE_DIR, 'data', 'report_rf.csv')

def analyze_ticker(ticker, model, scaler, imputer):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        price = info.get('currentPrice')
        
        features = {
            'EPS': info.get('trailingEps'),
            'BookValue': info.get('bookValue'),
            'SalesPerShare': info.get('revenuePerShare'),
            'MarketCap': info.get('marketCap'),
            'AvgVolume': info.get('averageDailyVolume3Month'),
            'TotalDebt': info.get('totalDebt'),
            'FreeCashFlow': info.get('freeCashflow'),
            'OperatingCashFlow': info.get('operatingCashflow'),
            'ROE': info.get('returnOnEquity'),
            'ProfitMargin': info.get('profitMargins'),
            'RevenueGrowth': info.get('revenueGrowth'),
            'ForwardPE': info.get('forwardPE'),
            'PriceToBook': info.get('priceToBook'),
            'DebtToEquity': info.get('debtToEquity')
        }
        
        if price is None: return None

        # Convert to DataFrame (preserve None/NaN)
        df = pd.DataFrame([features])
        
        # Impute missing values
        if imputer:
            df_imputed = imputer.transform(df)
        else:
            df_imputed = df.fillna(0) # Fallback
            
        # Scale Features
        df_scaled = scaler.transform(df_imputed)
        
        # Predict
        prediction = model.predict(df_scaled)[0]
        
        if prediction < 0.01: prediction = 0.01
            
        upside = ((prediction - price) / price) * 100
        
        return {
            "Ticker": ticker,
            "Price": price,
            "PredictedPrice": round(prediction, 2),
            "UpsideIn%": round(upside, 2)
        }
    except Exception:
        return None

def generate_report(tickers):
    if not os.path.exists(MODEL_PATH):
        print("Error: Model not found. Please run train_rf.py first.")
        return

    # Load dictionary containing model, scaler, imputer
    saved_data = joblib.load(MODEL_PATH)
    
    if not isinstance(saved_data, dict):
         print("Error: Old model format detected. Please retrain train_rf.py")
         return

    model = saved_data['model']
    scaler = saved_data['scaler']
    imputer = saved_data.get('imputer')
    price_threshold = saved_data.get('price_threshold', 1000)
    
    print(f"Analyzing {len(tickers)} tickers with Random Forest (max price: ${price_threshold:.2f})...")
    
    results = []
    skipped = 0
    for i, ticker in enumerate(tickers):
        if i % 10 == 0: print(f"Processing {i}/{len(tickers)}: {ticker}...")
        data = analyze_ticker(ticker, model, scaler, imputer)
        if data:
            if data['Price'] > price_threshold: 
                skipped += 1
                continue
            results.append(data)
            
    if not results:
        print("No valid data found (or all skipped).")
        return

    df = pd.DataFrame(results)
    
    def recommend(row):
        if row['UpsideIn%'] > 20: return "BUY"
        if row['UpsideIn%'] < -20: return "SELL"
        return "HOLD"
        
    df['Recommendation'] = df.apply(recommend, axis=1)
    df = df.sort_values(by='UpsideIn%', ascending=False)
    
    with open(REPORT_PATH, 'w', newline='') as f:
        f.write("ALL STOCKS (Random Forest)\n")
        df.to_csv(f, index=False)
        
    print(f"Report saved to {REPORT_PATH}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_tickers = []
        for arg in sys.argv[1:]:
            input_tickers.extend(arg.split(','))
        generate_report(input_tickers)
