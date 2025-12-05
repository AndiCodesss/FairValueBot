import sys
import pandas as pd
import joblib
import yfinance as yf
import os
import warnings

# This script uses the trained Neural Network (MLP) to predict stock prices.
# It loads both the model and the scaler.

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'valuation_model_ml.joblib')
REPORT_PATH = os.path.join(BASE_DIR, 'data', 'report_ml.csv')

def analyze_ticker(ticker, model, scaler):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        price = info.get('currentPrice')
        
        features = {
            'EPS': info.get('trailingEps', 0),
            'BookValue': info.get('bookValue', 0),
            'SalesPerShare': info.get('revenuePerShare', 0),
            'MarketCap': info.get('marketCap', 0),
            'AvgVolume': info.get('averageDailyVolume3Month', 0),
            'TotalDebt': info.get('totalDebt', 0),
            'FreeCashFlow': info.get('freeCashflow', 0),
            'OperatingCashFlow': info.get('operatingCashflow', 0),
            'ROE': info.get('returnOnEquity', 0),
            'ProfitMargin': info.get('profitMargins', 0),
            'RevenueGrowth': info.get('revenueGrowth', 0),
            'ForwardPE': info.get('forwardPE', 0),
            'PriceToBook': info.get('priceToBook', 0),
            'DebtToEquity': info.get('debtToEquity', 0)
        }
        
        if price is None: return None

        for key in features:
            if features[key] is None: features[key] = 0
                
        df = pd.DataFrame([features])
        
        # Scale the features!
        df_scaled = scaler.transform(df)
        
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
        print("Error: Model not found.")
        return

    # Load dictionary containing both model and scaler
    saved_data = joblib.load(MODEL_PATH)
    model = saved_data['model']
    scaler = saved_data['scaler']
    
    print(f"Analyzing {len(tickers)} tickers with Neural Network (MLP)...")
    
    results = []
    for i, ticker in enumerate(tickers):
        if i % 10 == 0: print(f"Processing {i}/{len(tickers)}: {ticker}...")
        data = analyze_ticker(ticker, model, scaler)
        if data: results.append(data)
            
    if not results: return

    df = pd.DataFrame(results)
    
    def recommend(row):
        if row['UpsideIn%'] > 20: return "BUY"
        if row['UpsideIn%'] < -20: return "SELL"
        return "HOLD"
        
    df['Recommendation'] = df.apply(recommend, axis=1)
    df = df.sort_values(by='UpsideIn%', ascending=False)
    
    with open(REPORT_PATH, 'w', newline='') as f:
        f.write("ALL STOCKS (Neural Network)\n")
        df.to_csv(f, index=False)
        
    print(f"Report saved to {REPORT_PATH}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_tickers = []
        for arg in sys.argv[1:]:
            input_tickers.extend(arg.split(','))
        generate_report(input_tickers)
