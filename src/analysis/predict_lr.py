import sys
import pandas as pd
import joblib
import yfinance as yf
import os
import warnings

# This script uses the trained Linear Regression model to predict stock prices.
# It takes a list of tickers as input and saves a report to CSV.

warnings.filterwarnings("ignore")

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'valuation_model_lr.joblib')
REPORT_PATH = os.path.join(BASE_DIR, 'data', 'report_lr.csv')

def analyze_ticker(ticker, model, scaler):
    # Fetches data for a single ticker and predicts its price
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        price = info.get('currentPrice')
        
        # We need the same 14 features we used for training
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
        
        if price is None:
            return None

        # Clean up Nones
        for key in features:
            if features[key] is None:
                features[key] = 0
                
        # Create DataFrame for prediction
        df = pd.DataFrame([features])
        
        # Scale the features
        df_scaled = scaler.transform(df)
        
        # Predict
        prediction = model.predict(df_scaled)[0]
        
        # Ensure prediction isn't negative (stocks can't be negative)
        if prediction < 0.01:
            prediction = 0.01
            
        # Calculate Upside Potential
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

    # Load dictionary containing model, scaler, and price threshold
    saved_data = joblib.load(MODEL_PATH)
    model = saved_data['model']
    scaler = saved_data['scaler']
    price_threshold = saved_data.get('price_threshold', 1000)  # Default if not saved
    
    print(f"Analyzing {len(tickers)} tickers with Linear Regression (max price: ${price_threshold:.2f})...")
    results = []
    skipped = 0
    
    for i, ticker in enumerate(tickers):
        if i % 10 == 0:
            print(f"Processing {i}/{len(tickers)}: {ticker}...")
        
        data = analyze_ticker(ticker, model, scaler)
        if data:
            # Skip stocks above the training threshold
            if data['Price'] > price_threshold:
                skipped += 1
                continue
            results.append(data)
            
    if not results:
        print("No valid data found.")
        return

    df = pd.DataFrame(results)
    
    # Add simple recommendation
    def recommend(row):
        if row['UpsideIn%'] > 20: return "BUY"
        if row['UpsideIn%'] < -20: return "SELL"
        return "HOLD"
        
    df['Recommendation'] = df.apply(recommend, axis=1)
    
    # Sort by best upside
    df = df.sort_values(by='UpsideIn%', ascending=False)
    
    # Save report
    with open(REPORT_PATH, 'w', newline='') as f:
        f.write("ALL STOCKS (Linear Regression)\n")
        df.to_csv(f, index=False)
        
    print(f"Report saved to {REPORT_PATH}")

if __name__ == "__main__":
    # Get tickers from command line arguments
    if len(sys.argv) > 1:
        # Tickers are passed as a single string separated by commas, or multiple args
        input_tickers = []
        for arg in sys.argv[1:]:
            input_tickers.extend(arg.split(','))
        generate_report(input_tickers)
    else:
        print("Please provide tickers as arguments.")
