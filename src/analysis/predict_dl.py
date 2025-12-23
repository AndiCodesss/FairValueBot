import sys
import yfinance as yf
import torch
import torch.nn as nn
import os
import warnings
import pandas as pd
import numpy as np

# This script uses the trained Deep Learning model (PyTorch) to predict stock prices.

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'valuation_model_dl.pth')
REPORT_PATH = os.path.join(BASE_DIR, 'data', 'report_dl.csv')

# We need to define the class again to load it
class ValuationNet(nn.Module):
    def __init__(self, input_size):
        super(ValuationNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.output(x)
        return x

def analyze_ticker(ticker, model, scaler_x, scaler_y):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        price = info.get('currentPrice')
        
        features_dict = {
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

        features_list = []
        for key in features_dict:
            val = features_dict[key]
            features_list.append(val if val is not None else 0)
            
        # Scale features using scaler_x
        features_array = np.array([features_list])
        features_scaled = scaler_x.transform(features_array)
        features_tensor = torch.FloatTensor(features_scaled)
        
        # Predict (outputs Z-score)
        model.eval()
        with torch.no_grad():
            z_score = model(features_tensor).numpy()
            # Convert Z-score back to real price using scaler_y
            prediction = scaler_y.inverse_transform(z_score)[0][0]
        
        if prediction < 0.01: prediction = 0.01
            
        upside = ((prediction - price) / price) * 100
        
        return {
            "Ticker": ticker,
            "Price": price,
            "PredictedPrice": round(float(prediction), 2),
            "UpsideIn%": round(upside, 2)
        }
    except Exception:
        return None

def generate_report(tickers):
    if not os.path.exists(MODEL_PATH):
        print("Error: Model not found.")
        return

    # Load Checkpoint (weights_only=False is needed for the scalers)
    checkpoint = torch.load(MODEL_PATH, weights_only=False)
    model_state = checkpoint['model_state_dict']
    scaler_x = checkpoint['scaler_x']
    scaler_y = checkpoint['scaler_y']
    input_size = checkpoint['input_size']
    price_threshold = checkpoint.get('price_threshold', 1000)  # Default if not saved
    
    model = ValuationNet(input_size)
    model.load_state_dict(model_state)
    
    print(f"Analyzing {len(tickers)} tickers with Deep Learning (max price: ${price_threshold:.2f})...")
    
    results = []
    skipped = 0
    for i, ticker in enumerate(tickers):
        if i % 10 == 0: print(f"Processing {i}/{len(tickers)}: {ticker}...")
        data = analyze_ticker(ticker, model, scaler_x, scaler_y)
        if data:
            # Skip stocks above the training threshold
            if data['Price'] > price_threshold:
                skipped += 1
                continue
            results.append(data)
            
    if not results: return

    df = pd.DataFrame(results)
    
    def recommend(row):
        if row['UpsideIn%'] > 20: return "BUY"
        if row['UpsideIn%'] < -20: return "SELL"
        return "HOLD"
        
    df['Recommendation'] = df.apply(recommend, axis=1)
    df = df.sort_values(by='UpsideIn%', ascending=False)
    
    with open(REPORT_PATH, 'w', newline='') as f:
        f.write("ALL STOCKS (Deep Learning)\n")
        df.to_csv(f, index=False)
        
    print(f"Report saved to {REPORT_PATH}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_tickers = []
        for arg in sys.argv[1:]:
            input_tickers.extend(arg.split(','))
        generate_report(input_tickers)
