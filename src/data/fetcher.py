import yfinance as yf
import pandas as pd
import os
import random
import requests
from io import StringIO

# This script downloads financial data from Yahoo Finance.
# We use this data to train our machine learning models.

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)
DATA_PATH = os.path.join(DATA_DIR, 'training_data.csv')

def get_tickers_from_wikipedia(url):
    # Helper function to scrape ticker symbols from Wikipedia tables
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        table = pd.read_html(StringIO(response.text))
        df = table[0]
        tickers = df['Symbol'].tolist()
        # Fix symbols (e.g., BRK.B -> BRK-B)
        return [t.replace('.', '-') for t in tickers]
    except Exception as e:
        print(f"Error fetching tickers from {url}: {e}")
        return []

def fetch_data():
    print("Step 1: Getting Ticker Lists...")
    
    # We want a massive dataset to train our models.
    # We fetch tickers from multiple official Wikipedia sources:
    # 1. S&P 500 (Large Cap)
    # 2. S&P 400 (Mid Cap)
    # 3. S&P 600 (Small Cap)
    # 4. Nasdaq 100 (Tech)
    # 5. Dow Jones (Blue Chip)
    
    sp500 = get_tickers_from_wikipedia('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    sp400 = get_tickers_from_wikipedia('https://en.wikipedia.org/wiki/List_of_S%26P_400_companies')
    sp600 = get_tickers_from_wikipedia('https://en.wikipedia.org/wiki/List_of_S%26P_600_companies')
    nasdaq100 = get_tickers_from_wikipedia('https://en.wikipedia.org/wiki/Nasdaq-100')
    dow30 = get_tickers_from_wikipedia('https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average')
    
    # Wikipedia doesn't have a single list of all 3000+ US stocks.
    # So we also pull a massive raw list from GitHub to ensure we have enough volume.
    def get_all_us_tickers():
        try:
            url = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/all/all_tickers.txt"
            response = requests.get(url)
            if response.status_code == 200:
                # The file has one ticker per line
                return response.text.splitlines()
        except Exception as e:
            print(f"Error fetching GitHub list: {e}")
        return []

    github_tickers = get_all_us_tickers()
    print(f"Fetched {len(github_tickers)} extra tickers from GitHub!")
    
    # Combine them all
    all_tickers = sp600 + sp400 + sp500 + nasdaq100 + dow30 + github_tickers
    
    # Remove duplicates
    all_tickers = list(dict.fromkeys(all_tickers))
    
    # Limit to 5000 stocks to get more training data
    # We prioritize the small/mid caps (since they are first in the list)
    selected_tickers = all_tickers[:5000]
    
    # Shuffle them so we train on a random mix
    random.shuffle(selected_tickers)
    
    print(f"Step 2: Fetching data for {len(selected_tickers)} stocks...")
    
    # Delete old file if it exists so we start fresh
    if os.path.exists(DATA_PATH):
        os.remove(DATA_PATH)
        
    batch_data = []
    total_saved = 0
    
    for i, ticker in enumerate(selected_tickers):
        try:
            # Print progress every 10 stocks
            if i % 10 == 0:
                print(f"Processing {i}/{len(selected_tickers)}: {ticker}...")
                
            stock = yf.Ticker(ticker)
            info = stock.info
            
            price = info.get('currentPrice')
            
            # If we don't have a price, we can't use this stock for training
            if price is None:
                continue
                
            # These are the 14 features we use for prediction
            # IMPORTANT: We do NOT use 0 as a default anymore. We want NaNs.
            # Convert Nones to None (explicitly) so Pandas handles them as NaNs.
            
            row = {
                'Ticker': ticker,
                'Price': price,
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
            
            # Note: We removed the loop that forced valid Nones into 0s.
            # Pandas will save these as empty strings (NaN) in the CSV.
            
            batch_data.append(row)
            
            # Save every 500 stocks (Chunking)
            if len(batch_data) >= 500:
                df = pd.DataFrame(batch_data)
                # If file doesn't exist, write header. If it does, append without header.
                header = not os.path.exists(DATA_PATH)
                df.to_csv(DATA_PATH, mode='a', header=header, index=False)
                total_saved += len(batch_data)
                print(f"  -> Saved chunk of {len(batch_data)} stocks (Total: {total_saved})")
                batch_data = [] # Clear batch
            
        except Exception:
            # Just skip if there's an error
            continue
            
    # Save any remaining data
    if batch_data:
        df = pd.DataFrame(batch_data)
        header = not os.path.exists(DATA_PATH)
        df.to_csv(DATA_PATH, mode='a', header=header, index=False)
        total_saved += len(batch_data)
        print(f"  -> Saved final chunk of {len(batch_data)} stocks")

    print(f"\nSuccess! Saved total of {total_saved} stocks to {DATA_PATH}")

if __name__ == "__main__":
    fetch_data()
