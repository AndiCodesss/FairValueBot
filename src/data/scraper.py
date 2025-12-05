import requests
from bs4 import BeautifulSoup
import sys

# This script scrapes the "Most Active" stocks from Yahoo Finance.
# It's a simple alternative to the Robot Framework scraper.

def get_active_tickers():
    url = "https://finance.yahoo.com/markets/stocks/most-active/?start=0&count=100"
    
    # We need a "User-Agent" so Yahoo thinks we are a real browser
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }
    
    try:
        response = requests.get(url, headers=headers)
        
        # Check if the request was successful
        if response.status_code != 200:
            print("Error: Could not access Yahoo Finance.")
            return []
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the table containing the stocks
        table = soup.find('table')
        
        tickers = []
        if table:
            rows = table.find_all('tr')
            
            # Loop through rows, skipping the header
            for row in rows[1:]:
                cols = row.find_all('td')
                if cols:
                    # The ticker is usually in the first column
                    ticker = cols[0].get_text(strip=True)
                    
                    # Sometimes there's extra text, so we just take the first word
                    ticker = ticker.split()[0]
                    tickers.append(ticker)
        
        return tickers

    except Exception as e:
        print(f"Error scraping tickers: {e}")
        return []

if __name__ == "__main__":
    tickers = get_active_tickers()
    
    # Print them separated by commas (useful for other scripts to read)
    print(",".join(tickers))
