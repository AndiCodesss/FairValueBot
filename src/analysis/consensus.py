import pandas as pd
import yfinance as yf
import os
import io

# This script generates the Final Consensus Report.
# It reads the reports from all 5 models and finds the stocks where multiple models agree.
# It also fetches extra financial data (Debt, Cash Flow) for the winners.

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
REPORT_LR = os.path.join(DATA_DIR, 'report_lr.csv')
REPORT_RF = os.path.join(DATA_DIR, 'report_rf.csv')
REPORT_ML = os.path.join(DATA_DIR, 'report_ml.csv')
REPORT_XGB = os.path.join(DATA_DIR, 'report_xgb.csv')
REPORT_DL = os.path.join(DATA_DIR, 'report_dl.csv')
OUTPUT_FILE = os.path.join(DATA_DIR, 'consensus_report.csv')

def get_financials(ticker):
    # Fetch extra details for the final report
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        def format_money(num):
            if num is None: return "N/A"
            if abs(num) >= 1e9: return f"${num/1e9:.2f}B"
            if abs(num) >= 1e6: return f"${num/1e6:.2f}M"
            return f"${num:.2f}"

        return {
            'TotalDebt': format_money(info.get('totalDebt')),
            'DebtToEquity': f"{info.get('debtToEquity', 0):.2f}" if info.get('debtToEquity') else "N/A",
            'FreeCashFlow': format_money(info.get('freeCashflow')),
            'OperatingCashFlow': format_money(info.get('operatingCashflow'))
        }
    except Exception:
        return {'TotalDebt': "Error", 'DebtToEquity': "Error", 'FreeCashFlow': "Error", 'OperatingCashFlow': "Error"}

def parse_report(file_path):
    # Reads a report CSV and skips the header lines
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found.")
        return None
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Find where the data starts
    start_index = -1
    for i, line in enumerate(lines):
        if "ALL STOCKS" in line:
            start_index = i + 1
            break
            
    if start_index == -1: return None
        
    csv_content = "".join(lines[start_index:])
    try:
        return pd.read_csv(io.StringIO(csv_content))
    except Exception:
        return None

def get_top_bottom(df, model_name):
    # Get the top 5 buys and bottom 5 sells from a model
    df = df.sort_values(by='UpsideIn%', ascending=False)
    top = df.head(5).copy()
    bottom = df.tail(5).copy()
    top['Model'] = model_name
    bottom['Model'] = model_name
    return top, bottom

def generate_consensus_data():
    print("Reading reports from all models...")
    df_lr = parse_report(REPORT_LR)
    df_rf = parse_report(REPORT_RF)
    df_ml = parse_report(REPORT_ML)
    df_xgb = parse_report(REPORT_XGB)
    df_dl = parse_report(REPORT_DL)
    
    if any(df is None for df in [df_lr, df_rf, df_ml, df_xgb, df_dl]):
        print("Error: Missing one or more reports.")
        return None, None

    # Get Top/Bottom picks from everyone
    top_lr, bot_lr = get_top_bottom(df_lr, 'Linear Regression')
    top_rf, bot_rf = get_top_bottom(df_rf, 'Random Forest')
    top_ml, bot_ml = get_top_bottom(df_ml, 'Neural Net')
    top_xgb, bot_xgb = get_top_bottom(df_xgb, 'XGBoost')
    top_dl, bot_dl = get_top_bottom(df_dl, 'Deep Learning')
    
    all_tops = pd.concat([top_lr, top_rf, top_ml, top_xgb, top_dl])
    all_bots = pd.concat([bot_lr, bot_rf, bot_ml, bot_xgb, bot_dl])
    
    # Find Consensus Winners (Stocks picked by multiple models)
    def find_consensus(df, recommendation):
        counts = df['Ticker'].value_counts()
        consensus = []
        for ticker, count in counts.items():
            rows = df[df['Ticker'] == ticker]
            models = ", ".join(rows['Model'].unique())
            avg_upside = rows['UpsideIn%'].mean()
            price = rows.iloc[0]['Price']
            
            consensus.append({
                'Ticker': ticker,
                'Votes': count,
                'Models': models,
                'AvgUpside%': round(avg_upside, 2),
                'Price': price,
                'Recommendation': recommendation
            })
        return pd.DataFrame(consensus)

    df_winners = find_consensus(all_tops, 'STRONG BUY')
    df_losers = find_consensus(all_bots, 'STRONG SELL')
    
    return df_winners, df_losers

def main():
    df_winners, df_losers = generate_consensus_data()
    
    if df_winners is None: return

    # Add extra financial data
    print("\nFetching extra details for Winners...")
    winner_details = [get_financials(t) for t in df_winners['Ticker']]
    df_winners = pd.concat([df_winners, pd.DataFrame(winner_details)], axis=1)
    
    print("Fetching extra details for Losers...")
    loser_details = [get_financials(t) for t in df_losers['Ticker']]
    df_losers = pd.concat([df_losers, pd.DataFrame(loser_details)], axis=1)
    
    # Save Final Report
    with open(OUTPUT_FILE, 'w', newline='') as f:
        f.write("CONSENSUS WINNERS\n")
        df_winners.to_csv(f, index=False)
        f.write("\nCONSENSUS LOSERS\n")
        df_losers.to_csv(f, index=False)
        
    print(f"\nSuccess! Final report saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
