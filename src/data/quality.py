import pandas as pd
import os

# This script checks our training data for missing values (NaNs).
# It's important to know if our data has holes in it!

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'training_data.csv')

def check_data_quality():
    print("Checking Data Quality...")
    
    if not os.path.exists(DATA_PATH):
        print("Error: Data file not found.")
        return

    # Load the data
    df = pd.read_csv(DATA_PATH)
    
    total_rows = len(df)
    print(f"Total Stocks: {total_rows}")
    print("-" * 40)
    print(f"{'Column':<20} | {'Missing':<10} | {'% Missing':<10}")
    print("-" * 40)
    
    # Count missing values (NaNs) AND Zeros (which we used to fill NaNs)
    # Since we filled NaNs with 0 in fetcher.py, a 0 usually means missing data.
    missing_counts = df.isna().sum()
    zero_counts = (df == 0).sum()
    
    for col in df.columns:
        # Total "bad" data = NaNs + Zeros
        bad_data = missing_counts[col] + zero_counts[col]
        percent = (bad_data / total_rows) * 100
        
        # Only print if there are actually missing values, or for key columns
        print(f"{col:<20} | {bad_data:<10} | {percent:.2f}%")

if __name__ == "__main__":
    check_data_quality()
