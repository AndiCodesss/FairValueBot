import subprocess
import sys

def main():
    # This script runs the entire project pipeline.
    # It trains the models, runs the robot to scrape data, and generates the final report.

    # Step 0: Fetch New Data
    # Uncomment these lines if you want to download fresh data from Yahoo Finance.
    print("Step 0: Fetching Data...")
    # subprocess.check_call([sys.executable, "src/data/fetcher.py"])
    # subprocess.check_call([sys.executable, "src/data/quality.py"])  # should be 0 since we fill nan with 0

    # Step 1: Train the Models
    # We need to make sure all 5 models are trained on the latest data.
    print("Step 1: Training models...")
    
    # Linear Regression (Simple baseline)
    #subprocess.check_call([sys.executable, "src/training/train_lr.py"])
    
    # Random Forest (Decision trees)
    #subprocess.check_call([sys.executable, "src/training/train_rf.py"])
    
    # Neural Network (Scikit-Learn MLP)
    #subprocess.check_call([sys.executable, "src/training/train_ml.py"])
    
    # XGBoost (Gradient Boosting - very powerful)
    #subprocess.check_call([sys.executable, "src/training/train_xgb.py"])
    
    # Deep Learning (PyTorch Custom Network)
    #subprocess.check_call([sys.executable, "src/training/train_dl.py"])

    # Step 2: Run the Robot
    # The robot scrapes Yahoo Finance for active tickers and runs the prediction scripts.
    print("\nStep 2: Running Robot Framework...")
    subprocess.check_call([sys.executable, "-m", "robot", "-d", "output", "robots/tasks.robot"])

    # Step 3: Generate Consensus Report
    # This script reads all the individual reports and finds the stocks where models agree.
    print("\nStep 3: Generating Final Consensus Report...")
    subprocess.check_call([sys.executable, "src/analysis/consensus.py"])

    print("\nDone! Please check 'data/consensus_report.csv' for the results.")

if __name__ == "__main__":
    main()
