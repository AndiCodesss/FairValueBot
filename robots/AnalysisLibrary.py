import sys
import os

# Ensure we can import from src (project root)
# This library is in <ROOT>/robots/
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# Import the existing analysis modules
from src.analysis import predict_lr
from src.analysis import predict_rf
from src.analysis import predict_ml
from src.analysis import predict_xgb
from src.analysis import predict_dl

class AnalysisLibrary:
    """
    Custom library to run stock analysis models from Robot Framework.
    """

    def run_linear_regression(self, tickers):
        """Runs the Linear Regression analsyis."""
        print(f"Starting Linear Regression Analysis for {len(tickers)} tickers...")
        predict_lr.generate_report(tickers)

    def run_random_forest(self, tickers):
        """Runs the Random Forest analysis."""
        print(f"Starting Random Forest Analysis for {len(tickers)} tickers...")
        predict_rf.generate_report(tickers)

    def run_neural_network(self, tickers):
        """Runs the Neural Network (MLP) analysis."""
        print(f"Starting Neural Network Analysis for {len(tickers)} tickers...")
        predict_ml.generate_report(tickers)

    def run_xgboost(self, tickers):
        """Runs the XGBoost analysis."""
        print(f"Starting XGBoost Analysis for {len(tickers)} tickers...")
        predict_xgb.generate_report(tickers)

    def run_deep_learning(self, tickers):
        """Runs the Deep Learning (PyTorch) analysis."""
        print(f"Starting Deep Learning Analysis for {len(tickers)} tickers...")
        predict_dl.generate_report(tickers)
