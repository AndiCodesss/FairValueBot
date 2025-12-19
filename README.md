# Fair Value Finder Bot (Council of Five Edition)

## Introduction
This bot automates a sophisticated Value Investing strategy using a "Council of Five" AI models to determine the intrinsic value of a stock. It combines **Robot Framework** for RPA (scraping and reporting) and **Python** (Scikit-Learn, XGBoost, PyTorch) for Machine Learning.

## Features
- **Powerful Data Engine**: Fetches **14 Key Financial Metrics** (EPS, Book Value, Sales, Market Cap, Volume, Debt, Cash Flow, ROE, Margins, Growth, PE, PB, Debt/Equity) using `yfinance`.
- **Council of Five**: Uses an ensemble of 5 models to analyze every stock:
    1.  **Linear Regression** (`_lr`): For simple linear relationships.
    2.  **Random Forest** (`_rf`): For complex, non-linear patterns.
    3.  **Neural Network** (`_ml`): Sklearn MLPRegressor for pattern recognition.
    4.  **XGBoost** (`_xgb`): Gradient Boosting for high-performance tabular prediction.
    5.  **Deep Learning** (`_dl`): Custom PyTorch Feed-Forward Network.
- **Consensus Voting**: Identifies "Strong Buy" signals only when multiple models agree.
- **RPA Automation**: Scrapes active tickers from Yahoo Finance using Robot Framework and a custom **AnalysisLibrary**.

## Prerequisites
- Python 3.9+
- Google Chrome (for Selenium)

## Installation
1.  Clone this repository and open the folder.
2.  Create the Conda environment:
    ```bash
    conda env create -f environment.yaml
    conda activate rpaproject
    ```
3.  Or, if you prefer pip with a virtual environment:
    ```bash
    python -m venv .venv
    .\.venv\Scripts\activate   # Windows
    pip install rpaframework pandas scikit-learn joblib yfinance xgboost torch
    ```

## Usage (The Easy Way)
Run the master script to fetch data, retrain models, and generate reports in one go:
```bash
python main.py
```

## Usage (Manual Steps)
1.  **Train the Models**:
    ```bash
    python src/training/train_lr.py
    python src/training/train_rf.py
    python src/training/train_ml.py
    python src/training/train_xgb.py
    python src/training/train_dl.py
    ```

2.  **Run the Bot (RPA)**:
    ```bash
    robot -d output robots/tasks.robot
    ```
    This will:
    - Open Yahoo Finance.
    - Scrape active tickers.
    - Analyze them using ALL 5 models.
    - Generate individual reports (`data/report_*.csv`).

3.  **Generate Consensus Report**:
    ```bash
    python src/analysis/consensus.py
    ```
    Check `data/consensus_report.csv` for the final winners.

## Assumptions & Design Decisions
- **Data Source**: We assume `yfinance` data is accurate. Missing values and infinite values (e.g., division by zero) are handled by filling with 0, assuming that missing/undefined financial data often implies "negligible" or "not applicable" for that metric.
- **Model Independence**: Each model is trained independently on the same dataset (`data/training_data.csv`) to ensure diversity in opinion.
- **Consensus Logic**: A stock is only a "Winner" if it has >20% upside potential according to the specific model. The final consensus requires votes from multiple models to filter out noise.
- **Market Efficiency**: The bot assumes that market inefficiencies exist and that fundamental metrics can predict fair value better than current market price.

## Folder Structure
- `data/`: Contains reports and `training_data.csv`.
- `models/`: Stores the 5 trained brains (`valuation_model_*.joblib/pth`).
- `robots/`: Contains `tasks.robot` (The main RPA process) and `AnalysisLibrary.py`.
- `src/`:
    - `data/`: Scripts for fetching and cleaning data.
    - `training/`: Scripts for training the 5 models.
    - `analysis/`: Scripts for prediction and consensus generation.
- `main.py`: Master script to run the entire pipeline.
