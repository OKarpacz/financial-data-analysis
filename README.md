# Financial Data Analysis and Machine Learning 📈

This project involves fetching financial data, processing it, and applying machine learning techniques to predict stock price movements. It demonstrates the end-to-end process of working with stock market data from data retrieval to machine learning model evaluation.

## Features

- **Fetch S&P 500 Tickers**: 
  - Retrieves a list of S&P 500 tickers from Wikipedia.
  - Saves the list of tickers for future use.
  
- **Download Historical Stock Data**:
  - Fetches historical stock data from Yahoo Finance.
  - Saves the data in CSV files for each ticker.

- **Data Compilation**:
  - Compiles and merges the historical data of all tickers into a single DataFrame.
  - Saves the combined DataFrame for further analysis.

- **Data Visualization**:
  - Generates and visualizes correlation heatmaps of the stock data to understand relationships between different stocks.

- **Data Processing for Machine Learning**:
  - Processes the stock data to create features and labels for machine learning.
  - Includes calculations for price changes and targets for classification.

- **Machine Learning**:
  - Applies machine learning models to predict stock price movements.
  - Evaluates model performance and displays predictions.

## Requirements

- Python 3.x
- Libraries: `requests`, `beautifulsoup4`, `pandas`, `numpy`, `matplotlib`, `yfinance`, `sklearn`, `plotly`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/OKarpacz/financial-data-analysis.git
