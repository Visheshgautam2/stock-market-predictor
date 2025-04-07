# 📈 Stock Price Prediction System

A Flask-based web application that predicts stock prices using machine learning and technical analysis. The system provides real-time predictions with confidence scores and historical data visualization.

---

## 🧠 Overview

This application combines **machine learning models** and **technical indicators** to analyze and predict stock prices. Users can interact with a simple web interface to get predictions, examine historical trends, and visualize technical indicators.

---

## 📁 Project Structure

Project.1/
├── stock_prediction/
│   ├── __init__.py         # Application factory and configuration
│   ├── data_collector.py   # Stock data fetching using yfinance
│   ├── data_processor.py   # Technical indicator calculations
│   ├── prediction_engine.py # ML model for predictions
│   └── routes.py           # API endpoints
├── static/                 # Frontend assets
│   └── index.html         # Web interface
└── app.py                 # Application entry point


---

## 🔧 Key Components

### 1. Data Collection (`data_collector.py`)
- Fetches historical stock data using **yfinance**
- Validates ticker symbols and data quality
- Collects market indicators for context

### 2. Data Processing (`data_processor.py`)
- Calculates technical indicators: **SMA**, **EMA**, **RSI**, **MACD**, volume
- Handles data cleaning and normalization

### 3. Prediction Engine (`prediction_engine.py`)
- Implements a machine learning model
- Generates stock price predictions
- Computes confidence scores
- Provides market trend analysis

### 4. Web Interface
- Real-time stock data visualization
- Interactive prediction results
- Historical price charts and technical indicators

---

## 🌟 Features

- ✅ Real-time stock price predictions
- 📊 Technical analysis integration
- 📈 Historical data visualization
- 🔍 Confidence scoring system
- 📉 Market trend analysis
- 💻 Interactive and clean web interface

---

## 🧪 Technology Stack

- **Backend:** Python, Flask
- **Data Analysis:** pandas, numpy
- **Machine Learning:** scikit-learn
- **Stock Data:** yfinance
- **Frontend:** HTML, JavaScript, Bootstrap

---

## 🚀 Usage

1. Enter a stock symbol (e.g., `AAPL`, `MSFT`)
2. View current price and ML-based prediction
3. Analyze prediction confidence scores
4. Examine historical trends
5. Review technical indicators like RSI, MACD, etc.

---

> 📌 This system combines technical analysis and machine learning to offer in-depth stock prediction and market insights.

