import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import logging

class DataCollector:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def get_stock_data(self, ticker, period="1mo"):
        """Collect historical stock data using yfinance"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=45)

            # Configure ticker
            stock = yf.Ticker(ticker)
            
            # Validate ticker
            try:
                info = stock.info
                if not info:
                    self.logger.error(f"Invalid ticker: {ticker}")
                    return None
            except Exception as e:
                self.logger.error(f"Error validating ticker {ticker}: {e}")
                return None

            # Get historical data
            df = stock.history(
                start=start_date,
                end=end_date,
                interval='1d',
                auto_adjust=True
            )

            if df.empty:
                self.logger.error(f"No data received for {ticker}")
                return None

            # Validate data quality
            if len(df) < 5:
                self.logger.error(f"Insufficient data points for {ticker}")
                return None

            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_columns):
                self.logger.error(f"Missing required columns for {ticker}")
                return None

            # Clean data
            df = df.dropna()
            
            self.logger.info(f"Successfully fetched {len(df)} records for {ticker}")
            return df

        except Exception as e:
            self.logger.error(f"Error fetching data for {ticker}: {str(e)}")
            return None

    def get_market_indicators(self):
        """Get market data (SPY) for broader market context"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=45)
            
            spy = yf.Ticker("SPY")
            df = spy.history(
                start=start_date,
                end=end_date,
                interval='1d',
                auto_adjust=True
            )
            
            if df.empty:
                return None
                
            return df.dropna()
            
        except Exception as e:
            self.logger.error(f"Error fetching market data: {str(e)}")
            return None