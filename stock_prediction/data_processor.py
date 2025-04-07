import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.logger = logging.getLogger(__name__)

    def calculate_technical_indicators(self, df):
        """Calculate technical indicators with proper error handling"""
        try:
            # Verify we have required columns
            required = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required):
                self.logger.error("Missing required columns")
                return None

            # Forward fill any missing data
            df = df.fillna(method='ffill').fillna(method='bfill')

            # Calculate basic indicators
            df['SMA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
            df['SMA_50'] = df['Close'].rolling(window=50, min_periods=1).mean()
            df['EMA_12'] = df['Close'].ewm(span=12, adjust=False, min_periods=1).mean()
            df['EMA_26'] = df['Close'].ewm(span=26, adjust=False, min_periods=1).mean()
            
            # RSI with error handling
            try:
                df['RSI'] = self._calculate_rsi(df['Close'])
            except Exception as e:
                self.logger.warning(f"Error calculating RSI: {e}")
                df['RSI'] = 50  # neutral value
            
            # MACD
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False, min_periods=1).mean()
            
            # Volume indicators
            df['Volume_SMA'] = df['Volume'].rolling(window=20, min_periods=1).mean()
            df['OBV'] = self._calculate_obv(df)
            
            # Price action
            df['ROC'] = df['Close'].pct_change(periods=10) * 100
            
            # Verify calculations
            if df.isnull().any().any():
                self.logger.warning("NaN values found after calculations")
                df = df.fillna(method='ffill').fillna(0)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in technical indicators: {e}")
            return None

    def prepare_features(self, df):
        """Prepare and validate features"""
        try:
            feature_columns = [
                'Open', 'High', 'Low', 'Close', 'Volume',
                'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
                'RSI', 'MACD', 'Signal_Line',
                'Volume_SMA', 'OBV', 'ROC'
            ]
            
            # Add market features if available
            if 'Market_Return' in df.columns:
                feature_columns.append('Market_Return')
            if 'Relative_Strength' in df.columns:
                feature_columns.append('Relative_Strength')
            
            # Validate data
            if df is None or df.empty:
                raise ValueError("Empty DataFrame provided")
            
            # Ensure all columns exist
            missing = [col for col in feature_columns if col not in df.columns]
            if missing:
                raise ValueError(f"Missing columns: {missing}")
            
            # Clean and prepare features
            features_df = df[feature_columns].copy()
            features_df = features_df.replace([np.inf, -np.inf], np.nan)
            features_df = features_df.fillna(method='ffill').fillna(0)
            
            # Scale features
            scaled_features = self.scaler.fit_transform(features_df)
            return scaled_features, df.index, feature_columns
            
        except Exception as e:
            self.logger.error(f"Error preparing features: {e}")
            return None, None, None

    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI with error handling"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        except Exception as e:
            self.logger.error(f"Error calculating RSI: {e}")
            return pd.Series(50, index=prices.index)  # neutral RSI

    def _calculate_obv(self, df):
        """Calculate On-Balance Volume"""
        try:
            return (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        except Exception as e:
            self.logger.error(f"Error calculating OBV: {e}")
            return pd.Series(0, index=df.index)

    def merge_with_market_data(self, stock_data, market_data):
        """Merge stock and market data safely"""
        try:
            if market_data is None:
                return stock_data
                
            market_returns = market_data['Close'].pct_change().fillna(0)
            stock_data['Market_Return'] = market_returns
            stock_data['Relative_Strength'] = stock_data['Close'].pct_change().fillna(0) - market_returns
            
            return stock_data
            
        except Exception as e:
            self.logger.error(f"Error merging market data: {e}")
            return stock_data