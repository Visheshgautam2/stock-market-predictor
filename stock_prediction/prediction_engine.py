import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
from sklearn.model_selection import cross_val_score

class PredictionEngine:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=500,  # Increased for better accuracy
            max_depth=10,      # Prevent overfitting
            min_samples_split=5,
            random_state=42
        )
        self.is_trained = False
        self.feature_importances = {}
        self.historical_accuracy = []

    def predict(self, data, features):
        """Generate predictions for stock data"""
        try:
            # Validate input data
            if data is None or len(data) < 20:
                raise ValueError('Insufficient historical data for prediction')

            # Get current price
            current_price = float(data['Close'].iloc[-1])

            # Train model if needed
            if not self.is_trained:
                self._train_on_historical(data, features)

            # Prepare prediction data
            latest_data = data[features].iloc[-1:].fillna(method='ffill')
            
            # Generate prediction
            predicted_change = float(self.model.predict(latest_data)[0])
            predicted_price = float(current_price * (1 + predicted_change))
            
            # Calculate metrics
            trend_strength = float(self._calculate_trend_strength(data))
            support, resistance = self._calculate_support_resistance(data)
            confidence = float(self._calculate_enhanced_confidence(
                data, features, predicted_price, support, resistance, trend_strength
            ))

            prediction = {
                'current_price': current_price,
                'predicted_price': predicted_price,
                'predicted_change': predicted_change,
                'confidence': confidence,
                'trend_strength': trend_strength,
                'support_level': float(support),
                'resistance_level': float(resistance),
                'prediction_date': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
            }

            # Validate prediction values
            for key, value in prediction.items():
                if isinstance(value, (int, float)) and (np.isnan(value) or np.isinf(value)):
                    prediction[key] = 0.0

            return prediction

        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return {
                'current_price': current_price if 'current_price' in locals() else 0.0,
                'predicted_price': current_price if 'current_price' in locals() else 0.0,
                'predicted_change': 0.0,
                'confidence': 0.35,
                'trend_strength': 0.0,
                'support_level': 0.0,
                'resistance_level': 0.0,
                'prediction_date': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
                'error': str(e)
            }

    def _train_on_historical(self, data, features):
        # Calculate multiple timeframe returns
        data['Target_1d'] = data['Close'].pct_change().shift(-1)
        data['Target_5d'] = (data['Close'].shift(-5) / data['Close'] - 1)
        data['Target_10d'] = (data['Close'].shift(-10) / data['Close'] - 1)
        
        # Prepare training data
        y = data['Target_1d'].dropna()
        X = data[features].iloc[:-1]
        
        # Perform cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=5)
        self.historical_accuracy = cv_scores
        
        # Train final model
        self.model.fit(X, y)
        self.is_trained = True
        self.feature_importances = dict(zip(features, self.model.feature_importances_))

    def _calculate_trend_strength(self, data):
        # Calculate trend strength using multiple indicators
        sma_20 = data['Close'].rolling(window=20).mean()
        sma_50 = data['Close'].rolling(window=50).mean()
        
        # Trend direction and strength
        price_above_sma20 = data['Close'].iloc[-1] > sma_20.iloc[-1]
        price_above_sma50 = data['Close'].iloc[-1] > sma_50.iloc[-1]
        sma20_above_sma50 = sma_20.iloc[-1] > sma_50.iloc[-1]
        
        # Calculate momentum
        roc = (data['Close'].iloc[-1] - data['Close'].iloc[-20]) / data['Close'].iloc[-20]
        
        trend_score = sum([price_above_sma20, price_above_sma50, sma20_above_sma50]) / 3
        trend_strength = (trend_score + abs(roc)) / 2
        return min(1.0, max(0.0, trend_strength))

    def _calculate_support_resistance(self, data):
        # Calculate support and resistance using price clusters
        price_clusters = pd.qcut(data['Close'].tail(100), q=10)
        support = data['Close'].tail(100).min()
        resistance = data['Close'].tail(100).max()
        
        return support, resistance

    def _calculate_enhanced_confidence(self, data, features, predicted_price, support, resistance, trend_strength):
        """Calculate prediction confidence using multiple weighted factors"""
        if not self.is_trained:
            return 0.0
        
        # Model performance score (from cross-validation)
        model_score = np.mean(self.historical_accuracy)
        
        # Feature importance score
        feature_scores = list(self.feature_importances.values())
        importance_score = np.mean(feature_scores) * (1 + np.std(feature_scores))
        
        # Volume analysis
        recent_volume = data['Volume'].tail(5).mean()
        avg_volume = data['Volume'].tail(20).mean()
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 0.5
        volume_score = min(1.0, volume_ratio)
        
        # Price stability score
        recent_volatility = data['Close'].tail(10).pct_change().std()
        stability_score = 1 / (1 + recent_volatility)
        
        # Trend conviction score
        price = data['Close'].iloc[-1]
        sma_20 = data['Close'].rolling(window=20).mean().iloc[-1]
        sma_50 = data['Close'].rolling(window=50).mean().iloc[-1]
        trend_conviction = np.mean([
            1 if price > sma_20 else 0,
            1 if price > sma_50 else 0,
            1 if sma_20 > sma_50 else 0
        ])
        
        # Support/Resistance validation
        price_range = resistance - support
        if price_range > 0:
            price_position = (predicted_price - support) / price_range
            range_score = 1 - abs(0.5 - price_position)
        else:
            range_score = 0.5
        
        # RSI extremes check
        rsi = self._calculate_rsi(data['Close'])
        rsi_latest = rsi.iloc[-1]
        rsi_score = 1 - min(abs(50 - rsi_latest) / 50, 1)
        
        # Dynamic weights based on market conditions
        weights = {
            'model_score': 0.25,
            'importance_score': 0.15,
            'volume_score': 0.15,
            'stability_score': 0.15,
            'trend_conviction': 0.1,
            'range_score': 0.1,
            'rsi_score': 0.1
        }
        
        # Calculate final confidence
        confidence = (
            weights['model_score'] * model_score +
            weights['importance_score'] * importance_score +
            weights['volume_score'] * volume_score +
            weights['stability_score'] * stability_score +
            weights['trend_conviction'] * trend_conviction +
            weights['range_score'] * range_score +
            weights['rsi_score'] * rsi_score
        )
        
        # Scale confidence to be between 0.35 and 0.95
        scaled_confidence = 0.35 + (confidence * 0.60)
        return min(0.95, max(0.35, scaled_confidence))

    def _calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _get_model_metrics(self):
        return {
            'cv_mean_accuracy': float(np.mean(self.historical_accuracy)),
            'cv_std_accuracy': float(np.std(self.historical_accuracy)),
            'n_features': len(self.feature_importances),
            'top_features': dict(sorted(self.feature_importances.items(), 
                                     key=lambda x: x[1], reverse=True)[:3])
        }