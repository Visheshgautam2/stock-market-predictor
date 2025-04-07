from flask import current_app, jsonify, send_from_directory
from datetime import datetime
import logging
import os

logger = logging.getLogger(__name__)

def init_routes(app):
    @app.route('/')
    def index():
        return send_from_directory(app.static_folder, 'index.html')

    @app.route('/static/<path:path>')
    def serve_static(path):
        return send_from_directory(app.static_folder, path)

    @app.route('/health')
    def health_check():
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat()
        })

    @app.route('/predict/<ticker>')
    def predict_stock(ticker):
        try:
            ticker = ticker.strip().upper()
            if not ticker or len(ticker) > 10:
                raise ValueError('Invalid ticker symbol')

            logger.info(f"Starting prediction process for {ticker}")

            # First try to get stock data
            stock_data = current_app.data_collector.get_stock_data(ticker)
            if stock_data is None:
                logger.error(f"Failed to fetch data for {ticker}")
                return jsonify({
                    'error': f'Unable to fetch data for {ticker}. Please verify the ticker symbol and try again.',
                    'current_price': 0,
                    'predicted_price': 0,
                    'confidence': 0,
                    'prediction_date': datetime.now().strftime('%Y-%m-%d')
                }), 404

            # Log successful data fetch
            logger.info(f"Successfully fetched {len(stock_data)} records for {ticker}")

            # Process the data
            current_price = float(stock_data['Close'].iloc[-1])
            historical_prices = stock_data['Close'].tolist()
            historical_dates = [d.strftime('%Y-%m-%d') for d in stock_data.index]

            # Calculate technical indicators
            processed_data = current_app.data_processor.calculate_technical_indicators(stock_data)
            if processed_data is None:
                raise ValueError('Failed to calculate technical indicators')

            # Get market data and merge
            market_data = current_app.data_collector.get_market_indicators()
            if market_data is not None:
                merged_data = current_app.data_processor.merge_with_market_data(processed_data, market_data)
            else:
                logger.warning("Market data unavailable, proceeding with stock data only")
                merged_data = processed_data

            # Prepare features
            features, dates, feature_columns = current_app.data_processor.prepare_features(merged_data)
            if features is None:
                raise ValueError('Failed to prepare features')

            # Generate prediction
            prediction = current_app.prediction_engine.predict(merged_data, feature_columns)
            if prediction is None:
                raise ValueError('Failed to generate prediction')

            # Add historical data to response
            prediction.update({
                'historical_prices': historical_prices,
                'historical_dates': historical_dates,
                'ticker': ticker,
                'last_updated': datetime.now().isoformat()
            })

            logger.info(f"Successfully generated prediction for {ticker}")
            return jsonify(prediction)

        except ValueError as ve:
            logger.error(f"Validation error for {ticker}: {str(ve)}")
            return jsonify({
                'error': str(ve),
                'current_price': current_price if 'current_price' in locals() else 0.0,
                'predicted_price': 0.0,
                'confidence': 0.35,
                'prediction_date': datetime.now().strftime('%Y-%m-%d')
            }), 400

        except Exception as e:
            logger.error(f"Unexpected error processing {ticker}: {str(e)}")
            return jsonify({
                'error': f"An unexpected error occurred while processing {ticker}",
                'current_price': current_price if 'current_price' in locals() else 0.0,
                'predicted_price': 0.0,
                'confidence': 0.35,
                'prediction_date': datetime.now().strftime('%Y-%m-%d')
            }), 500

    @app.route('/analyze/<ticker>')
    def analyze_stock(ticker):
        try:
            return jsonify({
                'ticker': ticker,
                'analysis': f'Technical analysis for {ticker}',
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500