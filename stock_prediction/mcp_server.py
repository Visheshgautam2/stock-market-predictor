from flask import Flask, request, jsonify, send_from_directory, Blueprint, current_app
from flask_cors import CORS
import os
from datetime import datetime
import logging
import numpy as np
from .data_collector import DataCollector
from .data_processor import DataProcessor
from .prediction_engine import PredictionEngine
from .claude_analyzer import ClaudeAnalyzer
from .routes import init_routes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='../static')
CORS(app, resources={r"/*": {"origins": "*"}})

bp = Blueprint('mcp_server', __name__)

# Initialize routes
def init_app(app):
    init_routes(app)
    app.register_blueprint(bp)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/')
def serve_frontend():
    """Serve the frontend HTML"""
    return send_from_directory(app.static_folder, 'index.html', mimetype='text/html')

@app.route('/api')
def api_home():
    """Home endpoint"""
    return jsonify({
        'message': 'Stock Prediction API',
        'status': 'running',
        'timestamp': datetime.now().isoformat()
    })

# Initialize components globally
data_collector = DataCollector()
data_processor = DataProcessor()
prediction_engine = PredictionEngine()
claude_analyzer = ClaudeAnalyzer()

@app.route('/collect', methods=['POST'])
def collect_data():
    data = request.json
    collected_data = data_collector.collect(data)
    return jsonify(collected_data)

@app.route('/process', methods=['POST'])
def process_data():
    data = request.json
    processed_data = data_processor.process(data)
    return jsonify(processed_data)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    prediction = prediction_engine.predict(data)
    return jsonify(prediction)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    analysis = claude_analyzer.analyze(data)
    return jsonify(analysis)

@app.route('/predict/<ticker>')
def predict_stock(ticker):
    try:
        ticker = ticker.strip().upper()
        if not ticker or len(ticker) > 10:
            raise ValueError('Invalid ticker symbol')

        app.logger.info(f"Fetching data for {ticker}")
        stock_data = data_collector.get_stock_data(ticker)
        
        if stock_data is None or stock_data.empty:
            app.logger.error(f"No data available for {ticker}")
            return jsonify({
                'error': f'Unable to fetch current market data for {ticker}. The market might be closed or the ticker might be invalid.',
                'current_price': 0,
                'predicted_price': 0,
                'confidence': 0,
                'prediction_date': datetime.now().strftime('%Y-%m-%d')
            }), 404

        current_price = float(stock_data['Close'].iloc[-1])
        historical_prices = stock_data['Close'].tolist()
        historical_dates = [d.strftime('%Y-%m-%d') for d in stock_data.index]
        
        app.logger.info(f"Processing technical indicators for {ticker}")
        processed_data = data_processor.calculate_technical_indicators(stock_data)
        if processed_data is None:
            raise ValueError('Error calculating technical indicators')

        market_data = data_collector.get_market_indicators()
        merged_data = data_processor.merge_with_market_data(processed_data, market_data)
        
        app.logger.info(f"Preparing features for {ticker}")
        features, dates, feature_columns = data_processor.prepare_features(merged_data)
        if features is None:
            raise ValueError('Error preparing features for prediction')
            
        app.logger.info(f"Generating prediction for {ticker}")
        prediction = prediction_engine.predict(merged_data, feature_columns)
        if prediction is None:
            raise ValueError('Error generating prediction')
            
        prediction.update({
            'historical_prices': historical_prices,
            'historical_dates': historical_dates,
            'ticker': ticker,
        })
        
        app.logger.info(f"Successfully generated prediction for {ticker}")
        return jsonify(prediction)
        
    except Exception as e:
        app.logger.error(f"Error processing request for {ticker}: {str(e)}")
        return jsonify({
            'error': str(e),
            'current_price': current_price if 'current_price' in locals() else 0.0,
            'predicted_price': 0.0,
            'confidence': 0.35,
            'prediction_date': datetime.now().strftime('%Y-%m-%d')
        }), 400

@app.route('/analyze/<ticker>')
def analyze_stock(ticker):
    try:
        # Placeholder analysis
        return jsonify({
            'ticker': ticker,
            'analysis': 'Sample analysis for ' + ticker,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)