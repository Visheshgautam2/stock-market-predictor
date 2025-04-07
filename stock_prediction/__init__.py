from flask import Flask
from flask_cors import CORS
from .data_collector import DataCollector
from .data_processor import DataProcessor
from .prediction_engine import PredictionEngine

def create_app():
    app = Flask(__name__, static_folder='../static', static_url_path='')
    CORS(app)
    
    # Initialize components
    app.data_collector = DataCollector()
    app.data_processor = DataProcessor()
    app.prediction_engine = PredictionEngine()
    
    # Register routes
    from .routes import init_routes
    init_routes(app)
    
    return app

__all__ = ['app']