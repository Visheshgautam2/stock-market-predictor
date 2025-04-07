import anthropic
from datetime import datetime

class ClaudeAnalyzer:
    def __init__(self, api_key=None):
        self.client = anthropic.Client(api_key) if api_key else None

    def analyze_stock_data(self, ticker, stock_data, prediction, news_data=None, market_data=None):
        """Generate analysis using Claude AI"""
        if not self.client:
            return "API key not configured for Claude analysis"

        # Prepare context for analysis
        latest_price = stock_data['Close'][-1] if not stock_data.empty else None
        predicted_price = prediction.get('predicted_price') if prediction else None
        
        prompt = self._generate_analysis_prompt(
            ticker, latest_price, predicted_price, 
            stock_data, news_data, market_data
        )

        try:
            response = self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content
        except Exception as e:
            return f"Error generating analysis: {str(e)}"

    def _generate_analysis_prompt(self, ticker, current_price, predicted_price, 
                                stock_data, news_data, market_data):
        """Generate prompt for Claude analysis"""
        prompt = f"""Analyze the following stock data for {ticker}:
        Current Price: ${current_price:.2f}
        Predicted Price: ${predicted_price:.2f}
        
        Please provide:
        1. Technical Analysis
        2. Price Movement Analysis
        3. Market Context
        4. Risk Assessment
        5. Short-term Outlook
        
        Base your analysis on the provided data and current market conditions."""
        
        return prompt