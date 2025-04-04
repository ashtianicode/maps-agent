import yfinance as yf
from typing import Dict, Optional

def get_market_data(symbol: Optional[str] = None) -> Dict[str, float]:
    """
    Retrieve market data for a given stock symbol.
    
    Args:
        symbol: Stock ticker symbol
        
    Returns:
        Dict containing market data with keys:
        - stock_price: Current stock price
        - volatility: Current implied volatility
        - risk_free_rate: Current risk-free rate
    """
    # Set default values and validate required parameters
    if symbol is None:
        raise ValueError("Stock symbol is required")
    
    try:
        # Get stock data
        stock = yf.Ticker(symbol)
        
        # Get current price
        current_price = stock.info['regularMarketPrice']
        
        # Get historical data for volatility calculation (1 year)
        hist = stock.history(period='1y')
        
        # Calculate annualized volatility
        daily_returns = hist['Close'].pct_change().dropna()
        annual_volatility = daily_returns.std() * (252 ** 0.5)  # Annualize daily volatility
        
        # Use 10-year Treasury yield as risk-free rate (simplified)
        risk_free_rate = 0.0425  # 4.25% as of 2024
        
        return {
            "stock_price": current_price,
            "volatility": annual_volatility,
            "risk_free_rate": risk_free_rate
        }
    except Exception as e:
        return {
            "error": f"Failed to fetch market data: {str(e)}"
        } 