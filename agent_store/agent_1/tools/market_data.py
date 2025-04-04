import yfinance as yf
from agents import function_tool

@function_tool
def get_market_data(ticker: str) -> dict:
    """
    Get market data required for options calculations.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'TSLA' for Tesla)
    
    Returns:
        dict: Dictionary containing current stock price, volatility, and risk-free rate
    """
    try:
        # Get stock data
        stock = yf.Ticker(ticker)
        
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