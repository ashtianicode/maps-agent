import yfinance as yf
import requests
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
from agents import function_tool

POLYGON_API_KEY = "cvBzL74Jt9Z0bK4ptDHXoV2On_8PYbxM"

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

def get_options_contracts(
    underlying_symbol: Optional[str] = None,
    expiration_date: Optional[str] = None,
    contract_type: Optional[str] = None,
    strike_price_min: Optional[float] = None,
    strike_price_max: Optional[float] = None,
    limit: Optional[int] = None
) -> Dict[str, Union[List[Dict], str]]:
    """
    Fetch options contract data from Polygon.io API.
    
    Args:
        underlying_symbol: Stock ticker symbol (e.g., 'NVDA')
        expiration_date: Filter by expiration date (YYYY-MM-DD format)
        contract_type: Filter by contract type ('call' or 'put')
        strike_price_min: Minimum strike price
        strike_price_max: Maximum strike price
        limit: Maximum number of contracts to return (default 10)
        
    Returns:
        Dict containing options contracts data or error message
    """
    try:
        # Build the API URL with parameters
        base_url = "https://api.polygon.io/v3/reference/options/contracts"
        params = {
            "apiKey": POLYGON_API_KEY,
            "order": "asc",
            "sort": "ticker",
            "limit": limit if limit is not None else 10
        }
        
        # Add optional filters
        if underlying_symbol:
            params["underlying_ticker"] = underlying_symbol
        if expiration_date:
            params["expiration_date"] = expiration_date
        if contract_type:
            params["contract_type"] = contract_type.lower()
        if strike_price_min is not None:
            params["strike_price.gte"] = strike_price_min
        if strike_price_max is not None:
            params["strike_price.lte"] = strike_price_max
            
        # Make the API request
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Process and return the results
        if data.get("status") == "OK":
            contracts = data.get("results", [])
            return {
                "contracts": contracts,
                "error": None
            }
        else:
            return {
                "contracts": [],
                "error": "Failed to fetch options data"
            }
            
    except Exception as e:
        return {
            "contracts": [],
            "error": f"Error fetching options data: {str(e)}"
        } 