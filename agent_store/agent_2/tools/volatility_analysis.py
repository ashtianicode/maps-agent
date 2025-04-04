import numpy as np
from scipy import optimize
import datetime
from typing import Dict, List, Optional, Union, Tuple
from agents import function_tool

# Import the Black-Scholes model from options_tools
from agent_store.agent_2.tools.options_tools import calculate_option_price

def calculate_implied_volatility(
    option_price: Optional[float] = None,
    underlying_price: Optional[float] = None,
    strike_price: Optional[float] = None,
    time_to_expiry: Optional[float] = None,
    risk_free_rate: Optional[float] = None,
    option_type: Optional[str] = None,
    dividend_yield: Optional[float] = None
) -> float:
    """
    Calculate the implied volatility for an option using a numerical method.
    
    This function uses a root-finding algorithm to determine the volatility
    that would make the Black-Scholes price equal to the observed market price.
    
    Args:
        option_price: The market price of the option
        underlying_price: The current price of the underlying asset
        strike_price: The strike price of the option
        time_to_expiry: Time to expiration in years (e.g., 0.25 for 3 months)
        risk_free_rate: The risk-free interest rate (annualized decimal, e.g., 0.03 for 3%)
        option_type: Either 'call' or 'put'
        dividend_yield: The dividend yield of the underlying asset (optional)
        
    Returns:
        The implied volatility as a decimal (e.g., 0.20 for 20% volatility)
    
    Example:
        Calculate implied volatility for a call option on AAPL priced at $8.50:
        >>> calculate_implied_volatility(8.50, 150.0, 145.0, 0.25, 0.03, 'call', 0.0)
        0.28
    """
    # Set default values and validate required parameters
    if option_price is None or underlying_price is None or strike_price is None or \
       time_to_expiry is None or risk_free_rate is None or option_type is None:
        raise ValueError("Missing required parameters for implied volatility calculation")
    
    option_type = option_type.lower()
    if option_type not in ["call", "put"]:
        raise ValueError("option_type must be either 'call' or 'put'")
    
    dividend_yield = dividend_yield or 0.0

    def objective_function(volatility):
        # Calculate option price using the Black-Scholes model with given volatility
        price = calculate_option_price(
            underlying_price=underlying_price,
            strike_price=strike_price,
            time_to_expiry=time_to_expiry,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            option_type=option_type,
            dividend_yield=dividend_yield
        )
        # Return the difference between the calculated price and the observed market price
        return price - option_price

    # Use a bounded root-finding method (bisection) to find implied volatility
    # Initial volatility search range: 0.001 to 4.0 (0.1% to 400%)
    try:
        implied_vol = optimize.bisect(objective_function, 0.001, 4.0, xtol=1e-6)
        return implied_vol
    except ValueError:
        # If the bisection method fails, try a broader range or another method
        try:
            result = optimize.minimize_scalar(
                lambda x: abs(objective_function(x)),
                bounds=(0.001, 10.0),
                method='bounded'
            )
            return result.x
        except:
            raise ValueError("Could not determine implied volatility. Check if option price is within valid range.")

def fetch_volatility_surface(
    symbol: Optional[str] = None,
    expiry_dates: Optional[List[str]] = None,
    strike_range: Optional[List[float]] = None,
) -> Dict[str, Dict[float, float]]:
    """
    Fetch the implied volatility surface for a given underlying asset.
    
    This function retrieves implied volatilities for various strike prices and
    expiry dates, creating a volatility surface visualization or dataset.
    
    Args:
        symbol: The ticker symbol of the underlying asset
        expiry_dates: Optional list of expiration dates in 'YYYY-MM-DD' format.
                      If None, will use standard monthly expirations
        strike_range: Optional list of [min_strike, max_strike] to limit the range.
                      If None, will use a reasonable range around current price
        
    Returns:
        A dictionary mapping expiry dates to dictionaries of {strike: implied_vol}
        
    Example:
        Fetch the volatility surface for AAPL:
        >>> vol_surface = fetch_volatility_surface('AAPL')
        >>> vol_surface['2023-12-15'][150.0]  # IV for Dec 15 expiry, $150 strike
        0.22
    """
    # Set default values and validate required parameters
    if symbol is None:
        raise ValueError("Symbol is required for fetching volatility surface")
    
    # In a real implementation, this would connect to a market data provider
    # For this example, we'll generate synthetic data
    
    # Mock current price and basic data
    # In production, get this from get_market_data or another source
    mock_current_price = 150.0
    mock_base_volatility = 0.25  # 25% base volatility
    
    # Generate default expiry dates if none provided
    if expiry_dates is None:
        today = datetime.date.today()
        expiry_dates = [
            (today + datetime.timedelta(days=30)).strftime('%Y-%m-%d'),  # 1 month
            (today + datetime.timedelta(days=60)).strftime('%Y-%m-%d'),  # 2 months
            (today + datetime.timedelta(days=90)).strftime('%Y-%m-%d'),  # 3 months
            (today + datetime.timedelta(days=180)).strftime('%Y-%m-%d'),  # 6 months
            (today + datetime.timedelta(days=365)).strftime('%Y-%m-%d'),  # 1 year
        ]
    
    # Generate default strike range if none provided
    if strike_range is None:
        min_strike = 0.7 * mock_current_price
        max_strike = 1.3 * mock_current_price
    else:
        min_strike, max_strike = strike_range
    
    # Generate a range of strikes
    strikes = np.linspace(min_strike, max_strike, 15)
    
    # Build the volatility surface
    vol_surface = {}
    
    for expiry in expiry_dates:
        vol_surface[expiry] = {}
        
        # Calculate days to expiry
        expiry_date = datetime.datetime.strptime(expiry, '%Y-%m-%d').date()
        days_to_expiry = (expiry_date - datetime.date.today()).days
        time_factor = days_to_expiry / 365.0  # Convert to years
        
        for strike in strikes:
            # Calculate moneyness (how far in/out of the money)
            moneyness = strike / mock_current_price
            
            # Implement a simple volatility smile:
            # - Higher implied vol for deep ITM and OTM options (smile shape)
            # - Volatility increasing with time to expiry (term structure)
            moneyness_factor = 1.5 * (moneyness - 1.0) ** 2  # Smile shape
            time_volatility = 0.05 * np.sqrt(time_factor)  # Term structure
            
            # Combine factors to get implied volatility
            implied_vol = mock_base_volatility + moneyness_factor + time_volatility
            
            # Add some randomness for realism
            noise = np.random.normal(0, 0.01)  # Small random noise
            implied_vol = max(0.05, implied_vol + noise)  # Ensure minimum volatility
            
            # Store in the surface dictionary
            vol_surface[expiry][float(strike)] = float(implied_vol)
    
    return vol_surface 