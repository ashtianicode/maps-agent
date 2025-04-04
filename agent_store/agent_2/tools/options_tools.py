import math
import numpy as np
from typing import Literal, Dict, Optional
from agents import function_tool

def calculate_option_price(
    option_type: Optional[str] = None,
    underlying_price: Optional[float] = None,
    strike_price: Optional[float] = None,
    time_to_expiry: Optional[float] = None,
    risk_free_rate: Optional[float] = None,
    volatility: Optional[float] = None,
    dividend_yield: Optional[float] = None
) -> float:
    """
    Calculate the theoretical price of a European option using the Black-Scholes model.
    
    Args:
        option_type: Type of option ('call' or 'put')
        underlying_price: Current price of the underlying asset
        strike_price: Strike price of the option
        time_to_expiry: Time to expiration in years
        risk_free_rate: Annual risk-free interest rate (as a decimal)
        volatility: Annual volatility of the underlying asset (as a decimal)
        dividend_yield: Annual dividend yield (as a decimal, optional)
        
    Returns:
        float: Theoretical price of the option
    """
    # Set default values and validate required parameters
    if option_type is None or underlying_price is None or strike_price is None or \
       time_to_expiry is None or risk_free_rate is None or volatility is None:
        raise ValueError("Missing required parameters for option price calculation")
    
    option_type = option_type.lower()
    if option_type not in ["call", "put"]:
        raise ValueError("option_type must be either 'call' or 'put'")
    
    dividend_yield = dividend_yield or 0.0
    
    d1 = (math.log(underlying_price / strike_price) + (risk_free_rate + volatility**2 / 2) * time_to_expiry) / (volatility * math.sqrt(time_to_expiry))
    d2 = d1 - volatility * math.sqrt(time_to_expiry)
    
    if option_type == "call":
        price = underlying_price * norm_cdf(d1) - strike_price * math.exp(-risk_free_rate * time_to_expiry) * norm_cdf(d2)
    else:  # put
        price = strike_price * math.exp(-risk_free_rate * time_to_expiry) * norm_cdf(-d2) - underlying_price * norm_cdf(-d1)
    
    return price

def calculate_option_greeks(
    option_type: Optional[str] = None,
    underlying_price: Optional[float] = None,
    strike_price: Optional[float] = None,
    time_to_expiry: Optional[float] = None,
    risk_free_rate: Optional[float] = None,
    volatility: Optional[float] = None,
    dividend_yield: Optional[float] = None
) -> Dict[str, float]:
    """
    Calculate the Greeks (delta, gamma, theta, vega, rho) for a European option.
    
    Args:
        option_type: Type of option ('call' or 'put')
        underlying_price: Current price of the underlying asset
        strike_price: Strike price of the option
        time_to_expiry: Time to expiration in years
        risk_free_rate: Annual risk-free interest rate (as a decimal)
        volatility: Annual volatility of the underlying asset (as a decimal)
        dividend_yield: Annual dividend yield (as a decimal, optional)
        
    Returns:
        Dict[str, float]: Dictionary containing the calculated Greeks
    """
    # Set default values and validate required parameters
    if option_type is None or underlying_price is None or strike_price is None or \
       time_to_expiry is None or risk_free_rate is None or volatility is None:
        raise ValueError("Missing required parameters for Greeks calculation")
    
    option_type = option_type.lower()
    if option_type not in ["call", "put"]:
        raise ValueError("option_type must be either 'call' or 'put'")
    
    dividend_yield = dividend_yield or 0.0
    
    d1 = (math.log(underlying_price / strike_price) + (risk_free_rate + volatility**2 / 2) * time_to_expiry) / (volatility * math.sqrt(time_to_expiry))
    d2 = d1 - volatility * math.sqrt(time_to_expiry)
    
    # Calculate delta
    if option_type == "call":
        delta = norm_cdf(d1)
    else:  # put
        delta = norm_cdf(d1) - 1
    
    # Calculate gamma (same for calls and puts)
    gamma = norm_pdf(d1) / (underlying_price * volatility * math.sqrt(time_to_expiry))
    
    # Calculate theta
    if option_type == "call":
        theta = (-underlying_price * norm_pdf(d1) * volatility / (2 * math.sqrt(time_to_expiry)) 
                - risk_free_rate * strike_price * math.exp(-risk_free_rate * time_to_expiry) * norm_cdf(d2))
    else:  # put
        theta = (-underlying_price * norm_pdf(d1) * volatility / (2 * math.sqrt(time_to_expiry))
                + risk_free_rate * strike_price * math.exp(-risk_free_rate * time_to_expiry) * norm_cdf(-d2))
    
    # Calculate vega (same for calls and puts)
    vega = underlying_price * math.sqrt(time_to_expiry) * norm_pdf(d1)
    
    # Calculate rho
    if option_type == "call":
        rho = strike_price * time_to_expiry * math.exp(-risk_free_rate * time_to_expiry) * norm_cdf(d2)
    else:  # put
        rho = -strike_price * time_to_expiry * math.exp(-risk_free_rate * time_to_expiry) * norm_cdf(-d2)
    
    return {
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega": vega,
        "rho": rho
    }

def norm_cdf(x: float) -> float:
    """Calculate the cumulative distribution function of the standard normal distribution."""
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def norm_pdf(x: float) -> float:
    """Calculate the probability density function of the standard normal distribution."""
    return math.exp(-x**2 / 2.0) / math.sqrt(2.0 * math.pi) 