import math
from typing import Literal
from agents import function_tool

@function_tool
def calculate_option_price(
    option_type: Literal["call", "put"],
    stock_price: float,
    strike_price: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
) -> float:
    """
    Calculate the price of an option using the Black-Scholes model.
    
    Args:
        option_type: Either "call" or "put"
        stock_price: Current price of the underlying stock
        strike_price: Strike price of the option
        time_to_expiry: Time to expiration in years
        risk_free_rate: Risk-free interest rate (as a decimal)
        volatility: Volatility of the underlying stock (as a decimal)
    
    Returns:
        float: The calculated option price
    """
    d1 = (math.log(stock_price / strike_price) + (risk_free_rate + volatility**2 / 2) * time_to_expiry) / (volatility * math.sqrt(time_to_expiry))
    d2 = d1 - volatility * math.sqrt(time_to_expiry)
    
    if option_type == "call":
        price = stock_price * norm_cdf(d1) - strike_price * math.exp(-risk_free_rate * time_to_expiry) * norm_cdf(d2)
    else:  # put
        price = strike_price * math.exp(-risk_free_rate * time_to_expiry) * norm_cdf(-d2) - stock_price * norm_cdf(-d1)
    
    return price

@function_tool
def calculate_option_greeks(
    option_type: Literal["call", "put"],
    stock_price: float,
    strike_price: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
) -> dict:
    """
    Calculate the Greeks (delta, gamma, theta, vega, rho) for an option.
    
    Args:
        option_type: Either "call" or "put"
        stock_price: Current price of the underlying stock
        strike_price: Strike price of the option
        time_to_expiry: Time to expiration in years
        risk_free_rate: Risk-free interest rate (as a decimal)
        volatility: Volatility of the underlying stock (as a decimal)
    
    Returns:
        dict: Dictionary containing the calculated Greeks
    """
    d1 = (math.log(stock_price / strike_price) + (risk_free_rate + volatility**2 / 2) * time_to_expiry) / (volatility * math.sqrt(time_to_expiry))
    d2 = d1 - volatility * math.sqrt(time_to_expiry)
    
    # Calculate delta
    if option_type == "call":
        delta = norm_cdf(d1)
    else:  # put
        delta = norm_cdf(d1) - 1
    
    # Calculate gamma (same for calls and puts)
    gamma = norm_pdf(d1) / (stock_price * volatility * math.sqrt(time_to_expiry))
    
    # Calculate theta
    if option_type == "call":
        theta = (-stock_price * norm_pdf(d1) * volatility / (2 * math.sqrt(time_to_expiry)) 
                - risk_free_rate * strike_price * math.exp(-risk_free_rate * time_to_expiry) * norm_cdf(d2))
    else:  # put
        theta = (-stock_price * norm_pdf(d1) * volatility / (2 * math.sqrt(time_to_expiry))
                + risk_free_rate * strike_price * math.exp(-risk_free_rate * time_to_expiry) * norm_cdf(-d2))
    
    # Calculate vega (same for calls and puts)
    vega = stock_price * math.sqrt(time_to_expiry) * norm_pdf(d1)
    
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