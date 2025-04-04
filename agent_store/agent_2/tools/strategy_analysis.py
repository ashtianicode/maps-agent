import numpy as np
from typing import Dict, List, Tuple, Union, Optional
from dataclasses import dataclass
from enum import Enum

# Import the existing tools
from agent_store.agent_2.tools.options_tools import calculate_option_price, calculate_option_greeks
from agent_store.agent_2.tools.market_data import get_market_data

class OptionPosition(Enum):
    LONG = 1
    SHORT = -1

@dataclass
class OptionContract:
    """Data class representing an option contract in a strategy"""
    option_type: str  # 'call' or 'put'
    strike_price: float
    time_to_expiry: float
    position: OptionPosition
    quantity: int = 1
    premium: Optional[float] = None  # Market price of the option (if known)
    volatility: Optional[float] = None  # Implied volatility (if known)

@dataclass
class StrategyAnalysisResult:
    """Result of strategy analysis"""
    strategy_name: str
    underlying_price: float
    max_profit: float
    max_loss: float
    breakeven_points: List[float]
    risk_reward_ratio: float
    profit_probability: Optional[float]
    payoff_range: Dict[str, List[float]]  # Dict with keys 'prices', 'payoffs'
    greeks: Dict[str, float]  # Net strategy greeks

def analyze_option_strategy(
    strategy_name: Optional[str] = None,
    underlying_symbol: Optional[str] = None,
    options: Optional[List[Dict[str, Union[str, float, int]]]] = None,
    risk_free_rate: Optional[float] = None,
    current_underlying_price: Optional[float] = None,
    price_range_percentage: Optional[float] = None,
) -> StrategyAnalysisResult:
    """
    Analyze an options strategy by calculating key metrics.
    
    Args:
        strategy_name: Name of the strategy (e.g., 'Bull Call Spread', 'Iron Condor')
        underlying_symbol: Ticker symbol of the underlying asset
        options: List of option contracts in the strategy. Each option is a dict with:
            - option_type: 'call' or 'put'
            - strike_price: Strike price of the option
            - time_to_expiry: Time to expiration in years
            - position: 'long' or 'short'
            - quantity: Number of contracts (optional, default: 1)
            - premium: Option price (optional, will be calculated if not provided)
        risk_free_rate: Risk-free interest rate (optional)
        current_underlying_price: Current price of the underlying (optional)
        price_range_percentage: Range for payoff analysis as percentage of current price (optional, default: 20.0)
        
    Returns:
        A StrategyAnalysisResult object containing the analysis
        
    Example:
        Analyze a bull call spread on AAPL:
        >>> analyze_option_strategy(
        ...     strategy_name="Bull Call Spread",
        ...     underlying_symbol="AAPL",
        ...     options=[
        ...         {"option_type": "call", "strike_price": 150, "time_to_expiry": 0.25, "position": "long"},
        ...         {"option_type": "call", "strike_price": 160, "time_to_expiry": 0.25, "position": "short"}
        ...     ]
        ... )
    """
    # Set default values
    strategy_name = strategy_name or "Custom Strategy"
    underlying_symbol = underlying_symbol or "UNKNOWN"
    options = options or []
    price_range_percentage = 20.0 if price_range_percentage is None else price_range_percentage
    
    # Fetch market data if not provided
    if current_underlying_price is None or risk_free_rate is None:
        market_data = get_market_data(symbol=underlying_symbol)
        current_underlying_price = market_data.get("price", 100.0)  # Default if unable to fetch
        risk_free_rate = market_data.get("risk_free_rate", 0.03)  # Default 3% if unable to fetch
        default_volatility = market_data.get("volatility", 0.25)  # Default 25% if unable to fetch
    else:
        default_volatility = 0.25  # Default volatility if not provided
    
    # Convert options to OptionContract objects and calculate missing premiums
    option_contracts = []
    for opt in options:
        position = OptionPosition.LONG if opt.get("position", "long").lower() == "long" else OptionPosition.SHORT
        quantity = opt.get("quantity", 1)
        volatility = opt.get("volatility", default_volatility)
        premium = opt.get("premium")
        
        # Calculate premium if not provided
        if premium is None:
            premium = calculate_option_price(
                underlying_price=current_underlying_price,
                strike_price=opt["strike_price"],
                time_to_expiry=opt["time_to_expiry"],
                risk_free_rate=risk_free_rate,
                volatility=volatility,
                option_type=opt["option_type"]
            )
        
        contract = OptionContract(
            option_type=opt["option_type"],
            strike_price=opt["strike_price"],
            time_to_expiry=opt["time_to_expiry"],
            position=position,
            quantity=quantity,
            premium=premium,
            volatility=volatility
        )
        option_contracts.append(contract)
    
    # Calculate payoff at expiration across a range of prices
    price_range_low = current_underlying_price * (1 - price_range_percentage / 100)
    price_range_high = current_underlying_price * (1 + price_range_percentage / 100)
    price_points = 100
    prices = np.linspace(price_range_low, price_range_high, price_points)
    
    # Calculate payoff for each price point
    payoffs = np.zeros(price_points)
    initial_cost = 0
    
    for contract in option_contracts:
        # Add initial premium paid/received to calculate strategy cost
        initial_cost += contract.position.value * contract.premium * contract.quantity * 100  # x100 for contract multiplier
        
        for i, price in enumerate(prices):
            payoff = 0
            
            if contract.option_type == "call":
                intrinsic_value = max(0, price - contract.strike_price)
                payoff = contract.position.value * intrinsic_value * contract.quantity * 100  # x100 for contract multiplier
            elif contract.option_type == "put":
                intrinsic_value = max(0, contract.strike_price - price)
                payoff = contract.position.value * intrinsic_value * contract.quantity * 100  # x100 for contract multiplier
            
            payoffs[i] += payoff
    
    # Adjust payoffs by initial cost to get net payoff
    payoffs -= initial_cost
    
    # Calculate key metrics
    max_profit = float(np.max(payoffs))
    max_loss = float(np.min(payoffs))
    
    # Find breakeven points where payoff crosses zero
    breakeven_points = []
    for i in range(1, len(payoffs)):
        if (payoffs[i-1] <= 0 and payoffs[i] > 0) or (payoffs[i-1] >= 0 and payoffs[i] < 0):
            # Linear interpolation to find more precise breakeven
            x1, y1 = prices[i-1], payoffs[i-1]
            x2, y2 = prices[i], payoffs[i]
            if y1 != y2:  # Avoid division by zero
                breakeven = x1 + (x2 - x1) * (-y1) / (y2 - y1)
                breakeven_points.append(float(breakeven))
    
    # Calculate risk-reward ratio if both max profit and max loss are non-zero
    risk_reward_ratio = 0
    if max_loss != 0 and max_profit != 0:
        risk_reward_ratio = abs(max_profit / max_loss)
    
    # Calculate aggregate Greeks for the strategy
    strategy_greeks = {
        "delta": 0.0,
        "gamma": 0.0,
        "theta": 0.0,
        "vega": 0.0,
        "rho": 0.0
    }
    
    for contract in option_contracts:
        greeks = calculate_option_greeks(
            underlying_price=current_underlying_price,
            strike_price=contract.strike_price,
            time_to_expiry=contract.time_to_expiry,
            risk_free_rate=risk_free_rate,
            volatility=contract.volatility,
            option_type=contract.option_type
        )
        
        position_multiplier = contract.position.value * contract.quantity
        for greek, value in greeks.items():
            strategy_greeks[greek] += value * position_multiplier
    
    # Calculate probability of profit (simplified approximation)
    # Using the percentage of price points where payoff > 0
    profit_probability = float(np.sum(payoffs > 0) / len(payoffs))
    
    # Create the result object
    result = StrategyAnalysisResult(
        strategy_name=strategy_name,
        underlying_price=current_underlying_price,
        max_profit=max_profit,
        max_loss=max_loss,
        breakeven_points=breakeven_points,
        risk_reward_ratio=risk_reward_ratio,
        profit_probability=profit_probability,
        payoff_range={
            "prices": prices.tolist(),
            "payoffs": payoffs.tolist()
        },
        greeks=strategy_greeks
    )
    
    return result


def calculate_payoff_diagram_data(
    strategy_name: Optional[str] = None,
    underlying_symbol: Optional[str] = None,
    options: Optional[List[Dict[str, Union[str, float, int]]]] = None,
    current_price: Optional[float] = None,
    price_points: Optional[int] = None,
) -> Dict[str, Union[List[float], Dict[str, List[float]]]]:
    """
    Calculate data for plotting a payoff diagram of an options strategy.
    
    Args:
        strategy_name: Name of the strategy (e.g., 'Bull Call Spread')
        underlying_symbol: Ticker symbol of the underlying asset
        options: List of option contracts in the strategy (see analyze_option_strategy)
        current_price: Current price of the underlying (optional)
        price_points: Number of price points to calculate (optional)
        
    Returns:
        Dictionary containing the data needed to plot a payoff diagram, including
        underlying prices, payoffs for each component, and the total strategy payoff.
        
    Example:
        Calculate payoff diagram data for a straddle on SPY:
        >>> calculate_payoff_diagram_data(
        ...     strategy_name="Long Straddle",
        ...     underlying_symbol="SPY",
        ...     options=[
        ...         {"option_type": "call", "strike_price": 400, "time_to_expiry": 0.5, "position": "long"},
        ...         {"option_type": "put", "strike_price": 400, "time_to_expiry": 0.5, "position": "long"}
        ...     ]
        ... )
    """
    # Set default values
    strategy_name = strategy_name or "Custom Strategy"
    underlying_symbol = underlying_symbol or "UNKNOWN"
    options = options or []
    price_points = price_points or 100
    
    # Get market data if current price is not provided
    if current_price is None:
        market_data = get_market_data(symbol=underlying_symbol)
        current_price = market_data.get("price", 100.0)  # Default if unable to fetch
    
    # Define the price range to cover reasonable scenarios
    # Consider both the current price and the strikes in the strategy
    strikes = [opt["strike_price"] for opt in options]
    min_strike = min(strikes)
    max_strike = max(strikes)
    
    # Ensure range is wide enough to show important points
    price_range_min = min(min_strike * 0.7, current_price * 0.7)
    price_range_max = max(max_strike * 1.3, current_price * 1.3)
    
    # Generate price points
    prices = np.linspace(price_range_min, price_range_max, price_points)
    
    # Initialize the data structure for individual option payoffs
    component_payoffs = {}
    total_payoff = np.zeros(price_points)
    
    # Calculate the payoff for each option
    for i, opt in enumerate(options):
        option_id = f"{opt['option_type']}_{opt['strike_price']}_{opt['position']}"
        position_multiplier = 1 if opt.get("position", "long").lower() == "long" else -1
        quantity = opt.get("quantity", 1)
        premium = opt.get("premium", 5.0)  # Default premium if not provided
        
        # Calculate payoff for this option at each price point
        payoffs = np.zeros(price_points)
        
        for j, price in enumerate(prices):
            intrinsic_value = 0
            if opt["option_type"] == "call":
                intrinsic_value = max(0, price - opt["strike_price"])
            else:  # put
                intrinsic_value = max(0, opt["strike_price"] - price)
            
            # Payoff = intrinsic value - premium (long) or premium - intrinsic value (short)
            if position_multiplier > 0:  # long
                payoffs[j] = (intrinsic_value - premium) * quantity * 100  # *100 for contract multiplier
            else:  # short
                payoffs[j] = (premium - intrinsic_value) * quantity * 100  # *100 for contract multiplier
        
        # Store the component payoff and add to total
        component_payoffs[option_id] = payoffs.tolist()
        total_payoff += payoffs
    
    # Compile the final result
    result = {
        "prices": prices.tolist(),
        "total_payoff": total_payoff.tolist(),
        "component_payoffs": component_payoffs,
        "breakeven_points": [],
        "strategy_name": strategy_name,
        "current_price": current_price
    }
    
    # Find breakeven points
    for i in range(1, len(total_payoff)):
        if (total_payoff[i-1] <= 0 and total_payoff[i] > 0) or (total_payoff[i-1] >= 0 and total_payoff[i] < 0):
            # Linear interpolation to find more precise breakeven
            x1, y1 = prices[i-1], total_payoff[i-1]
            x2, y2 = prices[i], total_payoff[i]
            if y1 != y2:  # Avoid division by zero
                breakeven = x1 + (x2 - x1) * (-y1) / (y2 - y1)
                result["breakeven_points"].append(float(breakeven))
    
    return result 