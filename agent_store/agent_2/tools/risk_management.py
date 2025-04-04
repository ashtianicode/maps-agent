import numpy as np
from scipy import stats
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from agents import function_tool

# Import tools from other modules
from agent_store.agent_2.tools.options_tools import calculate_option_price, calculate_option_greeks
from agent_store.agent_2.tools.market_data import get_market_data

def calculate_var(
    position_type: Optional[str] = None,
    underlying_symbol: Optional[str] = None,
    position_details: Optional[Dict[str, Union[str, float, int]]] = None,
    confidence_level: Optional[float] = None,
    time_horizon: Optional[int] = None,
    method: Optional[str] = None,
    simulations: Optional[int] = None
) -> Dict[str, float]:
    """
    Calculate Value at Risk (VaR) for an options position.
    
    Args:
        position_type: Type of position ('option' or 'portfolio')
        underlying_symbol: Stock ticker symbol
        position_details: Dictionary containing position details
        confidence_level: Confidence level for VaR calculation (optional, default: 0.95)
        time_horizon: Time horizon in days (optional, default: 1)
        method: VaR calculation method ('historical', 'parametric', or 'monte_carlo') (optional)
        simulations: Number of simulations for Monte Carlo method (optional)
        
    Returns:
        Dict containing VaR metrics
    """
    # Set default values and validate required parameters
    if position_type is None or underlying_symbol is None or position_details is None:
        raise ValueError("Missing required parameters for VaR calculation")
    
    position_type = position_type.lower()
    if position_type not in ["option", "strategy", "portfolio"]:
        raise ValueError("position_type must be 'option', 'strategy', or 'portfolio'")
    
    # Set default values
    confidence_level = 0.95 if confidence_level is None else confidence_level
    time_horizon = 1 if time_horizon is None else time_horizon
    method = 'parametric' if method is None else method.lower()
    simulations = 10000 if simulations is None else simulations
    
    # Get market data for the underlying asset
    market_data = get_market_data(symbol=underlying_symbol)
    current_price = market_data.get("price", 100.0)
    volatility = market_data.get("volatility", 0.25)
    risk_free_rate = market_data.get("risk_free_rate", 0.03)
    
    # Calculate position value and define parameters based on position type
    position_value = 0
    options_list = []
    
    if position_type == "option":
        # Single option position
        option_details = position_details
        option_type = option_details.get("option_type", "call")
        strike_price = option_details.get("strike_price", current_price)
        time_to_expiry = option_details.get("time_to_expiry", 0.25)
        quantity = option_details.get("quantity", 1)
        is_long = option_details.get("position", "long").lower() == "long"
        
        # Calculate option price
        option_price = calculate_option_price(
            underlying_price=current_price,
            strike_price=strike_price,
            time_to_expiry=time_to_expiry,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            option_type=option_type
        )
        
        # Calculate greeks
        greeks = calculate_option_greeks(
            underlying_price=current_price,
            strike_price=strike_price,
            time_to_expiry=time_to_expiry,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            option_type=option_type
        )
        
        position_value = option_price * quantity * 100  # *100 for contract multiplier
        if not is_long:
            position_value = -position_value  # For short positions
        
        options_list = [option_details]
        
    elif position_type == "strategy":
        # Strategy with multiple options
        options_list = position_details.get("options", [])
        
        for option in options_list:
            option_type = option.get("option_type", "call")
            strike_price = option.get("strike_price", current_price)
            time_to_expiry = option.get("time_to_expiry", 0.25)
            quantity = option.get("quantity", 1)
            is_long = option.get("position", "long").lower() == "long"
            
            option_price = calculate_option_price(
                underlying_price=current_price,
                strike_price=strike_price,
                time_to_expiry=time_to_expiry,
                risk_free_rate=risk_free_rate,
                volatility=volatility,
                option_type=option_type
            )
            
            option_value = option_price * quantity * 100
            if not is_long:
                option_value = -option_value
                
            position_value += option_value
    
    elif position_type == "portfolio":
        # Portfolio of positions
        positions = position_details.get("positions", [])
        components_var = {}
        
        for position in positions:
            pos_type = position.get("type", "option")
            pos_symbol = position.get("symbol", underlying_symbol)
            pos_weight = position.get("weight", 1.0)
            pos_details = position.get("details", {})
            
            # Recursively calculate VaR for each position
            component_var = calculate_var(
                position_type=pos_type,
                underlying_symbol=pos_symbol,
                position_details=pos_details,
                confidence_level=confidence_level,
                time_horizon=time_horizon,
                method=method
            )
            
            # Weight the VaR and add to total
            weighted_var = component_var["var_absolute"] * pos_weight
            components_var[f"{pos_symbol}_{pos_type}"] = weighted_var
            position_value += pos_weight * component_var["position_value"]
        
        # For portfolio, use simple summation for now (ignoring correlations)
        var_absolute = sum(components_var.values())
        var_percentage = var_absolute / position_value if position_value != 0 else 0
        
        return {
            "var_absolute": var_absolute,
            "var_percentage": var_percentage,
            "var_by_component": components_var,
            "position_value": position_value,
            "var_parameters": {
                "confidence_level": confidence_level,
                "time_horizon": time_horizon,
                "method": method
            }
        }
    
    # Calculate VaR based on selected method
    var_absolute = 0
    
    if method == "parametric":
        # Parametric VaR using delta-normal method
        # For options, use delta to approximate price changes
        
        total_delta = 0
        
        for option in options_list:
            option_type = option.get("option_type", "call")
            strike_price = option.get("strike_price", current_price)
            time_to_expiry = option.get("time_to_expiry", 0.25)
            quantity = option.get("quantity", 1)
            is_long = option.get("position", "long").lower() == "long"
            
            # Get delta
            greeks = calculate_option_greeks(
                underlying_price=current_price,
                strike_price=strike_price,
                time_to_expiry=time_to_expiry,
                risk_free_rate=risk_free_rate,
                volatility=volatility,
                option_type=option_type
            )
            
            option_delta = greeks["delta"]
            if not is_long:
                option_delta = -option_delta
                
            # Adjust for quantity and contract multiplier
            total_delta += option_delta * quantity * 100
            
        # Calculate VaR using the delta-normal method
        z_score = stats.norm.ppf(confidence_level)
        daily_vol = volatility / np.sqrt(252)  # Convert annual vol to daily
        var_1day = current_price * total_delta * daily_vol * z_score
        
        # Adjust for time horizon
        var_absolute = var_1day * np.sqrt(time_horizon)
        
    elif method == "historical":
        # Simplified historical simulation
        # In a real implementation, this would use historical price movements
        
        # For now, simulate based on normal distribution
        returns = np.random.normal(0, volatility / np.sqrt(252), 1000)
        
        # Calculate option values for each simulated price
        values = []
        
        for ret in returns:
            simulated_price = current_price * (1 + ret)
            sim_position_value = 0
            
            for option in options_list:
                option_type = option.get("option_type", "call")
                strike_price = option.get("strike_price", current_price)
                time_to_expiry = option.get("time_to_expiry", 0.25)
                quantity = option.get("quantity", 1)
                is_long = option.get("position", "long").lower() == "long"
                
                sim_option_price = calculate_option_price(
                    underlying_price=simulated_price,
                    strike_price=strike_price,
                    time_to_expiry=time_to_expiry,
                    risk_free_rate=risk_free_rate,
                    volatility=volatility,
                    option_type=option_type
                )
                
                sim_option_value = sim_option_price * quantity * 100
                if not is_long:
                    sim_option_value = -sim_option_value
                    
                sim_position_value += sim_option_value
                
            values.append(sim_position_value)
            
        # Calculate VaR from the simulated values
        initial_value = position_value
        losses = [initial_value - value for value in values]
        var_absolute = np.percentile(losses, confidence_level * 100)
        
        # Adjust for time horizon
        var_absolute *= np.sqrt(time_horizon)
        
    elif method == "monte_carlo":
        # Monte Carlo simulation
        dt = time_horizon / 252  # Convert to annual fraction
        
        # Simulate price paths
        price_paths = []
        for _ in range(simulations):
            z = np.random.normal()
            sim_price = current_price * np.exp((risk_free_rate - 0.5 * volatility**2) * dt + 
                                              volatility * np.sqrt(dt) * z)
            price_paths.append(sim_price)
            
        # Calculate position values for each simulated price
        values = []
        
        for sim_price in price_paths:
            sim_position_value = 0
            
            for option in options_list:
                option_type = option.get("option_type", "call")
                strike_price = option.get("strike_price", current_price)
                time_to_expiry = option.get("time_to_expiry", 0.25)
                quantity = option.get("quantity", 1)
                is_long = option.get("position", "long").lower() == "long"
                
                # Adjust time to expiry for the simulation horizon
                adjusted_time = time_to_expiry - dt if time_to_expiry > dt else 0.001
                
                sim_option_price = calculate_option_price(
                    underlying_price=sim_price,
                    strike_price=strike_price,
                    time_to_expiry=adjusted_time,
                    risk_free_rate=risk_free_rate,
                    volatility=volatility,
                    option_type=option_type
                )
                
                sim_option_value = sim_option_price * quantity * 100
                if not is_long:
                    sim_option_value = -sim_option_value
                    
                sim_position_value += sim_option_value
                
            values.append(sim_position_value)
            
        # Calculate VaR from the simulated values
        initial_value = position_value
        losses = [initial_value - value for value in values]
        var_absolute = np.percentile(losses, confidence_level * 100)
    
    # Calculate percentage VaR
    var_percentage = var_absolute / position_value if position_value != 0 else 0
    
    # Return the results
    return {
        "var_absolute": float(var_absolute),
        "var_percentage": float(var_percentage),
        "position_value": float(position_value),
        "var_parameters": {
            "confidence_level": confidence_level,
            "time_horizon": time_horizon,
            "method": method,
            "volatility_used": volatility
        }
    }

def calculate_stress_test(
    position_type: Optional[str] = None,
    underlying_symbol: Optional[str] = None,
    position_details: Optional[Dict[str, Union[str, float, int]]] = None,
    stress_scenarios: Optional[List[Dict[str, float]]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Perform stress testing on an options position.
    
    Args:
        position_type: Type of position ('option' or 'portfolio')
        underlying_symbol: Stock ticker symbol
        position_details: Dictionary containing position details
        stress_scenarios: List of scenarios to test (optional)
        
    Returns:
        Dict containing stress test results for each scenario
    """
    # Set default values and validate required parameters
    if position_type is None or underlying_symbol is None or position_details is None:
        raise ValueError("Missing required parameters for stress testing")
    
    position_type = position_type.lower()
    if position_type not in ["option", "strategy", "portfolio"]:
        raise ValueError("position_type must be 'option', 'strategy', or 'portfolio'")
    
    # Set default scenarios if none provided
    if stress_scenarios is None:
        stress_scenarios = [
            {
                "scenario_name": "Market Crash",
                "price_change": -0.15,  # -15%
                "volatility_change": 0.10,  # +10 percentage points
                "time_decay": 0
            },
            {
                "scenario_name": "Market Rally",
                "price_change": 0.10,  # +10%
                "volatility_change": -0.05,  # -5 percentage points
                "time_decay": 0
            },
            {
                "scenario_name": "Flat Market + Time Decay",
                "price_change": 0.0,  # No change
                "volatility_change": 0.0,  # No change
                "time_decay": 7  # 7 days
            }
        ]
    
    # Get market data for the underlying asset
    market_data = get_market_data(symbol=underlying_symbol)
    current_price = market_data.get("price", 100.0)
    volatility = market_data.get("volatility", 0.25)
    risk_free_rate = market_data.get("risk_free_rate", 0.03)
    
    # Calculate initial position value
    initial_value = 0
    options_list = []
    
    if position_type == "option":
        # Single option position
        option_details = position_details
        options_list = [option_details]
        
    elif position_type == "strategy":
        # Strategy with multiple options
        options_list = position_details.get("options", [])
        
    elif position_type == "portfolio":
        # Portfolio - calculate for each position separately
        positions = position_details.get("positions", [])
        stress_results = {}
        
        for position in positions:
            pos_type = position.get("type", "option")
            pos_symbol = position.get("symbol", underlying_symbol)
            pos_weight = position.get("weight", 1.0)
            pos_details = position.get("details", {})
            
            # Calculate stress test for this component
            component_stress = calculate_stress_test(
                position_type=pos_type,
                underlying_symbol=pos_symbol,
                position_details=pos_details,
                stress_scenarios=stress_scenarios
            )
            
            # Weight the results and add to total
            for scenario, results in component_stress.items():
                if scenario not in stress_results:
                    stress_results[scenario] = {
                        "pnl_amount": 0, 
                        "pnl_percentage": 0,
                        "new_position_value": 0
                    }
                
                stress_results[scenario]["pnl_amount"] += results["pnl_amount"] * pos_weight
                # We'll recalculate the percentage at the end
                stress_results[scenario]["new_position_value"] += results["new_position_value"] * pos_weight
        
        return stress_results
    
    # Calculate initial position value
    for option in options_list:
        option_type = option.get("option_type", "call")
        strike_price = option.get("strike_price", current_price)
        time_to_expiry = option.get("time_to_expiry", 0.25)
        quantity = option.get("quantity", 1)
        is_long = option.get("position", "long").lower() == "long"
        
        option_price = calculate_option_price(
            underlying_price=current_price,
            strike_price=strike_price,
            time_to_expiry=time_to_expiry,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            option_type=option_type
        )
        
        option_value = option_price * quantity * 100
        if not is_long:
            option_value = -option_value
            
        initial_value += option_value
    
    # Run stress tests
    stress_results = {}
    
    for scenario in stress_scenarios:
        scenario_name = scenario.get("scenario_name", "Unnamed Scenario")
        price_change = scenario.get("price_change", 0.0)
        volatility_change = scenario.get("volatility_change", 0.0)
        time_decay_days = scenario.get("time_decay", 0)
        
        # Calculate new market conditions
        new_price = current_price * (1 + price_change)
        new_volatility = volatility + volatility_change
        time_decay_years = time_decay_days / 365.0
        
        # Calculate new position value
        new_value = 0
        
        for option in options_list:
            option_type = option.get("option_type", "call")
            strike_price = option.get("strike_price", current_price)
            time_to_expiry = option.get("time_to_expiry", 0.25)
            quantity = option.get("quantity", 1)
            is_long = option.get("position", "long").lower() == "long"
            
            # Adjust time to expiry for time decay
            new_time_to_expiry = max(0.001, time_to_expiry - time_decay_years)
            
            new_option_price = calculate_option_price(
                underlying_price=new_price,
                strike_price=strike_price,
                time_to_expiry=new_time_to_expiry,
                risk_free_rate=risk_free_rate,
                volatility=new_volatility,
                option_type=option_type
            )
            
            new_option_value = new_option_price * quantity * 100
            if not is_long:
                new_option_value = -new_option_value
                
            new_value += new_option_value
        
        # Calculate profit/loss
        pnl_amount = new_value - initial_value
        pnl_percentage = pnl_amount / initial_value if initial_value != 0 else 0
        
        # Store results
        stress_results[scenario_name] = {
            "pnl_amount": float(pnl_amount),
            "pnl_percentage": float(pnl_percentage),
            "new_position_value": float(new_value)
        }
    
    return stress_results 