import json
from typing import Dict, List, Optional, Union, Tuple, Any
import base64
from io import BytesIO
from agents import function_tool

# Try to import plotting libraries - these may require installation in the environment
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

# Import tools from the agent
from agent_store.agent_2.tools.strategy_analysis import calculate_payoff_diagram_data

def plot_payoff_diagram(
    strategy_name: Optional[str] = None,
    underlying_symbol: Optional[str] = None,
    options: Optional[List[Dict[str, Union[str, float, int]]]] = None,
    current_price: Optional[float] = None,
    price_points: Optional[int] = None,
    return_format: Optional[str] = None,
    plot_components: Optional[bool] = None,
    highlight_breakeven: Optional[bool] = None,
    show_current_price: Optional[bool] = None,
    plot_title: Optional[str] = None,
    figsize: Optional[List[int]] = None
) -> Dict[str, Union[str, Dict]]:
    """
    Generate a payoff diagram for an options strategy.
    
    Args:
        strategy_name: Name of the strategy
        underlying_symbol: Stock ticker symbol
        options: List of option contracts in the strategy
        current_price: Current price of underlying (optional)
        price_points: Number of points to plot (optional)
        return_format: Format of the plot ('base64', 'json', or None for both)
        plot_components: Whether to plot individual components (optional)
        highlight_breakeven: Whether to highlight breakeven points (optional)
        show_current_price: Whether to show current price line (optional)
        plot_title: Custom plot title (optional)
        figsize: Figure size as [width, height] list (optional)
        
    Returns:
        Dict containing the plot data and/or image
    """
    # Set default values and validate required parameters
    if strategy_name is None or underlying_symbol is None or options is None:
        raise ValueError("Missing required parameters for payoff diagram")
    
    if not PLOTTING_AVAILABLE:
        return {
            "error": "Matplotlib is not available. Install it with 'pip install matplotlib numpy'.",
            "data": None,
            "image": None
        }
    
    try:
        # Set default values
        price_points = 100 if price_points is None else price_points
        return_format = 'base64' if return_format is None else return_format
        plot_components = True if plot_components is None else plot_components
        highlight_breakeven = True if highlight_breakeven is None else highlight_breakeven
        show_current_price = True if show_current_price is None else show_current_price
        figsize = [10, 6] if figsize is None else tuple(figsize)  # Convert list to tuple for matplotlib
        # Calculate the data for the payoff diagram
        payoff_data = calculate_payoff_diagram_data(
            strategy_name=strategy_name,
            underlying_symbol=underlying_symbol,
            options=options,
            current_price=current_price,
            price_points=price_points
        )
        
        # Return only the data if requested
        if return_format.lower() == 'json':
            return {"data": payoff_data, "image": None, "error": None}
        
        # Otherwise create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Extract data
        prices = payoff_data["prices"]
        total_payoff = payoff_data["total_payoff"]
        component_payoffs = payoff_data["component_payoffs"]
        breakeven_points = payoff_data["breakeven_points"]
        current_stock_price = payoff_data["current_price"]
        
        # Plot individual components if requested
        if plot_components and component_payoffs:
            for name, payoffs in component_payoffs.items():
                parts = name.split('_')
                option_type = parts[0]
                strike = parts[1]
                position = parts[2]
                label = f"{position.title()} {option_type} (K=${strike})"
                
                # Use different line styles based on position and option type
                linestyle = '--' if position == 'short' else '-'
                color = 'blue' if option_type == 'call' else 'red'
                
                ax.plot(prices, payoffs, linestyle=linestyle, color=color, alpha=0.5, linewidth=1, label=label)
        
        # Plot the total strategy payoff (thicker line)
        ax.plot(prices, total_payoff, 'k-', linewidth=2.5, label=f"{strategy_name} Payoff")
        
        # Add zero line
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        
        # Highlight breakeven points if requested
        if highlight_breakeven and breakeven_points:
            for be_point in breakeven_points:
                ax.axvline(x=be_point, color='green', linestyle='--', alpha=0.7)
                ax.annotate(f"BE: ${be_point:.2f}", 
                           xy=(be_point, 0), 
                           xytext=(be_point, max(total_payoff) * 0.2),
                           arrowprops=dict(arrowstyle='->', color='green'),
                           color='green',
                           fontsize=9)
        
        # Show current price if requested
        if show_current_price and current_stock_price:
            ax.axvline(x=current_stock_price, color='blue', linestyle=':', alpha=0.7)
            ax.annotate(f"Current: ${current_stock_price:.2f}",
                       xy=(current_stock_price, 0),
                       xytext=(current_stock_price, max(total_payoff) * 0.4),
                       arrowprops=dict(arrowstyle='->', color='blue'),
                       color='blue',
                       fontsize=9)
        
        # Add title and labels
        if plot_title:
            title = plot_title
        else:
            title = f"{strategy_name} Payoff Diagram - {underlying_symbol}"
        
        ax.set_title(title)
        ax.set_xlabel("Underlying Price ($)")
        ax.set_ylabel("Profit/Loss ($)")
        
        # Add max profit/loss annotations
        max_profit = max(total_payoff)
        max_loss = min(total_payoff)
        
        if max_profit > 0:
            ax.annotate(f"Max Profit: ${max_profit:.2f}",
                       xy=(prices[np.argmax(total_payoff)], max_profit),
                       xytext=(prices[np.argmax(total_payoff)], max_profit * 0.9),
                       fontsize=9,
                       color='darkgreen')
        
        if max_loss < 0:
            ax.annotate(f"Max Loss: ${max_loss:.2f}",
                       xy=(prices[np.argmin(total_payoff)], max_loss),
                       xytext=(prices[np.argmin(total_payoff)], max_loss * 0.9),
                       fontsize=9,
                       color='darkred')
        
        # Add grid and legend
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        # Save the plot to a BytesIO object and convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close(fig)
        
        return {
            "image": image_base64,
            "data": payoff_data,
            "error": None
        }
        
    except Exception as e:
        return {
            "error": f"Error creating payoff diagram: {str(e)}",
            "data": None,
            "image": None
        }

def plot_greeks_sensitivities(
    option_type: Optional[str] = None,
    strike_price: Optional[float] = None,
    time_to_expiry: Optional[float] = None,
    underlying_price: Optional[float] = None,
    volatility: Optional[float] = None,
    risk_free_rate: Optional[float] = None,
    greek: Optional[str] = None,
    variable: Optional[str] = None,
    range_percentage: Optional[float] = None,
    points: Optional[int] = None,
    return_format: Optional[str] = None,
    figsize: Optional[List[int]] = None
) -> Dict[str, Union[str, Dict]]:
    """
    Plot sensitivity of a Greek to changes in a variable.
    
    Args:
        option_type: Type of option ('call' or 'put')
        strike_price: Strike price of the option
        time_to_expiry: Time to expiration in years
        underlying_price: Current price of underlying
        volatility: Current implied volatility
        risk_free_rate: Risk-free interest rate
        greek: Greek to plot ('delta', 'gamma', 'theta', 'vega', 'rho')
        variable: Variable to vary ('underlying_price', 'volatility', 'time_to_expiry')
        range_percentage: Range to vary the variable by (optional)
        points: Number of points to plot (optional)
        return_format: Format of the plot ('base64', 'json', or None for both)
        figsize: Figure size as [width, height] list (optional)
        
    Returns:
        Dict containing the plot data and/or image
    """
    # Set default values and validate required parameters
    if option_type is None or strike_price is None or time_to_expiry is None or \
       underlying_price is None or volatility is None or risk_free_rate is None or \
       greek is None or variable is None:
        raise ValueError("Missing required parameters for Greeks sensitivity plot")
    
    option_type = option_type.lower()
    if option_type not in ["call", "put"]:
        raise ValueError("option_type must be either 'call' or 'put'")
    
    greek = greek.lower()
    if greek not in ["delta", "gamma", "theta", "vega", "rho"]:
        raise ValueError("Invalid Greek specified")
    
    variable = variable.lower()
    if variable not in ["underlying_price", "volatility", "time_to_expiry"]:
        raise ValueError("Invalid variable specified")
    
    if not PLOTTING_AVAILABLE:
        return {
            "error": "Matplotlib is not available. Install it with 'pip install matplotlib numpy'.",
            "data": None,
            "image": None
        }
    
    # Import here to avoid circular imports
    from agent_store.agent_2.tools.options_tools import calculate_option_greeks
    
    try:
        # Set default values
        greek = 'delta' if greek is None else greek
        variable = 'underlying_price' if variable is None else variable
        range_percentage = 20.0 if range_percentage is None else range_percentage
        points = 100 if points is None else points
        return_format = 'base64' if return_format is None else return_format
        figsize = [8, 5] if figsize is None else tuple(figsize)  # Convert list to tuple for matplotlib
        
        # Define the range of values for the variable
        if variable == "underlying_price":
            base_value = underlying_price
            min_value = base_value * (1 - range_percentage / 100)
            max_value = base_value * (1 + range_percentage / 100)
            x_label = "Underlying Price ($)"
        elif variable == "volatility":
            base_value = volatility
            min_value = max(0.01, base_value * (1 - range_percentage / 100))
            max_value = base_value * (1 + range_percentage / 100)
            x_label = "Volatility"
        elif variable == "time_to_expiry":
            base_value = time_to_expiry
            min_value = max(0.01, base_value * (1 - range_percentage / 100))
            max_value = base_value * (1 + range_percentage / 100)
            x_label = "Time to Expiry (years)"
        else:
            return {
                "error": f"Unsupported variable: {variable}",
                "data": None,
                "image": None
            }
        
        # Generate variable values
        var_values = np.linspace(min_value, max_value, points)
        greek_values = []
        
        # Calculate Greek values for each variable value
        for val in var_values:
            # Create parameters dictionary with the varying parameter
            params = {
                "underlying_price": underlying_price,
                "strike_price": strike_price,
                "time_to_expiry": time_to_expiry,
                "risk_free_rate": risk_free_rate,
                "volatility": volatility,
                "option_type": option_type
            }
            
            # Update the varying parameter
            params[variable] = val
            
            # Calculate Greeks
            greeks = calculate_option_greeks(**params)
            greek_values.append(greeks[greek.lower()])
        
        # Prepare data for return
        plot_data = {
            "x_values": var_values.tolist(),
            "y_values": greek_values,
            "variable": variable,
            "greek": greek,
            "option_type": option_type,
            "strike_price": strike_price
        }
        
        # Return only the data if requested
        if return_format.lower() == 'json':
            return {"data": plot_data, "image": None, "error": None}
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot the Greek sensitivity
        ax.plot(var_values, greek_values, linewidth=2)
        
        # Mark the current value with a vertical line
        ax.axvline(x=base_value, color='red', linestyle='--', alpha=0.5)
        
        # Highlight the current Greek value
        current_greek_idx = (np.abs(var_values - base_value)).argmin()
        current_greek_value = greek_values[current_greek_idx]
        ax.plot(base_value, current_greek_value, 'ro')
        ax.annotate(f"{greek.capitalize()}: {current_greek_value:.6f}",
                   xy=(base_value, current_greek_value),
                   xytext=(base_value, current_greek_value * 1.1),
                   fontsize=9,
                   ha='center')
        
        # Add title and labels
        ax.set_title(f"{greek.capitalize()} Sensitivity to {variable.replace('_', ' ').title()}")
        ax.set_xlabel(x_label)
        ax.set_ylabel(f"{greek.capitalize()}")
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Save the plot to a BytesIO object and convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close(fig)
        
        return {
            "image": image_base64,
            "data": plot_data,
            "error": None
        }
        
    except Exception as e:
        return {
            "error": f"Error creating Greeks sensitivity plot: {str(e)}",
            "data": None,
            "image": None
        }

def plot_volatility_surface(
    volatility_data: Optional[Dict[str, Dict[float, float]]] = None,
    underlying_symbol: Optional[str] = None,
    current_price: Optional[float] = None,
    return_format: Optional[str] = None,
    plot_type: Optional[str] = None,
    figsize: Optional[List[int]] = None
) -> Dict[str, Union[str, Dict]]:
    """
    Plot a volatility surface from volatility data.
    
    Args:
        volatility_data: Nested dict of volatilities by expiry and strike
        underlying_symbol: Stock ticker symbol
        current_price: Current price of underlying (optional)
        return_format: Format of the plot ('base64', 'json', or None for both)
        plot_type: Type of plot ('3d' or 'heatmap') (optional)
        figsize: Figure size as [width, height] list (optional)
        
    Returns:
        Dict containing the plot data and/or image
    """
    # Set default values and validate required parameters
    if volatility_data is None or underlying_symbol is None:
        raise ValueError("Missing required parameters for volatility surface plot")
    
    if not PLOTTING_AVAILABLE:
        return {
            "error": "Matplotlib is not available. Install it with 'pip install matplotlib numpy'.",
            "data": None,
            "image": None
        }
    
    try:
        # Set default values
        return_format = 'base64' if return_format is None else return_format
        plot_type = '3d' if plot_type is None else plot_type
        figsize = [10, 8] if figsize is None else tuple(figsize)  # Convert list to tuple for matplotlib
        
        # Check if we have the 3D plotting capabilities if needed
        if plot_type == '3d':
            try:
                from mpl_toolkits.mplot3d import Axes3D
                has_3d = True
            except ImportError:
                has_3d = False
                plot_type = 'heatmap'  # Fall back to heatmap
        
        # Process the volatility data into a format suitable for plotting
        expiries = sorted(volatility_data.keys())
        all_strikes = set()
        for expiry in expiries:
            all_strikes.update(volatility_data[expiry].keys())
        strikes = sorted(all_strikes)
        
        # Calculate days to expiry for each expiration date
        import datetime
        today = datetime.date.today()
        days_to_expiry = []
        for expiry in expiries:
            expiry_date = datetime.datetime.strptime(expiry, '%Y-%m-%d').date()
            days = (expiry_date - today).days
            days_to_expiry.append(days)
        
        # Create a 2D grid for the surface
        X, Y = np.meshgrid(strikes, days_to_expiry)
        Z = np.zeros_like(X, dtype=float)
        
        # Fill in the implied volatility values
        for i, days in enumerate(days_to_expiry):
            expiry = expiries[i]
            for j, strike in enumerate(strikes):
                if strike in volatility_data[expiry]:
                    Z[i, j] = volatility_data[expiry][strike]
        
        # Return only the data if requested
        if return_format.lower() == 'json':
            return {
                "data": {
                    "strikes": strikes,
                    "expiries": expiries,
                    "days_to_expiry": days_to_expiry,
                    "volatility_surface": Z.tolist()
                },
                "image": None,
                "error": None
            }
        
        # Create the plot based on the plot type
        if plot_type == '3d' and has_3d:
            # 3D surface plot
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot the surface
            surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, 
                                  linewidth=0, antialiased=True)
            
            # Add colorbar
            cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
            cbar.set_label('Implied Volatility')
            
            # Set labels
            ax.set_xlabel('Strike Price ($)')
            ax.set_ylabel('Days to Expiry')
            ax.set_zlabel('Implied Volatility')
            
            # Mark the current price if provided
            if current_price is not None:
                # Find the closest strike to the current price
                closest_strike_idx = np.abs(np.array(strikes) - current_price).argmin()
                closest_strike = strikes[closest_strike_idx]
                
                # Draw a vertical line at this strike
                x_line = np.ones(len(days_to_expiry)) * closest_strike
                y_line = np.array(days_to_expiry)
                z_line = Z[:, closest_strike_idx]
                ax.plot(x_line, y_line, z_line, 'r-', linewidth=3)
            
            ax.set_title(f'Implied Volatility Surface for {underlying_symbol}')
            
        else:
            # 2D heatmap plot
            fig, ax = plt.subplots(figsize=figsize)
            
            # Create the heatmap
            im = ax.imshow(Z, cmap='viridis', aspect='auto', origin='lower',
                          extent=[min(strikes), max(strikes), min(days_to_expiry), max(days_to_expiry)])
            
            # Add colorbar
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label('Implied Volatility')
            
            # Set labels and add grid
            ax.set_xlabel('Strike Price ($)')
            ax.set_ylabel('Days to Expiry')
            ax.grid(False)
            
            # Mark the current price if provided
            if current_price is not None:
                ax.axvline(x=current_price, color='red', linestyle='--', alpha=0.7)
            
            ax.set_title(f'Implied Volatility Surface for {underlying_symbol}')
        
        # Save the plot to a BytesIO object and convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close(fig)
        
        return {
            "image": image_base64,
            "data": {
                "strikes": strikes,
                "expiries": expiries,
                "days_to_expiry": days_to_expiry,
                "volatility_surface": Z.tolist()
            },
            "error": None
        }
        
    except Exception as e:
        return {
            "error": f"Error creating volatility surface plot: {str(e)}",
            "data": None,
            "image": None
        } 