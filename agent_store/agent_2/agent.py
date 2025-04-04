import asyncio
from agents import Agent, Runner, function_tool

# Import all tools
# Base tools from V1
from agent_store.agent_2.tools.options_tools import calculate_option_price, calculate_option_greeks
from agent_store.agent_2.tools.market_data import get_market_data, get_options_contracts

# New V2 tools
from agent_store.agent_2.tools.volatility_analysis import calculate_implied_volatility, fetch_volatility_surface
from agent_store.agent_2.tools.strategy_analysis import analyze_option_strategy, calculate_payoff_diagram_data
from agent_store.agent_2.tools.risk_management import calculate_var, calculate_stress_test
from agent_store.agent_2.tools.data_visualization import (
    plot_payoff_diagram,
    plot_greeks_sensitivities,
    plot_volatility_surface
)

# Create function tools
calculate_option_price_tool = function_tool(calculate_option_price)
calculate_option_greeks_tool = function_tool(calculate_option_greeks)
get_market_data_tool = function_tool(get_market_data)
get_options_contracts_tool = function_tool(get_options_contracts)
calculate_implied_volatility_tool = function_tool(calculate_implied_volatility)
fetch_volatility_surface_tool = function_tool(fetch_volatility_surface)
analyze_option_strategy_tool = function_tool(analyze_option_strategy)
calculate_payoff_diagram_data_tool = function_tool(calculate_payoff_diagram_data)
calculate_var_tool = function_tool(calculate_var)
calculate_stress_test_tool = function_tool(calculate_stress_test)
plot_payoff_diagram_tool = function_tool(plot_payoff_diagram)
plot_greeks_sensitivities_tool = function_tool(plot_greeks_sensitivities)
plot_volatility_surface_tool = function_tool(plot_volatility_surface)

options_calculator_agent = Agent(
    name="Options Calculator V2",
    handoff_description="Advanced options calculator agent with comprehensive analysis capabilities",
    instructions="""You are an advanced options calculator agent that helps users with sophisticated options analysis and calculations.
    
    Key Features:
    1. Real-time Market Data:
       - Fetch current stock prices, volatility, and risk-free rates
       - Access options contract data from Polygon.io API for accurate strike prices and expiration dates
    
    2. Options Pricing and Greeks:
       - Calculate option prices using the Black-Scholes model
       - Compute all major Greeks (delta, gamma, theta, vega, rho)
       - Calculate implied volatility from market prices
    
    3. Strategy Analysis:
       - Analyze complex options strategies (spreads, straddles, etc.)
       - Calculate maximum profit/loss and breakeven points
       - Generate payoff diagrams for visualization
    
    4. Risk Management:
       - Calculate Value at Risk (VaR)
       - Perform stress testing on options positions
       - Analyze portfolio sensitivity
    
    5. Data Visualization:
       - Plot payoff diagrams for strategies
       - Visualize Greeks sensitivities
       - Generate volatility surface plots
    
    When users ask about options:
    1. First use get_market_data to fetch current market data
    2. Use get_options_contracts to fetch available contracts if needed
    3. Calculate relevant metrics using appropriate tools
    4. Provide clear explanations of results
    5. Offer visualizations when helpful
    
    Always mention:
    - Market data sources and timestamps
    - Assumptions used in calculations
    - Risk factors and limitations
    - Alternative scenarios when relevant
    
    For options contract data, you have access to Polygon.io API which provides:
    - Contract specifications (strike, expiry, type)
    - Contract identifiers and exchange information
    - Underlying asset details
    
    Handle missing data gracefully and provide reasonable defaults when necessary.""",
    tools=[
        calculate_option_price_tool,
        calculate_option_greeks_tool,
        get_market_data_tool,
        get_options_contracts_tool,
        calculate_implied_volatility_tool,
        fetch_volatility_surface_tool,
        analyze_option_strategy_tool,
        calculate_payoff_diagram_data_tool,
        calculate_var_tool,
        calculate_stress_test_tool,
        plot_payoff_diagram_tool,
        plot_greeks_sensitivities_tool,
        plot_volatility_surface_tool
    ]
)

# Example usage (optional, for testing)
async def main():
    # Example query
    placeholder = "What is the price of a call option on Tesla with a strike price of $100 and a maturity of 1 year?"
    query = input("Enter a query: ") or placeholder
    result = await Runner.run(options_calculator_agent, query)
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())