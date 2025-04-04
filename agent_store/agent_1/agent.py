import asyncio
from agents import Agent, Runner
from agent_store.agent_1.tools.options_tools import calculate_option_price, calculate_option_greeks
from agent_store.agent_1.tools.market_data import get_market_data

options_calculator_agent = Agent(
    name="Options Calculator",
    handoff_description="Specialist agent for options calculations",
    instructions="""You are an options calculator agent that helps users with options pricing and Greeks calculations.
    You can:
    1. Calculate option prices using the Black-Scholes model
    2. Calculate option Greeks (delta, gamma, theta, vega, rho)
    3. Get real-time market data for stocks
    4. Explain the meaning and significance of different option metrics
    
    When users ask about option prices or Greeks:
    1. First use get_market_data to fetch current stock price, volatility, and risk-free rate
    2. Then use calculate_option_price or calculate_option_greeks with the fetched data
    3. Explain the results clearly
    
    Always mention the market data used in your calculations for transparency.""",
    tools=[calculate_option_price, calculate_option_greeks, get_market_data]
)

async def main():
    # Example query
    result = await Runner.run(options_calculator_agent, "What is the price of a call option on Tesla with a strike price of $100 and a maturity of 1 year?")
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
