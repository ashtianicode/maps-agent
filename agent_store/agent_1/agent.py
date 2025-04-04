import asyncio
from agents import Agent
from agents import Runner
from agent_store.agent_1.options_tools import calculate_option_price, calculate_option_greeks

options_calculator_agent = Agent(
    name="Options Calculator",
    handoff_description="Specialist agent for options calculations",
    instructions="""You are an options calculator agent that helps users with options pricing and Greeks calculations.
    You can:
    1. Calculate option prices using the Black-Scholes model
    2. Calculate option Greeks (delta, gamma, theta, vega, rho)
    3. Explain the meaning and significance of different option metrics
    
    When users ask about option prices or Greeks, use the appropriate tool and explain the results clearly.""",
    tools=[calculate_option_price, calculate_option_greeks]
)

async def main():
    result = await Runner.run(options_calculator_agent, "What is the price of a call option on Tesla with a strike price of $100 and a maturity of 1 year?")
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
