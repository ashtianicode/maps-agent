# Advanced Options Strategist V2

A sophisticated options analysis and strategy agent with comprehensive tools for pricing, Greeks calculation, strategy analysis, risk management, and visualization.

## Features

The Advanced Options Strategist V2 offers several major improvements over the V1 agent:

1. **Expanded Tool Set**: Increased from 3 basic tools to 11 specialized tools
2. **Advanced Analysis Capabilities**: Strategy analysis, payoff diagrams, risk metrics
3. **Risk Management**: VaR calculations, stress testing, and scenario analysis
4. **Visualization**: Payoff diagrams, Greek sensitivities, volatility surfaces
5. **Volatility Analysis**: Implied volatility calculation and surface visualization

## Usage

The agent can be invoked with the Runner interface from the OpenAI Agents SDK:

```python
from agents import Runner
from agent_store.agent_2.agent import options_calculator_agent_v2

async def main():
    result = await Runner.run(
        options_calculator_agent_v2, 
        "What would be a good options strategy if I'm bullish on AAPL but want limited risk?"
    )
    print(result.final_output)
```

## Example Queries

The agent can handle a wide range of options-related queries, such as:

### Basic Options Pricing

- "What is the price of a European call option on Tesla with strike $200 expiring in 2 months?"
- "Calculate the price and Greeks for an at-the-money Microsoft put expiring in 45 days."

### Strategy Analysis

- "Analyze a bull call spread on NVDA with long $300 call and short $320 call, both expiring in 3 months."
- "Compare iron condor vs iron butterfly strategies on SPY with 30 days to expiration."

### Risk Assessment

- "What is the 1-day 95% VaR for a portfolio of 10 AAPL $170 calls and 5 AAPL $170 puts?"
- "Perform a stress test on my long straddle position on AMD to see what happens if volatility drops 10%."

### Visualization

- "Show me a payoff diagram for a protective collar on MSFT with long stock, long $250 put, and short $270 call."
- "Plot how delta changes with respect to the underlying price for a call option."

### Advanced Analytics

- "What's the implied volatility of an AMZN call option trading at $15.5 with strike $150 and 60 days to expiry?"
- "Fetch and visualize the volatility surface for AAPL options."

## Tools

The agent integrates the following specialized tools:

### Core Tools
- `calculate_option_price`: Calculates option prices using the Black-Scholes model
- `calculate_option_greeks`: Calculates option Greeks (delta, gamma, theta, vega, rho)
- `get_market_data`: Fetches market data like stock prices, volatility, risk-free rates

### Volatility Analysis
- `calculate_implied_volatility`: Calculates implied volatility from option prices
- `fetch_volatility_surface`: Retrieves volatility data across strikes and expirations

### Strategy Analysis
- `analyze_option_strategy`: Analyzes payoff, risk-reward, breakeven points for strategies
- `calculate_payoff_diagram_data`: Generates data for payoff visualizations

### Risk Management
- `calculate_var`: Calculates Value at Risk for options positions
- `calculate_stress_test`: Tests positions under various market scenarios

### Visualization
- `plot_payoff_diagram`: Creates visual payoff diagrams for strategies
- `plot_greeks_sensitivities`: Visualizes how Greeks change with market variables
- `plot_volatility_surface`: Creates 3D or heatmap visualizations of vol surfaces

## Installation Requirements

To use all features of this agent, ensure the following packages are installed:

```bash
pip install numpy scipy pandas matplotlib
```

For the plotting capabilities, matplotlib is required. The agent will gracefully handle missing dependencies by returning data without visualizations when packages are unavailable.

## Limitations

- The agent uses the Black-Scholes model, which has known limitations for American options
- In demo mode, market data may be simulated rather than real-time
- Some advanced features like correlation analysis for portfolio VaR use simplified approaches 