import pytest
import datetime
import numpy as np
from typing import Dict, List, Optional, Union

# Import all tools
from agent_store.agent_2.tools.options_tools import (
    calculate_option_price,
    calculate_option_greeks,
)
from agent_store.agent_2.tools.market_data import get_market_data
from agent_store.agent_2.tools.volatility_analysis import (
    calculate_implied_volatility,
    fetch_volatility_surface,
)
from agent_store.agent_2.tools.strategy_analysis import (
    analyze_option_strategy,
    calculate_payoff_diagram_data,
    StrategyAnalysisResult,
)
from agent_store.agent_2.tools.risk_management import (
    calculate_var,
    calculate_stress_test,
)
from agent_store.agent_2.tools.data_visualization import (
    plot_payoff_diagram,
    plot_greeks_sensitivities,
    plot_volatility_surface,
)

# Common fixtures
@pytest.fixture
def option_params():
    """Common option parameters for testing."""
    return {
        "underlying_price": 100.0,
        "strike_price": 100.0,
        "time_to_expiry": 1.0,
        "risk_free_rate": 0.05,
        "volatility": 0.2,
    }

@pytest.fixture
def test_strategy():
    """Test strategy configuration."""
    return {
        "strategy_name": "Bull Call Spread",
        "underlying_symbol": "AAPL",
        "options": [
            {
                "option_type": "call",
                "strike_price": 100.0,
                "time_to_expiry": 0.25,
                "position": "long",
                "quantity": 1,
            },
            {
                "option_type": "call",
                "strike_price": 110.0,
                "time_to_expiry": 0.25,
                "position": "short",
                "quantity": 1,
            },
        ],
    }

@pytest.fixture
def position_details():
    """Test position details for risk management."""
    return {
        "option_type": "call",
        "strike_price": 100.0,
        "time_to_expiry": 0.25,
        "quantity": 1,
        "position": "long",
    }

# Test Option Tools
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_calculate_option_price(option_params, option_type):
    """Test the Black-Scholes option pricing function."""
    price = calculate_option_price(
        option_type=option_type,
        **option_params
    )
    assert isinstance(price, float)
    assert price > 0

@pytest.mark.parametrize("option_type", ["call", "put"])
def test_calculate_option_greeks(option_params, option_type):
    """Test the Greeks calculation function."""
    greeks = calculate_option_greeks(
        option_type=option_type,
        **option_params
    )
    
    # Check that all Greeks are present and are floats
    expected_greeks = ["delta", "gamma", "theta", "vega", "rho"]
    for greek in expected_greeks:
        assert greek in greeks
        assert isinstance(greeks[greek], float)
    
    # Check delta bounds
    if option_type == "call":
        assert 0 <= greeks["delta"] <= 1
    else:
        assert -1 <= greeks["delta"] <= 0

# Test Market Data
def test_get_market_data():
    """Test market data retrieval."""
    data = get_market_data("AAPL")
    assert isinstance(data, dict)
    
    # Check required fields
    expected_fields = ["stock_price", "volatility", "risk_free_rate"]
    for field in expected_fields:
        assert field in data
        assert isinstance(data[field], float)

# Test Volatility Analysis
def test_calculate_implied_volatility(option_params):
    """Test implied volatility calculation."""
    impl_vol = calculate_implied_volatility(
        option_price=5.0,
        underlying_price=option_params["underlying_price"],
        strike_price=option_params["strike_price"],
        time_to_expiry=option_params["time_to_expiry"],
        risk_free_rate=option_params["risk_free_rate"],
        option_type="call",
        dividend_yield=None,
    )
    assert isinstance(impl_vol, float)
    assert 0 < impl_vol < 1  # Reasonable vol range

def test_fetch_volatility_surface():
    """Test volatility surface generation."""
    surface = fetch_volatility_surface(
        symbol="AAPL",
        expiry_dates=None,
        strike_range=None,
    )
    assert isinstance(surface, dict)
    
    # Check structure of returned data
    for expiry, strikes in surface.items():
        assert isinstance(expiry, str)
        assert isinstance(strikes, dict)
        for strike, vol in strikes.items():
            assert isinstance(strike, float)
            assert isinstance(vol, float)
            assert vol > 0

# Test Strategy Analysis
def test_analyze_option_strategy(test_strategy):
    """Test strategy analysis function."""
    result = analyze_option_strategy(
        strategy_name=test_strategy["strategy_name"],
        underlying_symbol=test_strategy["underlying_symbol"],
        options=test_strategy["options"],
        risk_free_rate=None,
        current_underlying_price=None,
        price_range_percentage=None,
    )
    assert isinstance(result, StrategyAnalysisResult)
    assert isinstance(result.max_profit, float)
    assert isinstance(result.max_loss, float)
    assert isinstance(result.breakeven_points, list)
    assert isinstance(result.risk_reward_ratio, float)
    assert isinstance(result.profit_probability, float)
    assert isinstance(result.payoff_range, dict)
    assert isinstance(result.greeks, dict)

def test_calculate_payoff_diagram_data(test_strategy):
    """Test payoff diagram data calculation."""
    data = calculate_payoff_diagram_data(
        strategy_name=test_strategy["strategy_name"],
        underlying_symbol=test_strategy["underlying_symbol"],
        options=test_strategy["options"],
        current_price=None,
        price_points=None,
    )
    assert isinstance(data, dict)
    assert "prices" in data
    assert "total_payoff" in data
    assert "component_payoffs" in data

# Test Risk Management
def test_calculate_var(position_details):
    """Test Value at Risk calculation."""
    var_result = calculate_var(
        position_type="option",
        underlying_symbol="AAPL",
        position_details=position_details,
        confidence_level=None,
        time_horizon=None,
        method=None,
        simulations=None,
    )
    assert isinstance(var_result, dict)

def test_calculate_stress_test(position_details):
    """Test stress testing function."""
    stress_result = calculate_stress_test(
        position_type="option",
        underlying_symbol="AAPL",
        position_details=position_details,
        stress_scenarios=None,
    )
    assert isinstance(stress_result, dict)

# Test Data Visualization
@pytest.mark.parametrize("return_format", [None, "base64", "json"])
def test_plot_payoff_diagram(test_strategy, return_format):
    """Test payoff diagram plotting."""
    result = plot_payoff_diagram(
        strategy_name=test_strategy["strategy_name"],
        underlying_symbol=test_strategy["underlying_symbol"],
        options=test_strategy["options"],
        current_price=None,
        price_points=None,
        return_format=return_format,
        plot_components=None,
        highlight_breakeven=None,
        show_current_price=None,
        plot_title=None,
        figsize=None,
    )
    assert isinstance(result, dict)
    assert "image" in result
    assert "data" in result
    assert "error" in result

@pytest.mark.parametrize("greek", ["delta", "gamma", "theta", "vega", "rho"])
@pytest.mark.parametrize("variable", ["underlying_price", "volatility", "time_to_expiry"])
def test_plot_greeks_sensitivities(option_params, greek, variable):
    """Test Greeks sensitivity plotting."""
    result = plot_greeks_sensitivities(
        option_type="call",
        strike_price=option_params["strike_price"],
        time_to_expiry=option_params["time_to_expiry"],
        underlying_price=option_params["underlying_price"],
        volatility=option_params["volatility"],
        risk_free_rate=option_params["risk_free_rate"],
        greek=greek,
        variable=variable,
        range_percentage=None,
        points=None,
        return_format=None,
        figsize=None,
    )
    assert isinstance(result, dict)
    assert "image" in result
    assert "data" in result
    assert "error" in result

@pytest.mark.parametrize("plot_type", ["3d", "heatmap"])
def test_plot_volatility_surface(plot_type):
    """Test volatility surface plotting."""
    # First get some volatility data
    vol_data = fetch_volatility_surface("AAPL", None, None)
    
    result = plot_volatility_surface(
        volatility_data=vol_data,
        underlying_symbol="AAPL",
        current_price=None,
        return_format=None,
        plot_type=plot_type,
        figsize=None,
    )
    assert isinstance(result, dict)
    assert "image" in result
    assert "data" in result
    assert "error" in result 