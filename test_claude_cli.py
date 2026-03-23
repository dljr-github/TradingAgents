from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "claude_cli"
config["deep_think_llm"] = "claude-opus-4-6"       # Research manager, portfolio manager
config["quick_think_llm"] = "claude-sonnet-4-6"    # Analysts, debaters, trader
config["max_debate_rounds"] = 1
config["max_risk_discuss_rounds"] = 1
config["max_recur_limit"] = 150  # higher limit for CLI overhead
config["claude_cli_timeout"] = 300  # 5 min per invocation

# Use yfinance (no extra API keys needed)
config["data_vendors"] = {
    "core_stock_apis": "yfinance",
    "technical_indicators": "yfinance",
    "fundamental_data": "yfinance",
    "news_data": "yfinance",
}

ta = TradingAgentsGraph(debug=True, config=config)

# Test with SPY ETF
_, decision = ta.propagate("SPY", "2025-03-21")
print("\n" + "=" * 60)
print("FINAL DECISION:")
print("=" * 60)
print(decision)
