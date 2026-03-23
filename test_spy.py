from tradingagents.graph.trading_graph import TradingAgentsGraph

from tradingagents.default_config import DEFAULT_CONFIG

config = {**DEFAULT_CONFIG}
config.update({
    "llm_provider": "claude_cli",
    "deep_think_llm": "claude-sonnet-4-6",
    "quick_think_llm": "claude-sonnet-4-6",
    "claude_cli_path": "/home/darrell/.local/bin/claude",
    "claude_cli_timeout": 600,
})

graph = TradingAgentsGraph(
    config=config,
    debug=True,
)

# Run on SPY with today's date
final_state, decision = graph.propagate("SPY", "2026-03-07")
print("\n\n=== FINAL DECISION ===")
print(decision)
