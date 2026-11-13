from __future__ import annotations

import pandas as pd

from backtest import run_backtest
from data_loader import load_strategy_meta

META = load_strategy_meta()


def test_correctness_gate_matches_expected_results() -> None:
    signals = pd.read_csv("data/review3_signals.csv")
    conf = float(META["rf"]["tau_default"])
    slip_ticks = int(META["costs"]["slippage_ticks_default"])

    ai = run_backtest(signals, conf=conf, slip_ticks=slip_ticks, use_rf=True)["metrics"]
    no_ml = run_backtest(signals, conf=conf, slip_ticks=slip_ticks, use_rf=False)["metrics"]

    expected_ai = META["expected"]["ai_model"]
    expected_no_ml = META["expected"]["no_ml"]

    for key, expected_value in expected_ai.items():
        if key in {"Net P&L (Rs)"}:
            assert abs(float(ai[key]) - float(expected_value)) <= 1, (key, ai[key], expected_value)
        else:
            assert ai[key] == expected_value, (key, ai[key], expected_value)

    for key, expected_value in expected_no_ml.items():
        if key in {"Net P&L (Rs)"}:
            assert abs(float(no_ml[key]) - float(expected_value)) <= 1, (key, no_ml[key], expected_value)
        else:
            assert no_ml[key] == expected_value, (key, no_ml[key], expected_value)
