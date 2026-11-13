from __future__ import annotations

from copy import deepcopy
from datetime import date

import numpy as np
import pandas as pd

from backtest import _slipped_prices, run_backtest
from data_loader import ensure_app_ready
from indicators import compute_vwap, detect_candidate_signal
from pricing import black_scholes_delta, black_scholes_price


DATA = ensure_app_ready()


def _synthetic_bars() -> pd.DataFrame:
    """Create rising-then-falling bars that cross both pullback boundaries."""
    closes = [100.0, 90.0, 110.0, 90.0]
    return pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [
                    "2026-11-13 09:20:00+05:30",
                    "2026-11-13 09:25:00+05:30",
                    "2026-11-13 09:30:00+05:30",
                    "2026-11-13 09:35:00+05:30",
                ]
            ),
            "open": closes,
            "high": np.asarray(closes) + 1,
            "low": np.asarray(closes) - 1,
            "close": closes,
            "volume": [0.0] * len(closes),
        }
    )


def test_candidate_rule_detects_call_and_put_pullbacks() -> None:
    """The candidate rule should identify each direction on its crossing bar."""
    meta = deepcopy(DATA["meta"])
    meta["indicators"]["bb_window"] = 2
    meta["indicators"]["bb_k"] = 2.0
    decisions = detect_candidate_signal(
        _synthetic_bars(),
        meta,
        DATA["model_card"],
    )

    assert decisions.tolist() == ["HOLD", "HOLD", "BUY_CE", "BUY_PE"]


def test_zero_volume_vwap_is_equal_weight_typical_price() -> None:
    """Zero-volume sessions should use the cumulative typical-price mean."""
    bars = pd.DataFrame(
        {
            "datetime": pd.date_range("2026-11-13 09:15", periods=3, freq="5min"),
            "open": [10, 20, 30],
            "high": [11, 21, 31],
            "low": [9, 19, 29],
            "close": [10, 20, 30],
            "volume": [0, 0, 0],
        }
    )

    assert np.allclose(compute_vwap(bars), [10, 15, 20])


def test_slippage_respects_tick_cost_and_sell_floor() -> None:
    """Slippage applies to both fills and the sell premium cannot go below the floor."""
    tick = float(DATA["meta"]["costs"]["tick"])

    buy, sell = _slipped_prices(10.0, 0.0, 2, tick)

    assert buy == 10.0 + 2 * tick
    assert sell == 0.05


def test_backtest_limits_daily_entries_and_never_overlaps() -> None:
    """Meta-driven trade caps and busy-until rules prevent excessive or overlapping entries."""
    settings = deepcopy(DATA["meta"])
    settings["daily_loss_cap_rs"] = 1.0
    settings["costs"]["brokerage_per_order"] = float(settings["capital"])
    result = run_backtest(DATA["signals"], use_rf=False, meta=settings)
    trades = result["trades"]

    counts = pd.Series([trade["day"] for trade in trades]).value_counts()
    assert counts.max() <= settings["max_trades_per_day"]
    for previous, following in zip(trades, trades[1:]):
        assert following["row_index"] > previous["exit_index"]
    for day, day_trades in pd.DataFrame(trades).groupby("day"):
        assert len(day_trades) <= settings["max_trades_per_day"]


def test_higher_confidence_never_adds_ai_trades() -> None:
    """Increasing the threshold cannot increase accepted candidate trades."""
    lower = run_backtest(DATA["signals"], conf=0.30)["trades"]
    higher = run_backtest(DATA["signals"], conf=0.80)["trades"]

    assert len(higher) <= len(lower)


def test_black_scholes_parity_and_delta_bounds() -> None:
    """Call-put parity holds and deltas stay within their theoretical intervals."""
    pricing = DATA["meta"]["option_pricing"]
    spot = float(DATA["signals"].iloc[0]["close"])
    strike = float(pricing["strike_step"]) * round(
        spot / float(pricing["strike_step"])
    )
    time_to_expiry = float(pricing["days_to_expiry"]) / 365.0
    volatility = float(pricing["iv"])
    rate = float(pricing["rate"])
    call = black_scholes_price(
        spot, strike, time_to_expiry, "CE", volatility, rate
    )
    put = black_scholes_price(
        spot, strike, time_to_expiry, "PE", volatility, rate
    )
    call_delta = black_scholes_delta(
        spot, strike, time_to_expiry, "CE", volatility, rate
    )
    put_delta = black_scholes_delta(
        spot, strike, time_to_expiry, "PE", volatility, rate
    )

    assert abs(call - put - (spot - strike * np.exp(-rate * time_to_expiry))) < 1e-6
    assert 0.0 <= call_delta <= 1.0
    assert -1.0 <= put_delta <= 0.0
