from __future__ import annotations

from datetime import timedelta
from math import ceil

import pandas as pd
import pytest

import app as app_module
import live
from backtest import _slipped_prices, _trade_costs
from data_loader import ensure_app_ready
from pricing import reprice_trade


def _entry_timestamp() -> pd.Timestamp:
    """Choose a research bar inside the metadata-defined paper entry window."""
    data = ensure_app_ready()
    timestamps = pd.to_datetime(data["signals"]["datetime"])
    start, end = data["meta"]["entry_window"]
    eligible = timestamps.dt.strftime("%H:%M").between(start, end)
    return pd.Timestamp(timestamps.loc[eligible].iloc[0])


def _set_market(desk: live.PaperTradingDesk, timestamp: pd.Timestamp) -> None:
    """Install a deterministic, metadata-priced ATM quote for paper API tests."""
    data = ensure_app_ready()
    meta = data["meta"]
    signals = data["signals"]
    row_index = int(
        (pd.to_datetime(signals["datetime"]) - timestamp)
        .abs()
        .to_numpy()
        .argmin()
    )
    spot = float(signals.iloc[row_index]["close"])
    iv = float(meta["option_pricing"]["iv"])
    expiry = timestamp.date() + timedelta(
        days=ceil(float(meta["option_pricing"]["days_to_expiry"]))
    )
    chain = live._option_chain(
        spot,
        iv,
        expiry,
        timestamp.to_pydatetime(),
        meta,
    )
    desk.latest_market = {
        "timestamp": timestamp,
        "spot": spot,
        "iv": iv,
        "expiry": expiry,
        "option_chain": chain,
    }


def test_same_bar_paper_buy_and_sell_charges_two_slips_and_backtest_costs() -> None:
    """An unchanged same-bar premium loses two slips plus canonical costs."""
    desk = live.PaperTradingDesk()
    timestamp = _entry_timestamp()
    _set_market(desk, timestamp)

    opened = desk.submit_order("CE", 1)
    closed = desk.close_order(opened["position_id"])

    meta = ensure_app_ready()["meta"]
    quantity = opened["quantity"]
    slip = (
        float(meta["costs"]["slippage_ticks_default"])
        * float(meta["costs"]["tick"])
    )
    costs = _trade_costs(
        opened["buy"],
        closed["sell"],
        quantity,
        meta["costs"],
    )
    assert closed["pnl"] == pytest.approx(-2 * slip * quantity - costs)
    assert closed["costs"] == pytest.approx(costs)


def test_risk_rules_block_fourth_order_loss_cap_and_square_off(monkeypatch) -> None:
    """API paper orders enforce metadata trade-count, loss, and time limits."""
    desk = live.paper_desk
    timestamp = _entry_timestamp()
    monkeypatch.setattr(
        app_module,
        "_refresh_paper_market",
        lambda payload: _set_market(desk, timestamp),
    )
    client = app_module.app.test_client()
    order = {"side": "CE", "lots": 1}

    for _ in range(int(desk.meta["max_trades_per_day"])):
        opened = client.post("/api/live/order", json=order)
        assert opened.status_code == 200
        closed = client.post(
            "/api/live/close",
            json={"position_id": opened.get_json()["position"]["position_id"]},
        )
        assert closed.status_code == 200
    fourth = client.post("/api/live/order", json=order)
    assert fourth.status_code == 400
    assert "Maximum" in fourth.get_json()["error"]

    live.reset_paper_accounts()
    desk = live.paper_desk
    monkeypatch.setattr(
        app_module,
        "_refresh_paper_market",
        lambda payload: _set_market(desk, timestamp),
    )
    desk.user.realized_pnl[timestamp.date().isoformat()] = -float(
        desk.meta["daily_loss_cap_rs"]
    )
    loss_capped = client.post("/api/live/order", json=order)
    assert loss_capped.status_code == 400
    assert "daily loss cap" in loss_capped.get_json()["error"]

    live.reset_paper_accounts()
    desk = live.paper_desk
    hour, minute = map(int, desk.meta["square_off"].split(":"))
    after_square_off = timestamp.replace(hour=hour, minute=minute)
    monkeypatch.setattr(
        app_module,
        "_refresh_paper_market",
        lambda payload: _set_market(desk, after_square_off),
    )
    late = client.post("/api/live/order", json=order)
    assert late.status_code == 400
    assert "New orders stop" in late.get_json()["error"]


def test_copy_bot_replay_matches_expected_ai_fills_and_pnl() -> None:
    """Copying the full replay reproduces the stored AI trade count and P&L."""
    data = ensure_app_ready()
    live.set_live_provider("replay")
    live.reset_paper_accounts()
    live._replay_provider._cursor = len(live._replay_provider._frame) - 1
    live.paper_desk.set_settings(
        {"auto_exit": True, "enforce_rules": True, "copy_bot": True}
    )

    snapshot = live.get_live_snapshot(
        "replay",
        confidence=float(data["meta"]["rf"]["tau_default"]),
        iv=float(data["meta"]["option_pricing"]["iv"]),
    )

    expected = data["meta"]["expected"]["ai_model"]
    assert snapshot["user_account"]["trades"] == expected["Trades taken"]
    assert snapshot["user_account"]["pnl"] == pytest.approx(
        expected["Net P&L (Rs)"], abs=1.0
    )


@pytest.mark.parametrize("exit_reason", ["target", "stop", "time"])
def test_auto_exit_matches_pricing_simulator(exit_reason: str) -> None:
    """Paper target, stop, and max-hold exits occur on pricing.py's exact bar."""
    data = ensure_app_ready()
    meta = data["meta"]
    signals = data["signals"].reset_index(drop=True)
    volatility = float(meta["option_pricing"]["iv"])
    candidate = next(
        (index, row)
        for index, row in signals.iterrows()
        if row["decision"] in {"BUY_CE", "BUY_PE"}
        and reprice_trade(signals, int(index), iv=volatility, meta=meta)[
            "exit_reason"
        ]
        == exit_reason
    )
    index, row = candidate
    expected = reprice_trade(signals, int(index), iv=volatility, meta=meta)
    timestamp = pd.Timestamp(row["datetime"])
    initial_time = float(meta["option_pricing"]["days_to_expiry"]) / 365.0
    expiry = timestamp.date() + timedelta(
        days=ceil(float(meta["option_pricing"]["days_to_expiry"]))
    )
    desk = live.PaperTradingDesk()
    entry_buy, _ = _slipped_prices(
        expected["entry_premium"],
        expected["entry_premium"],
        meta["costs"]["slippage_ticks_default"],
        meta["costs"]["tick"],
    )
    desk.user.open_position(
        side=expected["kind"],
        strike=expected["strike"],
        lots=1,
        raw_premium=expected["entry_premium"],
        buy=entry_buy,
        timestamp=timestamp,
        target_pct=float(row["target_pct"]),
        stop_pct=float(row["stop_pct"]),
        expiry=expiry,
        initial_time_to_expiry=initial_time,
        source="manual",
    )

    for current_index in range(index, int(expected["exit_index"]) + 1):
        current = signals.iloc[current_index]
        current_time = pd.Timestamp(current["datetime"])
        desk._advance_positions(
            desk.user,
            current_time,
            float(current["close"]),
            volatility,
            new_bar=True,
            force_session_end=False,
            auto_exit=True,
        )

    assert len(desk.user.history) == 1
    trade = desk.user.history[0]
    assert trade["exit_reason"] == exit_reason
    assert trade["closed_at"] == pd.Timestamp(expected["exit_datetime"]).isoformat()
    _, expected_sell = _slipped_prices(
        expected["entry_premium"],
        expected["exit_premium"],
        meta["costs"]["slippage_ticks_default"],
        meta["costs"]["tick"],
    )
    assert trade["sell"] == pytest.approx(expected_sell)
