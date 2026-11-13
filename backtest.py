"""Out-of-sample backtest, transaction-cost, risk, and prediction calculations."""

from __future__ import annotations

from math import floor, sqrt
from time import perf_counter
from typing import Any, Mapping

import numpy as np
import pandas as pd

from data_loader import ensure_app_ready
from pricing import reprice_trade


def _settings(meta: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    """Resolve backtest settings from a caller or the process data cache."""
    return meta or ensure_app_ready()["meta"]


def _prepare_signals(frame: pd.DataFrame) -> pd.DataFrame:
    """Copy, validate, and chronologically sort signal bars."""
    signals = frame.copy()
    if "datetime" not in signals or "decision" not in signals:
        raise ValueError("Backtest input requires datetime and decision columns.")
    signals["datetime"] = pd.to_datetime(signals["datetime"], errors="coerce")
    if signals["datetime"].isna().any():
        raise ValueError("Backtest input contains an invalid datetime.")
    if "rf_confidence" not in signals:
        signals["rf_confidence"] = np.nan
    signals = signals.sort_values("datetime").reset_index(drop=True)
    if signals.empty:
        raise ValueError("Backtest input contains no bars.")
    return signals


def _trade_costs(
    buy: float,
    sell: float,
    quantity: int,
    costs: Mapping[str, Any],
) -> float:
    """Compute brokerage, exchange, statutory, stamp-duty, and GST charges."""
    buy_value = buy * quantity
    sell_value = sell * quantity
    brokerage = 2.0 * float(costs["brokerage_per_order"])
    exchange = (buy_value + sell_value) * float(costs["exchange_txn"])
    sebi = (buy_value + sell_value) * float(costs["sebi_fee"])
    return float(
        brokerage
        + exchange
        + sebi
        + sell_value * float(costs["stt_sell"])
        + buy_value * float(costs["stamp_buy"])
        + float(costs["gst"]) * (brokerage + exchange + sebi)
    )


def _slipped_prices(
    entry_premium: float,
    exit_premium: float,
    slippage_ticks: int | float,
    tick_size: float,
) -> tuple[float, float]:
    """Apply per-fill slippage and enforce the minimum executable sell premium."""
    slippage = float(slippage_ticks) * float(tick_size)
    return float(entry_premium + slippage), float(max(exit_premium - slippage, 0.05))


def _prediction_metrics(
    predictions: list[tuple[float, bool]],
    confidence: float,
) -> dict[str, Any]:
    """Measure classifier decisions against re-priced candidate outcomes."""
    if not predictions:
        return {
            "Accuracy rate %": 0.0,
            "Precision %": 0.0,
            "Recall %": 0.0,
            "F1-score": 0.0,
            "TP": 0,
            "FP": 0,
            "TN": 0,
            "FN": 0,
        }
    predicted = np.asarray([probability >= confidence for probability, _ in predictions])
    actual = np.asarray([outcome for _, outcome in predictions], dtype=bool)
    tp = int(np.count_nonzero(predicted & actual))
    fp = int(np.count_nonzero(predicted & ~actual))
    tn = int(np.count_nonzero(~predicted & ~actual))
    fn = int(np.count_nonzero(~predicted & actual))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "Accuracy rate %": round((tp + tn) / len(predictions) * 100.0, 1),
        "Precision %": round(precision * 100.0, 1),
        "Recall %": round(recall * 100.0, 1),
        "F1-score": round(f1, 3),
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
    }


def _trade_metrics(
    trades: list[dict[str, Any]],
    capital: float,
    all_days: list[str],
) -> dict[str, Any]:
    """Calculate realized return, drawdown, costs, win rate, and daily Sharpe."""
    pnl_values = np.asarray([trade["pnl"] for trade in trades], dtype=float)
    costs = float(sum(trade["costs"] for trade in trades))
    gross_pnl = float(sum(trade["gross_pnl"] for trade in trades))
    net_pnl = float(pnl_values.sum()) if len(pnl_values) else 0.0
    wins = pnl_values[pnl_values > 0]
    losses = pnl_values[pnl_values <= 0]
    profit_factor = (
        float(wins.sum() / abs(losses.sum()))
        if losses.sum() < 0
        else (float("inf") if len(wins) else 0.0)
    )

    daily_pnl = {day: 0.0 for day in all_days}
    for trade in trades:
        daily_pnl[trade["day"]] += float(trade["pnl"])
    daily_returns = np.asarray(
        [daily_pnl[day] / capital for day in all_days],
        dtype=float,
    )
    daily_std = float(np.std(daily_returns, ddof=1)) if len(daily_returns) > 1 else 0.0
    sharpe = (
        float(np.mean(daily_returns) / daily_std * sqrt(252.0))
        if daily_std > 0
        else 0.0
    )

    equity = capital
    peak = capital
    max_drawdown = 0.0
    equity_curve = [capital]
    for trade in trades:
        equity += float(trade["pnl"])
        peak = max(peak, equity)
        max_drawdown = min(max_drawdown, (equity - peak) / peak * 100.0)
        equity_curve.append(equity)

    return {
        "Trades taken": len(trades),
        "Net P&L (Rs)": round(net_pnl, 0),
        "ROI %": round(net_pnl / capital * 100.0, 2),
        "Sharpe (ann.)": round(sharpe, 2),
        "Max drawdown %": round(max_drawdown, 2),
        "Win rate %": round(float(np.count_nonzero(pnl_values > 0) / len(trades) * 100.0), 1)
        if trades
        else 0.0,
        "Profit factor": round(profit_factor, 2) if np.isfinite(profit_factor) else None,
        "Total costs (Rs)": round(costs, 0),
        "Gross P&L (Rs)": round(gross_pnl, 0),
        "equity_curve": equity_curve,
        "daily_pnl": daily_pnl,
    }


def run_backtest(
    frame: pd.DataFrame,
    conf: float | None = None,
    slip_ticks: int | float | None = None,
    iv: float | None = None,
    use_rf: bool = True,
    meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the AI-filtered or no-ML backtest for the selected bars."""
    settings = _settings(meta)
    signals = _prepare_signals(frame)
    threshold = float(settings["rf"]["tau_default"] if conf is None else conf)
    slippage_ticks = int(
        settings["costs"]["slippage_ticks_default"] if slip_ticks is None else slip_ticks
    )
    volatility = float(settings["option_pricing"]["iv"] if iv is None else iv)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Confidence threshold must be between zero and one.")
    if slippage_ticks < 0:
        raise ValueError("Slippage ticks cannot be negative.")
    if volatility <= 0:
        raise ValueError("Option implied volatility must be positive.")

    capital = float(settings["capital"])
    lot_size = int(settings["lot_size"])
    risk_fraction = float(settings["risk_per_trade"])
    maximum_trades = int(settings["max_trades_per_day"])
    loss_cap = float(settings["daily_loss_cap_rs"])
    costs_meta = settings["costs"]
    all_days = sorted(signals["datetime"].dt.date.astype(str).unique())

    trade_count_by_day: dict[str, int] = {}
    pnl_by_day: dict[str, float] = {}
    trades: list[dict[str, Any]] = []
    predictions: list[tuple[float, bool]] = []
    priced_candidates: dict[int, dict[str, Any]] = {}
    busy_until = -1
    prediction_started = perf_counter()

    for index, row in signals.iterrows():
        decision = str(row["decision"]).strip()
        if decision not in {"BUY_CE", "BUY_PE"}:
            continue
        confidence_value = pd.to_numeric(row["rf_confidence"], errors="coerce")
        if pd.notna(confidence_value):
            priced = reprice_trade(signals, int(index), iv=volatility, meta=settings)
            priced_candidates[index] = priced
            predictions.append((float(confidence_value), bool(priced["hit_target"])))
        if use_rf and (
            pd.isna(confidence_value) or float(confidence_value) < threshold
        ):
            continue

        day = row["datetime"].date().isoformat()
        trade_count_by_day.setdefault(day, 0)
        pnl_by_day.setdefault(day, 0.0)
        if index <= busy_until:
            continue
        if trade_count_by_day[day] >= maximum_trades:
            continue
        if pnl_by_day[day] <= -loss_cap:
            continue

        priced = priced_candidates.get(index)
        if priced is None:
            priced = reprice_trade(signals, int(index), iv=volatility, meta=settings)
            priced_candidates[index] = priced
        buy, sell = _slipped_prices(
            float(priced["entry_premium"]),
            float(priced["exit_premium"]),
            slippage_ticks,
            float(costs_meta["tick"]),
        )
        stop = float(row["stop_pct"])
        lots = max(1, floor(risk_fraction * capital / (stop * buy * lot_size)))
        quantity = int(lots * lot_size)
        trade_cost = _trade_costs(buy, sell, quantity, costs_meta)
        gross_pnl = (sell - buy) * quantity
        net_pnl = gross_pnl - trade_cost
        trades.append(
            {
                "day": day,
                "row_index": int(index),
                "exit_index": int(priced["exit_index"]),
                "entry_time": row["datetime"],
                "exit_time": priced["exit_datetime"],
                "side": "CE" if decision == "BUY_CE" else "PE",
                "spot": float(row["close"]),
                "strike": float(priced["strike"]),
                "lots": int(lots),
                "quantity": quantity,
                "buy": float(buy),
                "sell": float(sell),
                "exit_reason": str(priced["exit_reason"]),
                "costs": float(trade_cost),
                "gross_pnl": float(gross_pnl),
                "pnl": float(net_pnl),
                "hit_target": bool(priced["hit_target"]),
            }
        )
        trade_count_by_day[day] += 1
        pnl_by_day[day] += net_pnl
        busy_until = int(priced["exit_index"])

    metrics = _trade_metrics(trades, capital, all_days)
    metrics.update(_prediction_metrics(predictions, threshold))
    prediction_scoring_ms = (perf_counter() - prediction_started) * 1000.0
    return {
        "trades": trades,
        "metrics": {
            key: value
            for key, value in metrics.items()
            if key not in {"equity_curve", "daily_pnl"}
        },
        "equity_curve": metrics["equity_curve"],
        "daily_pnl": metrics["daily_pnl"],
        "prediction_metrics": _prediction_metrics(predictions, threshold),
        "prediction_scoring_ms": prediction_scoring_ms,
        "candidate_count": int(
            signals["decision"].isin(["BUY_CE", "BUY_PE"]).sum()
        ),
        "approved_count": int(
            sum(
                1
                for probability, _ in predictions
                if probability >= threshold
            )
        ),
    }


def run_correctness_gate() -> dict[str, Any]:
    """Run and assert the stored default AI and no-ML expected results."""
    data = ensure_app_ready()
    settings = data["meta"]
    confidence = float(settings["rf"]["tau_default"])
    slippage = int(settings["costs"]["slippage_ticks_default"])
    ai_result = run_backtest(data["signals"], confidence, slippage, use_rf=True)
    no_ml_result = run_backtest(data["signals"], confidence, slippage, use_rf=False)
    for name, result in (("ai_model", ai_result), ("no_ml", no_ml_result)):
        expected = settings["expected"][name]
        for key, expected_value in expected.items():
            actual_value = result["metrics"][key]
            if key == "Net P&L (Rs)":
                matches = abs(float(actual_value) - float(expected_value)) <= 1.0
            else:
                matches = actual_value == expected_value
            if not matches:
                raise AssertionError(
                    f"Default {name} mismatch for {key}: "
                    f"computed {actual_value!r}, expected {expected_value!r}"
                )
    return {"ai_model": ai_result["metrics"], "no_ml": no_ml_result["metrics"]}
