"""Flask routes for the LuminX backtest dashboard and paper-only live demo."""

from __future__ import annotations

import csv
import io
import math
import os
import threading
import time
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from flask import Flask, Response, jsonify, render_template, request

import config
import live as live_module
from backtest import run_backtest
from data_loader import ensure_app_ready
from live import default_expiry_date, get_live_snapshot, set_live_provider

app = Flask(__name__)
IST = ZoneInfo("Asia/Kolkata")
STARTUP_DATA = ensure_app_ready()
META = STARTUP_DATA["meta"]
_provider_lock = threading.Lock()
_active_live_provider = config.LIVE_PROVIDER


def _elapsed(started: float) -> float:
    """Return elapsed wall-clock milliseconds rounded for API output."""
    return round((time.perf_counter() - started) * 1000.0, 2)


def _parse_clamped_float(
    raw: str | None,
    default: float,
    minimum: float,
    maximum: float,
    label: str,
    decimals: int,
) -> float:
    """Parse a numeric query value, clamp it to its supported slider range."""
    if raw is None or raw == "":
        return float(default)
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite.")
    return round(min(maximum, max(minimum, value)), decimals)


def _parse_slippage(raw: str | None) -> int:
    """Parse integer slippage ticks and clamp to the supported control range."""
    if raw is None or raw == "":
        return int(META["costs"]["slippage_ticks_default"])
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("slip must be an integer number of ticks.") from exc
    if not math.isfinite(value) or not value.is_integer():
        raise ValueError("slip must be an integer number of ticks.")
    return int(min(10, max(0, value)))


def _parse_date(raw: str | None) -> str | None:
    """Validate an optional test date, with an empty value meaning all days."""
    if raw is None or raw == "" or raw.lower() == "all":
        return None
    try:
        return date.fromisoformat(raw).isoformat()
    except ValueError as exc:
        raise ValueError("date must be 'all' or an ISO calendar date.") from exc


def _select_signals(selected_day: str | None) -> pd.DataFrame:
    """Return a copy of the full out-of-sample data or one selected test day."""
    signals = STARTUP_DATA["signals"]
    if selected_day is None:
        return signals.copy()
    datetimes = pd.to_datetime(signals["datetime"])
    selected = signals.loc[datetimes.dt.date.astype(str) == selected_day].copy()
    if selected.empty:
        raise ValueError("date is not present in the out-of-sample test data.")
    return selected


def _parameters() -> tuple[float, int, float, str | None]:
    """Parse and normalize backtest query values."""
    confidence = _parse_clamped_float(
        request.args.get("conf"),
        float(META["rf"]["tau_default"]),
        0.30,
        0.80,
        "conf",
        2,
    )
    slippage = _parse_slippage(request.args.get("slip"))
    default_iv_pct = float(META["option_pricing"]["iv"]) * 100.0
    iv_pct = _parse_clamped_float(
        request.args.get("iv"),
        default_iv_pct,
        5.0,
        60.0,
        "iv",
        1,
    )
    return confidence, slippage, iv_pct / 100.0, _parse_date(request.args.get("date"))


def _json_error(message: str, started: float, status: int = 400) -> tuple[Response, int]:
    """Build a bounded JSON error response with request timing."""
    return jsonify({"error": message, "elapsed_ms": _elapsed(started)}), status


def _equity_timeline(
    result: dict[str, Any],
    capital: float,
    start_date: str,
) -> list[dict[str, Any]]:
    """Pair realized equity points with their entry-day dates."""
    points = [{"date": start_date, "equity": capital}]
    equity = capital
    for trade in result["trades"]:
        equity += float(trade["pnl"])
        points.append(
            {
                "date": pd.Timestamp(trade["entry_time"]).isoformat(),
                "equity": equity,
            }
        )
    return points


def _buy_and_hold(frame: pd.DataFrame, capital: float) -> list[dict[str, Any]]:
    """Scale the underlying index close series from the initial capital."""
    closes = pd.to_numeric(frame["close"], errors="coerce")
    if closes.empty or pd.isna(closes.iloc[0]) or closes.iloc[0] == 0:
        return []
    initial = float(closes.iloc[0])
    return [
        {
            "date": str(timestamp),
            "equity": float(capital * close / initial),
        }
        for timestamp, close in zip(frame["datetime"], closes)
    ]


def _exit_label(reason: str, square_off: str) -> str:
    """Map simulator exit codes to the requested plain-English table labels."""
    labels = {
        "target": "Target hit",
        "stop": "Stop loss hit",
        "time": "Max hold time",
        "session_end": "Session end",
    }
    if reason == "square_off":
        square_time = datetime.strptime(square_off, "%H:%M")
        return f"{square_time.strftime('%I:%M %p').lstrip('0')} exit"
    return labels.get(reason, reason.replace("_", " ").title())


@app.get("/")
def index() -> str:
    """Render the single-page LuminX dashboard."""
    return render_template("index.html")


@app.get("/api/meta")
def api_meta() -> tuple[Response, int] | Response:
    """Return slider defaults, test days, validation, risk, and provider settings."""
    started = time.perf_counter()
    dates = sorted(
        pd.to_datetime(STARTUP_DATA["signals"]["datetime"])
        .dt.date.astype(str)
        .unique()
        .tolist()
    )
    payload = {
        "meta": META,
        "model_card": STARTUP_DATA["model_card"],
        "defaults": {
            "conf": META["rf"]["tau_default"],
            "slip": META["costs"]["slippage_ticks_default"],
            "iv": float(META["option_pricing"]["iv"]) * 100.0,
            "date": "all",
        },
        "today": datetime.now(IST).date().isoformat(),
        "default_expiry": default_expiry_date(),
        "ranges": {
            "conf": {"min": 0.30, "max": 0.80, "step": 0.01},
            "slip": {"min": 0, "max": 10, "step": 1},
            "iv": {"min": 5, "max": 60, "step": 0.1},
        },
        "test_days": dates,
        "validation": META["validation"],
        "risk_rules": {
            "max_trades_per_day": META["max_trades_per_day"],
            "daily_loss_cap_rs": META["daily_loss_cap_rs"],
            "risk_per_trade": META["risk_per_trade"],
            "square_off": META["square_off"],
        },
        "provider": _active_live_provider,
        "live_refresh_seconds": config.LIVE_REFRESH_SECONDS,
        "replay_seconds": config.REPLAY_SECONDS,
        "iv_source": config.IV_SOURCE,
        "elapsed_ms": _elapsed(started),
    }
    return jsonify(payload)


@app.get("/api/backtest")
def api_backtest() -> tuple[Response, int] | Response:
    """Calculate AI and no-ML performance for the selected controls."""
    started = time.perf_counter()
    try:
        confidence, slippage, volatility, selected_day = _parameters()
        signals = _select_signals(selected_day)
    except ValueError as exc:
        return _json_error(str(exc), started)

    calculation_started = time.perf_counter()
    ai_result = run_backtest(
        signals,
        conf=confidence,
        slip_ticks=slippage,
        iv=volatility,
        use_rf=True,
    )
    no_ml_result = run_backtest(
        signals,
        conf=confidence,
        slip_ticks=slippage,
        iv=volatility,
        use_rf=False,
    )
    backtest_ms = _elapsed(calculation_started)
    candidate_count = ai_result["candidate_count"]
    scoring_ms = ai_result["prediction_scoring_ms"]
    trades = []
    for trade in ai_result["trades"]:
        trades.append(
            {
                **trade,
                "entry_time": pd.Timestamp(trade["entry_time"]).isoformat(),
                "exit_time": pd.Timestamp(trade["exit_time"]).isoformat(),
                "exit_reason_label": _exit_label(
                    trade["exit_reason"],
                    str(META["square_off"]),
                ),
            }
        )
    payload = {
        "conf": confidence,
        "slip_ticks": slippage,
        "iv": volatility,
        "date": selected_day or "all",
        "metrics": ai_result["metrics"],
        "no_ml_metrics": no_ml_result["metrics"],
        "comparison": {
            "ai_model": ai_result["metrics"],
            "no_ml": no_ml_result["metrics"],
        },
        "trades": trades,
        "equity_curve": ai_result["equity_curve"],
        "no_ml_equity_curve": no_ml_result["equity_curve"],
        "equity_curves": {
            "ai_model": _equity_timeline(
                ai_result,
                float(META["capital"]),
                pd.Timestamp(signals.iloc[0]["datetime"]).isoformat(),
            ),
            "no_ml": _equity_timeline(
                no_ml_result,
                float(META["capital"]),
                pd.Timestamp(signals.iloc[0]["datetime"]).isoformat(),
            ),
            "buy_and_hold": _buy_and_hold(signals, float(META["capital"])),
        },
        "prediction_metrics": ai_result["prediction_metrics"],
        "signals": {
            "candidates": ai_result["candidate_count"],
            "approved": ai_result["approved_count"],
        },
        "timings": {
            "backtest_ms": backtest_ms,
            "scoring_ms_per_signal": (
                round(scoring_ms / candidate_count, 4) if candidate_count else 0.0
            ),
            "api_ms": _elapsed(started),
        },
        "elapsed_ms": _elapsed(started),
    }
    return jsonify(payload)


@app.get("/api/chart")
def api_chart() -> tuple[Response, int] | Response:
    """Return OHLC bars, indicator lines, and confidence-aware candidate markers."""
    started = time.perf_counter()
    try:
        confidence, _, _, selected_day = _parameters()
        signals = _select_signals(selected_day)
    except ValueError as exc:
        return _json_error(str(exc), started)
    if selected_day is None:
        dates = pd.to_datetime(signals["datetime"]).dt.date
        latest_day = dates.max()
        signals = signals.loc[dates == latest_day].copy()
    candles = []
    markers = []
    for _, row in signals.iterrows():
        timestamp = pd.Timestamp(row["datetime"]).isoformat()
        candles.append(
            {
                "datetime": timestamp,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "vwap": _finite_or_none(row["vwap"]),
                "bb_upper": _finite_or_none(row["bb_upper"]),
                "bb_lower": _finite_or_none(row["bb_lower"]),
                "z_score": _finite_or_none(row["z_score"]),
            }
        )
        decision = str(row["decision"])
        if decision in {"BUY_CE", "BUY_PE"}:
            probability = _finite_or_none(row["rf_confidence"])
            markers.append(
                {
                    "datetime": timestamp,
                    "price": float(row["close"]),
                    "decision": decision,
                    "confidence": probability,
                    "approved": probability is not None and probability >= confidence,
                    "z_score": _finite_or_none(row["z_score"]),
                }
            )
    return jsonify(
        {
            "date": str(pd.Timestamp(signals.iloc[-1]["datetime"]).date()),
            "hint": "Pick a day for detail" if selected_day is None else "",
            "candles": candles,
            "markers": markers,
            "elapsed_ms": _elapsed(started),
        }
    )


def _finite_or_none(value: Any) -> float | None:
    """Convert a numeric value to JSON-safe float or null."""
    if pd.isna(value):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


@app.get("/api/stress")
def api_stress() -> tuple[Response, int] | Response:
    """Return the requested slippage and implied-volatility sensitivity grid."""
    started = time.perf_counter()
    try:
        confidence = _parse_clamped_float(
            request.args.get("conf"),
            float(META["rf"]["tau_default"]),
            0.30,
            0.80,
            "conf",
            2,
        )
    except ValueError as exc:
        return _json_error(str(exc), started)

    default_slippage = int(META["costs"]["slippage_ticks_default"])
    slippages = sorted(
        {min(10, value) for value in (0, default_slippage, 2 * default_slippage, 5 * default_slippage)}
    )
    default_iv_pct = float(META["option_pricing"]["iv"]) * 100.0
    ivs_pct = sorted(
        {
            min(60.0, max(5.0, default_iv_pct + offset))
            for offset in (-5.0, 0.0, 5.0)
        }
    )
    stress_results = []
    signals = STARTUP_DATA["signals"]
    for iv_pct in ivs_pct:
        for slippage in slippages:
            result = run_backtest(
                signals,
                conf=confidence,
                slip_ticks=slippage,
                iv=iv_pct / 100.0,
                use_rf=True,
            )
            stress_results.append(
                {
                    "iv": iv_pct,
                    "slippage_ticks": slippage,
                    "net_pnl": result["metrics"]["Net P&L (Rs)"],
                    "sharpe": result["metrics"]["Sharpe (ann.)"],
                }
            )
    return jsonify({"grid": stress_results, "elapsed_ms": _elapsed(started)})


@app.get("/api/trades.csv")
def api_trades_csv() -> tuple[Response, int] | Response:
    """Download the selected AI trades as a CSV file."""
    started = time.perf_counter()
    try:
        confidence, slippage, volatility, selected_day = _parameters()
        signals = _select_signals(selected_day)
    except ValueError as exc:
        return _json_error(str(exc), started)
    result = run_backtest(
        signals,
        conf=confidence,
        slip_ticks=slippage,
        iv=volatility,
        use_rf=True,
    )
    output = io.StringIO()
    fields = [
        "entry_time",
        "exit_time",
        "side",
        "strike",
        "lots",
        "buy",
        "sell",
        "exit_reason_label",
        "costs",
        "pnl",
    ]
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for trade in result["trades"]:
        writer.writerow(
            {
                **trade,
                "exit_reason_label": _exit_label(
                    trade["exit_reason"],
                    str(META["square_off"]),
                ),
            }
        )
    response = Response(output.getvalue(), mimetype="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=luminx-trades.csv"
    response.headers["X-Elapsed-Ms"] = str(_elapsed(started))
    return response


@app.get("/api/live/snapshot")
def api_live_snapshot() -> tuple[Response, int] | Response:
    """Return the cached live or replay snapshot with user-selected pricing inputs."""
    started = time.perf_counter()
    try:
        confidence = _parse_clamped_float(
            request.args.get("conf"),
            float(META["rf"]["tau_default"]),
            0.30,
            0.80,
            "conf",
            2,
        )
        iv_pct = _parse_clamped_float(
            request.args.get("iv"),
            float(META["option_pricing"]["iv"]) * 100.0,
            5.0,
            60.0,
            "iv",
            1,
        )
        expiry = request.args.get("expiry")
        if expiry:
            expiry_date = date.fromisoformat(expiry)
            if expiry_date < datetime.now(IST).date():
                raise ValueError("expiry must be today or a future date.")
        else:
            expiry = None
        provider = _active_live_provider
        snapshot = get_live_snapshot(
            provider,
            confidence=confidence,
            iv=iv_pct / 100.0,
            expiry=expiry,
        )
    except ValueError as exc:
        return _json_error(str(exc), started)
    return jsonify(
        {
            "status": "ok",
            "provider": snapshot["provider"],
            "snapshot": snapshot,
            "elapsed_ms": _elapsed(started),
        }
    )


@app.post("/api/live/provider")
def api_live_provider() -> tuple[Response, int] | Response:
    """Switch the process-wide paper-demo feed provider."""
    global _active_live_provider
    started = time.perf_counter()
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or payload.get("provider") not in {"yahoo", "replay"}:
        return _json_error("provider must be 'yahoo' or 'replay'.", started)
    provider = str(payload["provider"])
    with _provider_lock:
        set_live_provider(provider)
        _active_live_provider = provider
    return jsonify({"status": "ok", "provider": provider, "elapsed_ms": _elapsed(started)})


def _refresh_paper_market(payload: dict[str, Any]) -> None:
    """Refresh the latest quote using optional controls from the live ticket."""
    confidence = _parse_clamped_float(
        str(payload["confidence"]) if payload.get("confidence") is not None else None,
        float(META["rf"]["tau_default"]),
        0.30,
        0.80,
        "confidence",
        2,
    )
    iv = _parse_clamped_float(
        str(payload["iv"]) if payload.get("iv") is not None else None,
        float(META["option_pricing"]["iv"]),
        0.05,
        0.60,
        "iv",
        4,
    )
    expiry = payload.get("expiry")
    if expiry is not None and not isinstance(expiry, str):
        raise ValueError("expiry must be an ISO calendar date.")
    get_live_snapshot(
        _active_live_provider,
        confidence=confidence,
        iv=iv,
        expiry=expiry,
    )


@app.post("/api/live/order")
def api_live_order() -> tuple[Response, int] | Response:
    """Open one simulated ATM option position after enforcing paper checks."""
    started = time.perf_counter()
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return _json_error("A JSON order object is required.", started)
    side = payload.get("side")
    lots = payload.get("lots")
    if side not in {"CE", "PE"}:
        return _json_error("side must be CE or PE.", started)
    if not isinstance(lots, int) or isinstance(lots, bool) or lots < 1:
        return _json_error("lots must be a positive integer.", started)
    try:
        _refresh_paper_market(payload)
        position = live_module.paper_desk.submit_order(str(side), lots)
    except ValueError as exc:
        return _json_error(str(exc), started)
    return jsonify(
        {
            "status": "ok",
            "position": position,
            "account": live_module.paper_desk.user.summary(),
            "elapsed_ms": _elapsed(started),
        }
    )


@app.post("/api/live/close")
def api_live_close() -> tuple[Response, int] | Response:
    """Close a simulated user position at the latest slipped paper price."""
    started = time.perf_counter()
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return _json_error("A JSON close object is required.", started)
    position_id = payload.get("position_id")
    if not isinstance(position_id, str) or not position_id:
        return _json_error("position_id is required.", started)
    try:
        _refresh_paper_market(payload)
        trade = live_module.paper_desk.close_order(position_id)
    except ValueError as exc:
        return _json_error(str(exc), started)
    return jsonify(
        {
            "status": "ok",
            "trade": trade,
            "account": live_module.paper_desk.user.summary(),
            "elapsed_ms": _elapsed(started),
        }
    )


@app.post("/api/live/settings")
def api_live_settings() -> tuple[Response, int] | Response:
    """Update the auto-exit, risk enforcement, and bot-copy settings."""
    started = time.perf_counter()
    payload = request.get_json(silent=True)
    keys = {"auto_exit", "enforce_rules", "copy_bot"}
    if not isinstance(payload, dict) or set(payload) != keys:
        return _json_error(
            "Settings must include auto_exit, enforce_rules, and copy_bot.",
            started,
        )
    if any(not isinstance(payload[key], bool) for key in keys):
        return _json_error("Paper settings must be boolean.", started)
    settings = live_module.paper_desk.set_settings(payload)
    return jsonify({"status": "ok", "settings": settings, "elapsed_ms": _elapsed(started)})


@app.post("/api/live/reset")
def api_live_reset() -> Response:
    """Reset both paper accounts without persisting any positions."""
    started = time.perf_counter()
    live_module.reset_paper_accounts()
    return jsonify(
        {
            "status": "ok",
            "user_account": live_module.paper_desk.user.summary(),
            "bot_account": live_module.paper_desk.bot.summary(),
            "elapsed_ms": _elapsed(started),
        }
    )


@app.post("/api/live/replay")
def api_live_replay() -> tuple[Response, int] | Response:
    """Control replay playback without contacting a market-data provider."""
    started = time.perf_counter()
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return _json_error("A JSON replay action is required.", started)
    action = payload.get("action")
    speed = payload.get("speed")
    if speed is not None and (
        not isinstance(speed, int) or isinstance(speed, bool)
    ):
        return _json_error("speed must be one of 1, 3, 6, or 15.", started)
    try:
        replay = live_module.control_replay(action, speed)
    except ValueError as exc:
        return _json_error(str(exc), started)
    return jsonify({"status": "ok", "replay": replay, "elapsed_ms": _elapsed(started)})


@app.get("/api/live/trades.csv")
def api_live_trades_csv() -> Response:
    """Download all closed user paper trades as a CSV attachment."""
    started = time.perf_counter()
    output = io.StringIO(newline="")
    fields = [
        "side",
        "strike",
        "lots",
        "entry_time",
        "exit_time",
        "buy",
        "sell",
        "exit_reason",
        "costs",
        "pnl",
    ]
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for trade in reversed(live_module.paper_desk.user.history):
        writer.writerow(
            {
                **trade,
                "entry_time": trade["opened_at"],
                "exit_time": trade["closed_at"],
                "exit_reason": _exit_label(
                    trade["exit_reason"], str(META["square_off"])
                ),
            }
        )
    response = Response(output.getvalue(), mimetype="text/csv")
    response.headers["Content-Disposition"] = (
        "attachment; filename=luminx-paper-trades.csv"
    )
    response.headers["X-Elapsed-Ms"] = str(_elapsed(started))
    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")))
