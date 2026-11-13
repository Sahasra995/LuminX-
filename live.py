"""Shared live/replay feed cache and server-side paper-trading state."""

from __future__ import annotations

import threading
import time
import logging
from datetime import date, datetime, time as day_time, timedelta
from functools import lru_cache
from math import floor
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import config
from backtest import _slipped_prices, _trade_costs, run_backtest
from data_loader import ensure_app_ready
from indicators import compute_feature_frame, detect_candidate_signal, feature_vector_for_row
from pricing import black_scholes_delta, black_scholes_price

IST = ZoneInfo("Asia/Kolkata")


def _meta() -> dict[str, Any]:
    """Return the process-cached strategy metadata and model card."""
    data = ensure_app_ready()
    return {"meta": data["meta"], "model_card": data["model_card"], "model": data["model"]}


class BaseProvider:
    """Common contract for providers returning normalized OHLCV bars."""

    name = "base"

    def get_bars(self, lookback_days: int) -> pd.DataFrame:
        """Return normalized timestamp, OHLC, and volume bars."""
        raise NotImplementedError


class ReplayProvider(BaseProvider):
    """Expose the complete out-of-sample feed for controlled offline playback."""

    name = "replay"
    SPEEDS = (1, 3, 6, 15)

    def __init__(self) -> None:
        """Initialize the replay cursor at the first out-of-sample bar."""
        self._lock = threading.Lock()
        self._frame = ensure_app_ready()["signals"].copy()
        self._frame["datetime"] = pd.to_datetime(self._frame["datetime"])
        self._frame = self._frame.sort_values("datetime").reset_index(drop=True)
        self._cursor = 0
        self._playing = False
        self._speed = self.SPEEDS[0]

    def get_bars(
        self, lookback_days: int = config.LIVE_LOOKBACK_DAYS
    ) -> pd.DataFrame:
        """Return all bars through the currently selected replay bar."""
        del lookback_days
        with self._lock:
            return self._frame.iloc[: self._cursor + 1].copy()

    def control(self, action: str, speed: int | None = None) -> dict[str, Any]:
        """Apply one explicit replay control action and return cursor state."""
        if speed is not None and speed not in self.SPEEDS:
            raise ValueError(f"speed must be one of {', '.join(map(str, self.SPEEDS))}.")
        with self._lock:
            if speed is not None:
                self._speed = speed
            found_signal = True
            if action == "play":
                self._playing = self._cursor < len(self._frame) - 1
            elif action == "pause":
                self._playing = False
            elif action == "step":
                self._cursor = min(self._cursor + 1, len(self._frame) - 1)
                if self._cursor >= len(self._frame) - 1:
                    self._playing = False
            elif action == "next_signal":
                candidates = self._frame["decision"].isin(["BUY_CE", "BUY_PE"]).to_numpy()
                next_indices = np.flatnonzero(candidates & (np.arange(len(candidates)) > self._cursor))
                if len(next_indices):
                    self._playing = False
                    self._cursor = int(next_indices[0])
                else:
                    self._playing = False
                    found_signal = False
            else:
                raise ValueError("action must be play, pause, step, or next_signal.")
            return self.snapshot(found_signal=found_signal)

    def advance_playback(self) -> dict[str, Any]:
        """Advance one bar only while playback is active."""
        with self._lock:
            if self._playing:
                self._cursor = min(self._cursor + 1, len(self._frame) - 1)
                if self._cursor >= len(self._frame) - 1:
                    self._playing = False
            return self.snapshot()

    def reset(self) -> None:
        """Rewind the replay for a new paper demonstration."""
        with self._lock:
            self._cursor = 0
            self._playing = False

    def snapshot(self, found_signal: bool = True) -> dict[str, Any]:
        """Return current replay controls; caller must hold the provider lock."""
        timestamp = self._frame.iloc[self._cursor]["datetime"]
        return {
            "cursor": self._cursor,
            "total_bars": len(self._frame),
            "current_time": pd.Timestamp(timestamp).isoformat(),
            "playing": self._playing,
            "speed": self._speed,
            "found_signal": found_signal,
            "at_end": self._cursor >= len(self._frame) - 1,
        }


class YahooProvider(BaseProvider):
    """Fetch delayed Bank NIFTY intraday bars from Yahoo Finance."""

    name = "yahoo"
    symbol = config.LIVE_SYMBOL

    def get_bars(
        self, lookback_days: int = config.LIVE_LOOKBACK_DAYS
    ) -> pd.DataFrame:
        """Fetch, normalize, convert to IST, and drop the forming Yahoo bar."""
        import yfinance as yf

        history = yf.Ticker(self.symbol).history(
            period=f"{lookback_days}d",
            interval=f"{config.LIVE_BAR_INTERVAL_MINUTES}m",
            auto_adjust=False,
        )
        if history.empty:
            raise ValueError("Yahoo Finance returned no Bank NIFTY bars.")
        history = history.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        timestamps = pd.DatetimeIndex(history.index)
        if timestamps.tz is None:
            timestamps = timestamps.tz_localize(IST)
        else:
            timestamps = timestamps.tz_convert(IST)
        history = history.assign(datetime=timestamps).reset_index(drop=True)
        history = history.iloc[:-1]
        columns = ["datetime", "open", "high", "low", "close", "volume"]
        missing = [column for column in columns if column not in history]
        if missing:
            raise ValueError(f"Yahoo Finance response is missing: {', '.join(missing)}")
        return history[columns].dropna(subset=["datetime", "close"]).reset_index(drop=True)


class PaperAccount:
    """Track a cash-backed paper account without broker connectivity."""

    def __init__(self, meta: dict[str, Any]) -> None:
        self.meta = meta
        self.capital = float(meta["capital"])
        self.cash = self.capital
        self.positions: dict[str, dict[str, Any]] = {}
        self.history: list[dict[str, Any]] = []
        self.trade_count: dict[str, int] = {}
        self.realized_pnl: dict[str, float] = {}

    def open_position(
        self,
        *,
        side: str,
        strike: float,
        lots: int,
        raw_premium: float,
        buy: float,
        timestamp: pd.Timestamp,
        target_pct: float,
        stop_pct: float,
        expiry: date,
        initial_time_to_expiry: float,
        source: str,
        plan_id: str | None = None,
    ) -> dict[str, Any]:
        """Debit a paper buy and add an open position after its cash check."""
        quantity = int(lots * int(self.meta["lot_size"]))
        if buy * quantity > self.cash + 1e-9:
            raise ValueError("Not enough paper cash for this order.")
        day = timestamp.date().isoformat()
        _, initial_sell = _slipped_prices(
            buy,
            raw_premium,
            self.meta["costs"]["slippage_ticks_default"],
            self.meta["costs"]["tick"],
        )
        position = {
            "position_id": uuid4().hex,
            "side": side,
            "strike": float(strike),
            "lots": int(lots),
            "quantity": quantity,
            "buy": float(buy),
            "entry_premium": float(buy),
            "entry_reference_premium": float(raw_premium),
            "mark_premium": float(raw_premium),
            "mark_exit_premium": initial_sell,
            "opened_at": timestamp.isoformat(),
            "mark_time": timestamp.isoformat(),
            "entry_day": day,
            "expiry": expiry.isoformat(),
            "initial_time_to_expiry": float(initial_time_to_expiry),
            "target_pct": float(target_pct),
            "stop_pct": float(stop_pct),
            "bars_held": 0,
            "source": source,
            "plan_id": plan_id,
            "iv": float(self.meta["option_pricing"]["iv"]),
            "unrealized_pnl": 0.0,
            "estimated_costs": 0.0,
        }
        self.cash -= buy * quantity
        self.trade_count[day] = self.trade_count.get(day, 0) + 1
        self.positions[position["position_id"]] = position
        return position

    def mark_position(
        self,
        position: dict[str, Any],
        raw_premium: float,
        timestamp: pd.Timestamp,
        *,
        new_bar: bool,
    ) -> None:
        """Mark one position and estimate its liquidation costs."""
        if new_bar and timestamp.isoformat() != position["opened_at"]:
            position["bars_held"] += 1
        _, sell = _slipped_prices(
            float(position["buy"]),
            raw_premium,
            self.meta["costs"]["slippage_ticks_default"],
            self.meta["costs"]["tick"],
        )
        cost = _trade_costs(
            float(position["buy"]),
            sell,
            int(position["quantity"]),
            self.meta["costs"],
        )
        position.update(
            {
                "mark_premium": float(raw_premium),
                "mark_exit_premium": sell,
                "mark_time": timestamp.isoformat(),
                "change_pct": (
                    raw_premium / float(position["entry_reference_premium"]) - 1.0
                )
                * 100.0
                if float(position["entry_reference_premium"])
                else 0.0,
                "unrealized_pnl": (
                    sell - float(position["buy"])
                ) * int(position["quantity"]) - cost,
                "estimated_costs": cost,
            }
        )

    def close_position(
        self,
        position_id: str,
        timestamp: pd.Timestamp,
        sell: float,
        reason: str,
    ) -> dict[str, Any]:
        """Sell one position, charge backtest costs, and realize P&L."""
        position = self.positions.get(position_id)
        if position is None:
            raise ValueError("That paper position is no longer open.")
        sell = max(float(sell), 0.05)
        cost = _trade_costs(
            float(position["buy"]),
            sell,
            int(position["quantity"]),
            self.meta["costs"],
        )
        gross = (sell - float(position["buy"])) * int(position["quantity"])
        pnl = gross - cost
        self.cash += sell * int(position["quantity"]) - cost
        closed = {
            **position,
            "sell": sell,
            "closed_at": timestamp.isoformat(),
            "exit_reason": reason,
            "gross_pnl": float(gross),
            "costs": float(cost),
            "pnl": float(pnl),
        }
        self.history.append(closed)
        day = str(position["entry_day"])
        self.realized_pnl[day] = self.realized_pnl.get(day, 0.0) + pnl
        del self.positions[position_id]
        return closed

    def summary(self) -> dict[str, Any]:
        """Summarize equity and trading quality over this in-memory account."""
        closed_pnl = sum(float(trade["pnl"]) for trade in self.history)
        open_pnl = sum(
            float(position["unrealized_pnl"])
            for position in self.positions.values()
        )
        pnl = closed_pnl + open_pnl
        winners = sum(float(trade["pnl"]) > 0 for trade in self.history)
        total_costs = sum(float(trade["costs"]) for trade in self.history) + sum(
            float(position["estimated_costs"])
            for position in self.positions.values()
        )
        return {
            "capital": self.capital,
            "available_cash": float(self.cash),
            "equity": self.capital + pnl,
            "pnl": pnl,
            "pnl_pct": pnl / self.capital * 100.0 if self.capital else 0.0,
            "trades": len(self.history) + len(self.positions),
            "win_rate": winners / len(self.history) * 100.0 if self.history else 0.0,
            "total_costs": total_costs,
        }

    def pnl_for_day(self, day: str) -> float:
        """Return realized and open paper P&L attributed to one entry day."""
        return float(self.realized_pnl.get(day, 0.0)) + sum(
            float(position["unrealized_pnl"])
            for position in self.positions.values()
            if position["entry_day"] == day
        )


@lru_cache(maxsize=12)
def _replay_bot_trades(confidence: float, iv: float) -> tuple[dict[str, Any], ...]:
    """Cache the canonical walk-forward bot fills used by research replay."""
    data = ensure_app_ready()
    result = run_backtest(
        data["signals"],
        conf=confidence,
        slip_ticks=int(data["meta"]["costs"]["slippage_ticks_default"]),
        iv=iv,
        use_rf=True,
        meta=data["meta"],
    )
    return tuple(result["trades"])


class PaperTradingDesk:
    """Coordinate isolated user and bot accounts for replay and live feeds."""

    def __init__(self) -> None:
        self.meta = _meta()["meta"]
        self._lock = threading.RLock()
        self.user = PaperAccount(self.meta)
        self.bot = PaperAccount(self.meta)
        self.settings = {
            "auto_exit": True,
            "enforce_rules": True,
            "copy_bot": False,
        }
        self.latest_market: dict[str, Any] | None = None
        self._last_replay_index = -1
        self._last_live_timestamp: str | None = None
        self._bot_entries: dict[int, dict[str, Any]] = {}
        self._bot_exits: dict[int, dict[str, Any]] = {}
        self._plan_key: tuple[float, float] | None = None
        self._copy_positions: dict[str, str] = {}
        self.last_message = ""

    def reset(self) -> None:
        """Clear both paper accounts and rewind the replay cursor."""
        with self._lock:
            self.user = PaperAccount(self.meta)
            self.bot = PaperAccount(self.meta)
            self.latest_market = None
            self._last_replay_index = -1
            self._last_live_timestamp = None
            self._copy_positions.clear()
            self.last_message = ""

    def set_settings(self, settings: dict[str, bool]) -> dict[str, bool]:
        """Replace the three user-controlled paper trading settings."""
        with self._lock:
            self.settings = dict(settings)
            return dict(self.settings)

    def _ensure_replay_plan(self, confidence: float, iv: float) -> None:
        key = (float(confidence), float(iv))
        if self._plan_key == key:
            return
        trades = _replay_bot_trades(*key)
        self._bot_entries = {int(trade["row_index"]): trade for trade in trades}
        self._bot_exits = {int(trade["exit_index"]): trade for trade in trades}
        self._plan_key = key

    def _risk_block(
        self,
        account: PaperAccount,
        timestamp: pd.Timestamp,
        *,
        enforce: bool | None = None,
    ) -> str | None:
        """Return the first enabled entry constraint that blocks a paper buy."""
        if not (self.settings["enforce_rules"] if enforce is None else enforce):
            return None
        time_value = timestamp.strftime("%H:%M")
        window_start, window_end = map(str, self.meta["entry_window"])
        square_off = str(self.meta["square_off"])
        if time_value >= square_off:
            return f"New orders stop at {square_off}."
        if time_value < window_start or time_value > window_end:
            return f"Orders are allowed only from {window_start} to {window_end}."
        day = timestamp.date().isoformat()
        if account.positions:
            return "Close your open paper position before entering another."
        if account.trade_count.get(day, 0) >= int(self.meta["max_trades_per_day"]):
            return f"Maximum {self.meta['max_trades_per_day']} trades for this day reached."
        loss_cap = float(self.meta["daily_loss_cap_rs"])
        if account.realized_pnl.get(day, 0.0) <= -loss_cap:
            return f"The daily loss cap of Rs {loss_cap:,.0f} has been reached."
        return None

    def _premium(
        self,
        spot: float,
        strike: float,
        side: str,
        iv: float,
        time_to_expiry: float,
    ) -> float:
        """Price a paper option using the shared strategy pricing metadata."""
        pricing = self.meta["option_pricing"]
        return black_scholes_price(
            spot,
            strike,
            time_to_expiry,
            side,
            iv=iv,
            rate=float(pricing["rate"]),
            meta=self.meta,
        )

    def _position_premium(
        self,
        position: dict[str, Any],
        timestamp: pd.Timestamp,
        spot: float,
        *,
        new_bar: bool = False,
    ) -> float:
        """Mark a position using its original expiry horizon and elapsed bars."""
        minutes_per_year = 365.0 * 24.0 * 60.0
        bars_held = int(position["bars_held"])
        if new_bar and timestamp.isoformat() != position["opened_at"]:
            bars_held += 1
        elapsed = (
            bars_held
            * config.LIVE_BAR_INTERVAL_MINUTES
            / minutes_per_year
        )
        time_to_expiry = max(
            float(position["initial_time_to_expiry"]) - elapsed, 1e-6
        )
        return self._premium(
            spot,
            float(position["strike"]),
            str(position["side"]),
            float(position.get("iv", self.meta["option_pricing"]["iv"])),
            time_to_expiry,
        )

    def _advance_positions(
        self,
        account: PaperAccount,
        timestamp: pd.Timestamp,
        spot: float,
        iv: float,
        *,
        new_bar: bool,
        force_session_end: bool,
        auto_exit: bool | None = None,
    ) -> None:
        """Update marks and apply auto, square-off, and session-end exits."""
        square_off = str(self.meta["square_off"])
        for position_id, position in list(account.positions.items()):
            entry_day = str(position["entry_day"])
            if timestamp.date().isoformat() != entry_day:
                exit_timestamp = pd.Timestamp(position["mark_time"])
                account.close_position(
                    position_id,
                    exit_timestamp,
                    float(position["mark_exit_premium"]),
                    "session_end",
                )
                continue
            if new_bar:
                position["iv"] = iv
            raw = self._position_premium(
                position, timestamp, spot, new_bar=new_bar
            )
            account.mark_position(position, raw, timestamp, new_bar=new_bar)
            is_copy = position.get("source") == "copy_bot"
            return_fraction = (
                raw / float(position["entry_reference_premium"]) - 1.0
                if float(position["entry_reference_premium"])
                else 0.0
            )
            reason = None
            exits_enabled = (
                self.settings["auto_exit"] if auto_exit is None else auto_exit
            )
            if exits_enabled and not is_copy:
                if return_fraction >= float(position["target_pct"]):
                    reason = "target"
                elif return_fraction <= -float(position["stop_pct"]):
                    reason = "stop"
                elif int(position["bars_held"]) >= int(self.meta["max_hold_bars"]):
                    reason = "time"
            if reason is None and timestamp.strftime("%H:%M") >= square_off:
                reason = "square_off"
            if reason is None and force_session_end:
                reason = "session_end"
            if reason:
                account.close_position(
                    position_id,
                    timestamp,
                    float(position["mark_exit_premium"]),
                    reason,
                )
                copied_from = position.get("plan_id")
                if account is self.user and copied_from:
                    self._copy_positions.pop(str(copied_from), None)

    def _open_bot_plan(
        self,
        trade: dict[str, Any],
        row: pd.Series,
        expiry: date,
        iv: float,
    ) -> None:
        """Open one pre-priced research bot trade and optionally copy it."""
        timestamp = pd.Timestamp(trade["entry_time"])
        key = f"{trade['row_index']}:{trade['exit_index']}"
        slip = (
            int(self.meta["costs"]["slippage_ticks_default"])
            * float(self.meta["costs"]["tick"])
        )
        raw_entry = float(trade["buy"]) - slip
        initial_tte = float(self.meta["option_pricing"]["days_to_expiry"]) / 365.0
        args = {
            "side": str(trade["side"]),
            "strike": float(trade["strike"]),
            "lots": int(trade["lots"]),
            "raw_premium": raw_entry,
            "buy": float(trade["buy"]),
            "timestamp": timestamp,
            "target_pct": float(row.get("target_pct", self.meta["target_pct"])),
            "stop_pct": float(row.get("stop_pct", self.meta["stop_pct"])),
            "expiry": expiry,
            "initial_time_to_expiry": initial_tte,
            "source": "bot",
            "plan_id": key,
        }
        bot_position = self.bot.open_position(**args)
        bot_position["iv"] = iv
        if not self.settings["copy_bot"]:
            return
        block = self._risk_block(self.user, timestamp)
        if block:
            self.last_message = f"Bot trade was not copied: {block}"
            return
        try:
            copied = self.user.open_position(**{**args, "source": "copy_bot"})
        except ValueError as exc:
            self.last_message = f"Bot trade was not copied: {exc}"
            return
        copied["iv"] = iv
        self._copy_positions[key] = copied["position_id"]

    def _close_bot_plan(self, trade: dict[str, Any]) -> None:
        """Close a replay bot fill at the backtest simulator's exact sell."""
        key = f"{trade['row_index']}:{trade['exit_index']}"
        timestamp = pd.Timestamp(trade["exit_time"])
        bot_position = next(
            (
                position
                for position in self.bot.positions.values()
                if position.get("plan_id") == key
            ),
            None,
        )
        if bot_position:
            self.bot.close_position(
                bot_position["position_id"],
                timestamp,
                float(trade["sell"]),
                str(trade["exit_reason"]),
            )
        user_position_id = self._copy_positions.pop(key, None)
        if user_position_id and user_position_id in self.user.positions:
            self.user.close_position(
                user_position_id,
                timestamp,
                float(trade["sell"]),
                str(trade["exit_reason"]),
            )

    def _open_live_bot(
        self,
        timestamp: pd.Timestamp,
        spot: float,
        side: str,
        iv: float,
        expiry: date,
    ) -> None:
        """Open one approved live bot paper trade and mirror it when requested."""
        if self.bot.positions or self._risk_block(
            self.bot, timestamp, enforce=True
        ):
            return
        pricing = self.meta["option_pricing"]
        strike_step = float(pricing["strike_step"])
        strike = round(spot / strike_step) * strike_step
        initial_tte = _time_to_expiry(timestamp.to_pydatetime(), expiry)
        raw = self._premium(spot, strike, side, iv, initial_tte)
        slip = int(self.meta["costs"]["slippage_ticks_default"]) * float(
            self.meta["costs"]["tick"]
        )
        buy = raw + slip
        lots = max(
            1,
            floor(
                float(self.meta["risk_per_trade"])
                * float(self.meta["capital"])
                / (
                    float(self.meta["stop_pct"])
                    * buy
                    * int(self.meta["lot_size"])
                ),
            ),
        )
        args = {
            "side": side,
            "strike": strike,
            "lots": lots,
            "raw_premium": raw,
            "buy": buy,
            "timestamp": timestamp,
            "target_pct": float(self.meta["target_pct"]),
            "stop_pct": float(self.meta["stop_pct"]),
            "expiry": expiry,
            "initial_time_to_expiry": initial_tte,
            "source": "bot",
        }
        try:
            bot_position = self.bot.open_position(**args)
        except ValueError:
            return
        bot_position["iv"] = iv
        if self.settings["copy_bot"]:
            block = self._risk_block(self.user, timestamp)
            if block:
                self.last_message = f"Bot trade was not copied: {block}"
                return
            try:
                copied = self.user.open_position(**{**args, "source": "copy_bot"})
            except ValueError as exc:
                self.last_message = f"Bot trade was not copied: {exc}"
                return
            copied["iv"] = iv
            self._copy_positions[bot_position["position_id"]] = copied["position_id"]

    def _process_bar(
        self,
        timestamp: pd.Timestamp,
        spot: float,
        decision: str,
        probability: float | None,
        confidence: float,
        iv: float,
        expiry: date,
        *,
        new_bar: bool,
        force_session_end: bool,
        replay: bool,
        replay_trade: dict[str, Any] | None = None,
        replay_exit: dict[str, Any] | None = None,
    ) -> None:
        """Advance account marks and handle bot entries and exits for one bar."""
        self._advance_positions(
            self.user,
            timestamp,
            spot,
            iv,
            new_bar=new_bar,
            force_session_end=force_session_end,
            auto_exit=self.settings["auto_exit"],
        )
        if replay:
            for position in self.bot.positions.values():
                raw = self._position_premium(
                    position, timestamp, spot, new_bar=new_bar
                )
                self.bot.mark_position(position, raw, timestamp, new_bar=new_bar)
            if replay_trade is not None:
                self._open_bot_plan(
                    replay_trade, replay_trade["source_row"], expiry, iv
                )
            if replay_exit is not None:
                self._close_bot_plan(replay_exit)
            return

        self._advance_positions(
            self.bot,
            timestamp,
            spot,
            iv,
            new_bar=new_bar,
            force_session_end=force_session_end,
            auto_exit=True,
        )
        if (
            decision in {"BUY_CE", "BUY_PE"}
            and probability is not None
            and probability >= confidence
            and not self.bot.positions
        ):
            self._open_live_bot(
                timestamp,
                spot,
                "CE" if decision == "BUY_CE" else "PE",
                iv,
                expiry,
            )
        for bot_id, user_id in list(self._copy_positions.items()):
            if bot_id in self.bot.positions or user_id not in self.user.positions:
                continue
            copied = self.user.positions[user_id]
            self.user.close_position(
                user_id,
                timestamp,
                float(copied["mark_exit_premium"]),
                "bot_exit",
            )
            self._copy_positions.pop(bot_id, None)

    def process_snapshot(
        self,
        bars: pd.DataFrame,
        decisions: list[str],
        probabilities: list[float | None],
        *,
        provider: str,
        confidence: float,
        iv: float,
        expiry: date,
    ) -> None:
        """Apply every unseen bar, even when replay controls skip ahead."""
        if bars.empty:
            return
        with self._lock:
            timestamps = pd.to_datetime(bars["datetime"], errors="coerce")
            self.latest_market = {
                "timestamp": pd.Timestamp(timestamps.iloc[-1]),
                "spot": float(bars.iloc[-1]["close"]),
                "iv": float(iv),
                "expiry": expiry,
                "option_chain": None,
            }
            if provider == "replay":
                self._ensure_replay_plan(confidence, iv)
                end_index = len(bars) - 1
                for index in range(self._last_replay_index + 1, end_index + 1):
                    row = bars.iloc[index]
                    timestamp = pd.Timestamp(row["datetime"])
                    full = _replay_provider._frame
                    session_end = (
                        index >= len(full) - 1
                        or pd.Timestamp(full.iloc[index + 1]["datetime"]).date()
                        != timestamp.date()
                    )
                    trade = self._bot_entries.get(index)
                    if trade is not None:
                        trade = {**trade, "source_row": row}
                    self._process_bar(
                        timestamp,
                        float(row["close"]),
                        decisions[index],
                        probabilities[index],
                        confidence,
                        iv,
                        expiry,
                        new_bar=True,
                        force_session_end=session_end,
                        replay=True,
                        replay_trade=trade,
                        replay_exit=self._bot_exits.get(index),
                    )
                self._last_replay_index = max(self._last_replay_index, end_index)
            else:
                last_seen = (
                    pd.Timestamp(self._last_live_timestamp)
                    if self._last_live_timestamp
                    else None
                )
                for index, row in bars.iterrows():
                    timestamp = pd.Timestamp(row["datetime"])
                    if last_seen is not None and timestamp <= last_seen:
                        continue
                    next_timestamp = (
                        pd.Timestamp(bars.iloc[index + 1]["datetime"])
                        if index + 1 < len(bars)
                        else None
                    )
                    session_end = (
                        next_timestamp is not None
                        and next_timestamp.date() != timestamp.date()
                    )
                    self._process_bar(
                        timestamp,
                        float(row["close"]),
                        decisions[index],
                        probabilities[index],
                        confidence,
                        iv,
                        expiry,
                        new_bar=True,
                        force_session_end=session_end,
                        replay=False,
                    )
                    self._last_live_timestamp = timestamp.isoformat()
            self._mark_latest_positions()

    def _mark_latest_positions(self) -> None:
        """Refresh open position marks against the latest known market state."""
        if self.latest_market is None:
            return
        market = self.latest_market
        timestamp = pd.Timestamp(market["timestamp"])
        for account in (self.user, self.bot):
            for position in account.positions.values():
                raw = self._position_premium(
                    position, timestamp, float(market["spot"])
                )
                account.mark_position(position, raw, timestamp, new_bar=False)

    def set_market_option_chain(self, option_chain: list[dict[str, Any]]) -> None:
        """Store current ATM option prices for immediate order-ticket use."""
        with self._lock:
            if self.latest_market is not None:
                self.latest_market["option_chain"] = option_chain

    def risk_lots(self, side: str) -> int:
        """Return the metadata-driven one-percent risk lot size for one side."""
        with self._lock:
            if not self.latest_market or not self.latest_market.get("option_chain"):
                return 1
            row = next(
                item for item in self.latest_market["option_chain"] if item["atm"]
            )
            raw = float(row[side.lower()]["premium"])
            buy = raw + (
                int(self.meta["costs"]["slippage_ticks_default"])
                * float(self.meta["costs"]["tick"])
            )
            return max(
                1,
                floor(
                    float(self.meta["risk_per_trade"])
                    * float(self.meta["capital"])
                    / (
                        float(self.meta["stop_pct"])
                        * buy
                        * int(self.meta["lot_size"])
                    )
                ),
            )

    def submit_order(self, side: str, lots: int) -> dict[str, Any]:
        """Open a cash-checked, risk-checked user paper position."""
        with self._lock:
            if side not in {"CE", "PE"}:
                raise ValueError("side must be CE or PE.")
            if not isinstance(lots, int) or isinstance(lots, bool) or lots < 1:
                raise ValueError("lots must be a positive integer.")
            market = self.latest_market
            if not market or not market.get("option_chain"):
                raise ValueError("No current option price is available yet.")
            timestamp = pd.Timestamp(market["timestamp"])
            blocked = self._risk_block(self.user, timestamp)
            if blocked:
                raise ValueError(blocked)
            row = next(
                item for item in market["option_chain"] if item["atm"]
            )
            raw = float(row[side.lower()]["premium"])
            strike = float(row["strike"])
            slip = (
                int(self.meta["costs"]["slippage_ticks_default"])
                * float(self.meta["costs"]["tick"])
            )
            buy = raw + slip
            initial_tte = _time_to_expiry(
                timestamp.to_pydatetime(), market["expiry"]
            )
            position = self.user.open_position(
                side=side,
                strike=strike,
                lots=lots,
                raw_premium=raw,
                buy=buy,
                timestamp=timestamp,
                target_pct=float(self.meta["target_pct"]),
                stop_pct=float(self.meta["stop_pct"]),
                expiry=market["expiry"],
                initial_time_to_expiry=initial_tte,
                source="manual",
            )
            position["iv"] = float(market["iv"])
            self.last_message = ""
            return position

    def close_order(self, position_id: str) -> dict[str, Any]:
        """Close a user position at the latest repriced premium less slippage."""
        with self._lock:
            if not isinstance(position_id, str) or not position_id:
                raise ValueError("position_id is required.")
            position = self.user.positions.get(position_id)
            if position is None:
                raise ValueError("That paper position is no longer open.")
            if not self.latest_market:
                raise ValueError("No current market price is available yet.")
            market = self.latest_market
            timestamp = pd.Timestamp(market["timestamp"])
            raw = self._position_premium(
                position, timestamp, float(market["spot"])
            )
            _, sell = _slipped_prices(
                float(position["buy"]),
                raw,
                self.meta["costs"]["slippage_ticks_default"],
                self.meta["costs"]["tick"],
            )
            trade = self.user.close_position(
                position_id, timestamp, sell, "manual"
            )
            self.last_message = ""
            return trade

    def snapshot(self, replay: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return account summaries, trades, settings, and replay state."""
        with self._lock:
            self._mark_latest_positions()
            market_day = (
                pd.Timestamp(self.latest_market["timestamp"]).date().isoformat()
                if self.latest_market
                else datetime.now(IST).date().isoformat()
            )
            user_account = self.user.summary()
            user_account["pnl_today"] = self.user.pnl_for_day(market_day)
            return {
                "user_positions": list(self.user.positions.values()),
                "user_closed_trades": list(reversed(self.user.history)),
                "user_account": user_account,
                "bot_positions": list(self.bot.positions.values()),
                "bot_closed_trades": list(reversed(self.bot.history)),
                "bot_account": self.bot.summary(),
                "paper_settings": dict(self.settings),
                "paper_message": self.last_message,
                "risk_lots": {side: self.risk_lots(side) for side in ("CE", "PE")},
                "replay": replay,
            }


def _time_to_expiry(now: datetime, expiry: date) -> float:
    """Calculate years until the requested expiry's 15:30 IST close."""
    current = now.replace(tzinfo=IST) if now.tzinfo is None else now.astimezone(IST)
    expiry_close = datetime.combine(expiry, day_time(15, 30), tzinfo=IST)
    return max((expiry_close - current).total_seconds() / (365.0 * 24.0 * 60.0 * 60.0), 1e-6)


def _next_tuesday(today: date) -> date:
    """Return the next Tuesday, including today when today is Tuesday."""
    days_until_tuesday = (1 - today.weekday()) % 7
    return today + timedelta(days=days_until_tuesday)


def _default_expiry(now: datetime) -> date:
    """Choose the next Tuesday in the market's local timezone."""
    return _next_tuesday(now.astimezone(IST).date())


def default_expiry_date() -> str:
    """Return the default live expiry date as an ISO string."""
    return _default_expiry(datetime.now(IST)).isoformat()


def _market_is_open(now: datetime) -> bool:
    """Check weekday and regular NSE index market hours."""
    local = now.astimezone(IST)
    return local.weekday() < 5 and day_time(9, 15) <= local.time() <= day_time(15, 30)


def _provider_instance(name: str) -> BaseProvider:
    """Create the selected provider instance."""
    if name == "yahoo":
        return YahooProvider()
    if name == "replay":
        return _replay_provider
    raise ValueError("provider must be 'yahoo' or 'replay'.")


_lock = threading.RLock()
_active_provider = config.LIVE_PROVIDER
_last_fetch_at = 0.0
_last_vix_at = 0.0
_last_vix_value: float | None = None
_last_fetch_by_provider: dict[str, float] = {}
_last_bars_by_provider: dict[str, pd.DataFrame] = {}
_last_errors_by_provider: dict[str, str | None] = {}
_last_bars: pd.DataFrame | None = None
_last_fetch_error: str | None = None
_replay_provider = ReplayProvider()
paper_desk = PaperTradingDesk()


def reset_paper_accounts() -> None:
    """Reset both in-memory paper accounts and rewind the replay feed."""
    paper_desk.reset()
    _replay_provider.reset()


def control_replay(action: str, speed: int | None = None) -> dict[str, Any]:
    """Apply a replay action only while replay is the active data provider."""
    if _active_provider != "replay":
        raise ValueError("Replay controls are available only with the replay provider.")
    return _replay_provider.control(action, speed)


def set_live_provider(provider: str) -> None:
    """Switch feed source and clear the provider-specific data cache."""
    global _active_provider, _last_fetch_at, _last_fetch_error, _last_bars
    if provider not in {"yahoo", "replay"}:
        raise ValueError("provider must be 'yahoo' or 'replay'.")
    with _lock:
        _active_provider = provider
        _last_fetch_at = 0.0
        _last_fetch_error = None
        _last_bars = _last_bars_by_provider.get(provider)
        if provider == "replay":
            _replay_provider.control("pause")


def _fetch_bars(provider_name: str) -> tuple[pd.DataFrame | None, str | None, float]:
    """Fetch at the provider's allowed rate and retain the last successful bars."""
    global _last_bars, _last_fetch_at, _last_fetch_error
    now = time.monotonic()
    interval = 0.0 if provider_name == "replay" else config.LIVE_REFRESH_SECONDS
    with _lock:
        previous_fetch = _last_fetch_by_provider.get(provider_name, 0.0)
        if provider_name != "replay" and previous_fetch and now - previous_fetch < interval:
            return (
                _last_bars_by_provider.get(provider_name),
                _last_errors_by_provider.get(provider_name),
                0.0,
            )
        provider = _provider_instance(provider_name)
        fetch_started = time.perf_counter()
        try:
            bars = provider.get_bars(config.LIVE_LOOKBACK_DAYS)
            if bars.empty:
                raise ValueError("Market feed returned no closed bars.")
            _last_bars = bars.copy()
            _last_bars_by_provider[provider_name] = bars.copy()
            _last_fetch_error = None
            _last_errors_by_provider[provider_name] = None
        except Exception as exc:
            _last_fetch_error = str(exc)
            _last_errors_by_provider[provider_name] = _last_fetch_error
            _last_bars = _last_bars_by_provider.get(provider_name)
        finally:
            _last_fetch_at = time.monotonic()
            _last_fetch_by_provider[provider_name] = _last_fetch_at
        return _last_bars, _last_fetch_error, (time.perf_counter() - fetch_started) * 1000.0


def _vix_value(provider: str) -> float | None:
    """Try a Yahoo India VIX observation when the feed source requests it."""
    global _last_vix_at, _last_vix_value
    if config.IV_SOURCE != "vix" or provider != "yahoo":
        return None
    with _lock:
        if _last_vix_at and time.monotonic() - _last_vix_at < config.LIVE_REFRESH_SECONDS:
            return _last_vix_value
    try:
        import yfinance as yf

        data = yf.Ticker(config.VIX_SYMBOL).history(period="1d", interval="5m")
        if data.empty:
            _last_vix_at = time.monotonic()
            _last_vix_value = None
            return None
        value = float(data["Close"].dropna().iloc[-1]) / 100.0
        _last_vix_at = time.monotonic()
        _last_vix_value = value if value > 0 else None
        return _last_vix_value
    except Exception as exc:
        logging.warning("India VIX unavailable; using configured IV fallback: %s", exc)
        _last_vix_at = time.monotonic()
        _last_vix_value = None
        return None


def _option_chain(
    spot: float,
    iv: float,
    expiry: date,
    now: datetime,
    meta: dict[str, Any],
) -> list[dict[str, Any]]:
    """Re-price CE and PE options at ATM and the neighboring strikes."""
    pricing = meta["option_pricing"]
    strike_step = float(pricing["strike_step"])
    atm = round(spot / strike_step) * strike_step
    time_to_expiry = _time_to_expiry(now, expiry)
    chain = []
    for offset in range(-2, 3):
        strike = atm + offset * strike_step
        row: dict[str, Any] = {"strike": float(strike), "atm": offset == 0}
        for kind in ("CE", "PE"):
            row[kind.lower()] = {
                "premium": black_scholes_price(
                    spot,
                    strike,
                    time_to_expiry,
                    kind,
                    iv=iv,
                    rate=float(pricing["rate"]),
                    meta=meta,
                ),
                "delta": black_scholes_delta(
                    spot,
                    strike,
                    time_to_expiry,
                    kind,
                    iv=iv,
                    rate=float(pricing["rate"]),
                    meta=meta,
                ),
            }
        chain.append(row)
    return chain


def _snapshot_badges(
    provider: str,
    volume_unavailable: bool,
    market_open: bool,
    fetch_error: str | None,
) -> list[dict[str, str]]:
    """Build user-facing provider, market, volume, and feed-health badges."""
    badges = []
    if provider == "yahoo":
        badges.append({"kind": "status", "label": "Data delayed ~15 min"})
    else:
        badges.append({"kind": "status", "label": "Replay"})
    if volume_unavailable:
        badges.append(
            {
                "kind": "status",
                "label": (
                    "Index volume unavailable — VWAP uses equal weights; "
                    "model confidence less reliable"
                ),
            }
        )
    if not market_open:
        badges.append({"kind": "status", "label": "Market closed — showing last session"})
    if fetch_error:
        badges.append({"kind": "error", "label": "Feed unavailable, retrying"})
    return badges


def get_live_snapshot(
    provider_name: str | None = None,
    confidence: float | None = None,
    iv: float | None = None,
    expiry: str | None = None,
) -> dict[str, Any]:
    """Build the live snapshot and advance isolated paper accounts."""
    started = time.perf_counter()
    data = _meta()
    meta = data["meta"]
    provider = provider_name or _active_provider
    if provider not in {"yahoo", "replay"}:
        raise ValueError("provider must be 'yahoo' or 'replay'.")
    now = datetime.now(IST)
    selected_expiry = date.fromisoformat(expiry) if expiry else _default_expiry(now)
    if selected_expiry < now.date():
        raise ValueError("expiry must be today or a future date.")
    confidence_threshold = float(
        meta["rf"]["tau_default"] if confidence is None else confidence
    )
    requested_iv = float(meta["option_pricing"]["iv"] if iv is None else iv)
    fetch_started = time.perf_counter()
    bars, fetch_error, fetch_ms = _fetch_bars(provider)
    if bars is None or bars.empty:
        return {
            "provider": provider,
            "symbol": config.LIVE_SYMBOL,
            "bars": [],
            "last_bar_time": None,
            "data_age_seconds": (
                max(0.0, time.monotonic() - _last_fetch_at)
                if _last_fetch_at
                else None
            ),
            "signal": {
                "candidate": "HOLD",
                "state": "Waiting for pullback",
                "probability": None,
                "approved": False,
            },
            "option_chain": [],
            **paper_desk.snapshot(
                _replay_provider.snapshot() if provider == "replay" else None
            ),
            "badges": _snapshot_badges(
                provider, False, _market_is_open(now), fetch_error
            ),
            "fetch_ms": fetch_ms,
            "scoring_ms": 0.0,
            "snapshot_ms": (time.perf_counter() - started) * 1000.0,
            "error": fetch_error or "No market bars are available.",
        }

    normalized = bars.copy()
    normalized["datetime"] = pd.to_datetime(normalized["datetime"], errors="coerce")
    normalized = (
        normalized.dropna(subset=["datetime", "close"])
        .sort_values("datetime")
        .reset_index(drop=True)
    )
    if normalized.empty:
        raise ValueError("Market feed contains no valid bars.")

    feature_frame = compute_feature_frame(normalized, meta, data["model_card"])
    detected = detect_candidate_signal(normalized, meta, data["model_card"])
    if provider == "replay" and "decision" in normalized:
        decisions = normalized["decision"].fillna("HOLD").astype(str).tolist()
    else:
        decisions = detected.astype(str).tolist()
    feature_frame["decision"] = decisions
    latest_index = len(feature_frame) - 1

    source_iv = _vix_value(provider)
    if config.IV_SOURCE == "vix":
        iv_used = (
            source_iv
            if source_iv is not None
            else float(meta["option_pricing"]["iv"])
        )
        selected_iv_source = "vix" if source_iv is not None else "meta fallback"
    else:
        iv_used = requested_iv
        selected_iv_source = "slider"

    scoring_started = time.perf_counter()
    probabilities: list[float | None] = [None] * len(normalized)
    for index, decision in enumerate(decisions):
        if decision not in {"BUY_CE", "BUY_PE"}:
            continue
        if provider == "replay" and "rf_confidence" in normalized:
            value = pd.to_numeric(
                normalized.iloc[index]["rf_confidence"], errors="coerce"
            )
            probabilities[index] = (
                float(value) if pd.notna(value) and np.isfinite(value) else None
            )
            continue
        feature_vector = feature_vector_for_row(
            feature_frame,
            index,
            data["model_card"]["feature_order"],
            meta,
            data["model_card"],
        )
        model_input = pd.DataFrame(
            [[feature_vector[name] for name in data["model_card"]["feature_order"]]],
            columns=data["model_card"]["feature_order"],
        )
        probabilities[index] = float(
            data["model"].predict_proba(model_input)[0, 1]
        )
    scoring_ms = (time.perf_counter() - scoring_started) * 1000.0
    decisions = [str(value) for value in decisions]
    latest = feature_frame.iloc[latest_index]
    timestamp = pd.Timestamp(latest["datetime"])
    spot = float(latest["close"])
    decision = decisions[latest_index]
    probability = probabilities[latest_index]
    approved = (
        probability is not None and probability >= confidence_threshold
    )
    volume_unavailable = bool(
        normalized.loc[
            normalized["datetime"].dt.date == timestamp.date(), "volume"
        ]
        .fillna(0)
        .sum()
        <= 0
    )

    paper_desk.process_snapshot(
        normalized,
        decisions,
        probabilities,
        provider=provider,
        confidence=confidence_threshold,
        iv=iv_used,
        expiry=selected_expiry,
    )
    local_now = datetime.now(IST)
    age_seconds = max(
        0.0,
        (local_now - timestamp.to_pydatetime().astimezone(IST)).total_seconds(),
    )
    chart_bars = feature_frame.loc[
        feature_frame["datetime"].dt.date == timestamp.date()
    ]
    rows = []
    for index, bar in chart_bars.iterrows():
        rows.append(
            {
                "datetime": pd.Timestamp(bar["datetime"]).isoformat(),
                "open": float(bar["open"]),
                "high": float(bar["high"]),
                "low": float(bar["low"]),
                "close": float(bar["close"]),
                "vwap": _finite(bar["vwap"]),
                "bb_upper": _finite(bar["bb_upper"]),
                "bb_lower": _finite(bar["bb_lower"]),
                "z_score": _finite(bar["z"]),
                "decision": str(bar["decision"]),
                "confidence": probabilities[index],
            }
        )
    previous_close = float(chart_bars.iloc[0]["close"])
    change_pct = (
        (spot - previous_close) / previous_close * 100.0
        if previous_close
        else 0.0
    )
    if decision == "HOLD":
        signal_state = "Waiting for pullback"
    elif approved:
        signal_state = f"{decision.replace('_', ' ')} signal — AI {probability:.2f} ✓"
    elif probability is not None:
        signal_state = f"Rejected — AI {probability:.2f}"
    else:
        signal_state = "Signal probability unavailable"
    pricing_time = (
        timestamp.to_pydatetime() if provider == "replay" else local_now
    )
    option_chain = _option_chain(
        spot,
        iv_used,
        selected_expiry,
        pricing_time,
        meta,
    )
    paper_desk.set_market_option_chain(option_chain)
    paper = paper_desk.snapshot(
        _replay_provider.snapshot() if provider == "replay" else None
    )
    return {
        "provider": provider,
        "symbol": config.LIVE_SYMBOL,
        "bar_interval_minutes": config.LIVE_BAR_INTERVAL_MINUTES,
        "last_bar_time": timestamp.isoformat(),
        "data_age_seconds": age_seconds,
        "spot": spot,
        "change_today_pct": change_pct,
        "vwap": _finite(latest["vwap"]),
        "z_score": _finite(latest["z"]),
        "bb_mid": _finite(latest["bb_mid"]),
        "bb_upper": _finite(latest["bb_upper"]),
        "bb_lower": _finite(latest["bb_lower"]),
        "signal": {
            "candidate": decision,
            "state": signal_state,
            "probability": probability,
            "approved": bool(approved),
        },
        "option_chain": option_chain,
        "expiry": selected_expiry.isoformat(),
        "iv_used": iv_used,
        "iv_source": selected_iv_source,
        "volume_unavailable": volume_unavailable,
        "market_open": _market_is_open(local_now),
        "badges": _snapshot_badges(
            provider,
            volume_unavailable,
            _market_is_open(local_now),
            fetch_error,
        ),
        "bars": rows,
        **paper,
        "paper": {
            "open_position": paper["bot_positions"][0] if paper["bot_positions"] else None,
            "closed_trades": paper["bot_closed_trades"],
            "pnl_today": paper["bot_account"]["pnl"],
        },
        "fetch_ms": fetch_ms,
        "scoring_ms": scoring_ms,
        "snapshot_ms": (time.perf_counter() - started) * 1000.0,
        "feed_age_ms": (time.perf_counter() - fetch_started) * 1000.0,
    }


def _finite(value: Any) -> float | None:
    """Convert optional indicator values to JSON-safe finite floats."""
    if value is None or pd.isna(value):
        return None
    number = float(value)
    return number if np.isfinite(number) else None


