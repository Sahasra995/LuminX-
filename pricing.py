"""Black-Scholes option pricing and deterministic trade re-pricing."""

from __future__ import annotations

from math import exp, log, sqrt
from typing import Any, Mapping

import pandas as pd
from scipy.stats import norm

from data_loader import ensure_app_ready


def _pricing_settings(
    meta: Mapping[str, Any] | None = None,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Resolve option-pricing and risk settings from supplied metadata or startup cache."""
    settings = meta or ensure_app_ready()["meta"]
    return settings["option_pricing"], settings


def _d1_d2(
    spot: float,
    strike: float,
    time_to_expiry_years: float,
    iv: float,
    rate: float,
) -> tuple[float, float, float]:
    """Calculate the Black-Scholes terms after validating the contract inputs."""
    if spot <= 0 or strike <= 0 or iv <= 0:
        raise ValueError("Spot, strike, and implied volatility must be positive.")
    time = max(float(time_to_expiry_years), 1e-6)
    sigma_root_time = iv * sqrt(time)
    d1 = (
        log(spot / strike) + (rate + iv * iv / 2.0) * time
    ) / sigma_root_time
    return d1, d1 - sigma_root_time, time


def black_scholes_price(
    spot: float,
    strike: float,
    time_to_expiry_years: float,
    kind: str,
    iv: float | None = None,
    rate: float | None = None,
    meta: Mapping[str, Any] | None = None,
) -> float:
    """Price a European call or put with the shared Black-Scholes formula."""
    pricing, _ = _pricing_settings(meta)
    sigma = float(pricing["iv"] if iv is None else iv)
    interest_rate = float(pricing["rate"] if rate is None else rate)
    d1, d2, time = _d1_d2(spot, strike, time_to_expiry_years, sigma, interest_rate)
    option_kind = kind.upper()
    if option_kind == "CE":
        return float(
            spot * norm.cdf(d1)
            - strike * exp(-interest_rate * time) * norm.cdf(d2)
        )
    if option_kind == "PE":
        return float(
            strike * exp(-interest_rate * time) * norm.cdf(-d2)
            - spot * norm.cdf(-d1)
        )
    raise ValueError(f"Unsupported option kind: {kind!r}")


def black_scholes_delta(
    spot: float,
    strike: float,
    time_to_expiry_years: float,
    kind: str,
    iv: float | None = None,
    rate: float | None = None,
    meta: Mapping[str, Any] | None = None,
) -> float:
    """Return the shared Black-Scholes delta for a European call or put."""
    pricing, _ = _pricing_settings(meta)
    sigma = float(pricing["iv"] if iv is None else iv)
    interest_rate = float(pricing["rate"] if rate is None else rate)
    d1, _, _ = _d1_d2(spot, strike, time_to_expiry_years, sigma, interest_rate)
    option_kind = kind.upper()
    if option_kind == "CE":
        return float(norm.cdf(d1))
    if option_kind == "PE":
        return float(norm.cdf(d1) - 1.0)
    raise ValueError(f"Unsupported option kind: {kind!r}")


def reprice_trade(
    frame: pd.DataFrame,
    row_index: int,
    target_pct: float | None = None,
    stop_pct: float | None = None,
    max_hold_bars: int | None = None,
    square_off_time: str | None = None,
    iv: float | None = None,
    days_to_expiry: float | None = None,
    strike_step: float | None = None,
    meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Re-price a candidate and apply the specified first-match exit rules."""
    pricing, settings = _pricing_settings(meta)
    bars = frame.reset_index(drop=True).copy()
    if not 0 <= row_index < len(bars):
        raise IndexError(f"Candidate row index is out of range: {row_index}")
    if "datetime" not in bars.columns or "close" not in bars.columns:
        raise ValueError("Trade re-pricing requires datetime and close columns.")
    bars["datetime"] = pd.to_datetime(bars["datetime"], errors="coerce")
    if bars["datetime"].isna().any():
        raise ValueError("Trade re-pricing received an invalid datetime.")

    row = bars.iloc[row_index]
    decision = str(row["decision"]).strip()
    if decision not in {"BUY_CE", "BUY_PE"}:
        raise ValueError("Only BUY_CE and BUY_PE candidate rows can be re-priced.")
    target = float(row["target_pct"] if target_pct is None else target_pct)
    stop = float(row["stop_pct"] if stop_pct is None else stop_pct)
    hold_limit = int(settings["max_hold_bars"] if max_hold_bars is None else max_hold_bars)
    square_off = str(settings["square_off"] if square_off_time is None else square_off_time)
    volatility = float(pricing["iv"] if iv is None else iv)
    expiry_days = float(pricing["days_to_expiry"] if days_to_expiry is None else days_to_expiry)
    strike_increment = float(pricing["strike_step"] if strike_step is None else strike_step)
    if target <= 0 or stop <= 0 or hold_limit <= 0 or expiry_days <= 0:
        raise ValueError("Trade target, stop, hold limit, and expiry must be positive.")
    if strike_increment <= 0:
        raise ValueError("Option strike step must be positive.")

    option_kind = "CE" if decision == "BUY_CE" else "PE"
    spot_at_entry = float(row["close"])
    strike = round(spot_at_entry / strike_increment) * strike_increment
    initial_time = expiry_days / 365.0
    entry_premium = black_scholes_price(
        spot_at_entry,
        strike,
        initial_time,
        option_kind,
        iv=volatility,
        rate=float(pricing["rate"]),
        meta=settings,
    )
    entry_day = row["datetime"].date()
    day_indices = [i for i, timestamp in enumerate(bars["datetime"]) if timestamp.date() == entry_day]
    last_day_index = day_indices[-1]
    exit_index = last_day_index
    exit_reason = "session_end"

    for current_index in range(row_index + 1, last_day_index + 1):
        current = bars.iloc[current_index]
        bars_elapsed = current_index - row_index
        remaining_time = initial_time - bars_elapsed * 5.0 / 525600.0
        premium = black_scholes_price(
            float(current["close"]),
            strike,
            remaining_time,
            option_kind,
            iv=volatility,
            rate=float(pricing["rate"]),
            meta=settings,
        )
        return_fraction = premium / entry_premium - 1.0
        if return_fraction >= target:
            exit_index, exit_reason = current_index, "target"
        elif return_fraction <= -stop:
            exit_index, exit_reason = current_index, "stop"
        elif bars_elapsed >= hold_limit:
            exit_index, exit_reason = current_index, "time"
        elif current["datetime"].strftime("%H:%M") >= square_off:
            exit_index, exit_reason = current_index, "square_off"
        else:
            continue
        break

    exit_row = bars.iloc[exit_index]
    elapsed_bars = exit_index - row_index
    remaining_time = initial_time - elapsed_bars * 5.0 / 525600.0
    exit_premium = black_scholes_price(
        float(exit_row["close"]),
        strike,
        remaining_time,
        option_kind,
        iv=volatility,
        rate=float(pricing["rate"]),
        meta=settings,
    )
    hit_target = exit_premium / entry_premium - 1.0 >= target - 1e-9
    return {
        "kind": option_kind,
        "strike": float(strike),
        "entry_premium": float(entry_premium),
        "exit_premium": float(exit_premium),
        "exit_reason": exit_reason,
        "exit_index": int(exit_index),
        "exit_datetime": exit_row["datetime"],
        "hit_target": bool(hit_target),
    }
