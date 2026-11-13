"""Shared indicator and Random Forest feature calculations."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from data_loader import ensure_app_ready


def _settings(
    meta: Mapping[str, Any] | None = None,
    model_card: Mapping[str, Any] | None = None,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Resolve supplied or cached indicator and feature settings."""
    data = ensure_app_ready()
    return meta or data["meta"], model_card or data["model_card"]


def _prepare_bars(frame: pd.DataFrame) -> pd.DataFrame:
    """Copy bars, normalize supported datetime column names, and validate inputs."""
    bars = frame.copy()
    if "datetime" not in bars.columns:
        if "Datetime" in bars.columns:
            bars = bars.rename(columns={"Datetime": "datetime"})
        else:
            raise ValueError("Input bars must contain a datetime column.")
    required = {"open", "high", "low", "close", "volume"}
    missing = sorted(required.difference(bars.columns))
    if missing:
        raise ValueError(f"Input bars are missing columns: {', '.join(missing)}")
    bars["datetime"] = pd.to_datetime(bars["datetime"], errors="coerce")
    if bars["datetime"].isna().any():
        raise ValueError("Input bars contain an invalid datetime.")
    for column in ("high", "low", "close", "volume"):
        bars[column] = pd.to_numeric(bars[column], errors="coerce")
    return bars


def _formula_period(model_card: Mapping[str, Any], feature: str) -> int:
    """Read a rolling or lag period from the feature formula in model_card.json."""
    formula = str(model_card["feature_formulas"][feature])
    match = re.search(r"(?:shift|diff|pct_change|rolling)\((\d+)\)", formula)
    if not match:
        raise ValueError(f"Could not read the period for model feature {feature}.")
    return int(match.group(1))


def _market_open_offset(model_card: Mapping[str, Any]) -> int:
    """Read the elapsed-minute origin from model_card's feature formula."""
    formula = str(model_card["feature_formulas"]["mins_since_open"])
    match = re.search(r"-\s*(\d+)", formula)
    if not match:
        raise ValueError("Could not read the market-open offset from model_card.json.")
    return int(match.group(1))


def compute_vwap(frame: pd.DataFrame) -> pd.Series:
    """Compute daily volume-weighted typical price with equal weights when volume is absent."""
    bars = _prepare_bars(frame)
    session = bars["datetime"].dt.normalize()
    typical_price = (bars["high"] + bars["low"] + bars["close"]) / 3.0
    volume = bars["volume"].fillna(0.0).clip(lower=0.0)
    cumulative_value = (typical_price * volume).groupby(session, sort=False).cumsum()
    cumulative_volume = volume.groupby(session, sort=False).cumsum()
    equal_weight_vwap = typical_price.groupby(session, sort=False).expanding().mean()
    equal_weight_vwap.index = equal_weight_vwap.index.droplevel(0)
    daily_volume = volume.groupby(session, sort=False).transform("sum")
    weighted_vwap = cumulative_value.div(cumulative_volume.where(cumulative_volume > 0))
    result = weighted_vwap.where(daily_volume > 0, equal_weight_vwap)
    result.index = frame.index
    return result.astype(float)


def compute_indicators(
    frame: pd.DataFrame,
    meta: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Calculate daily VWAP and Bollinger bands using population standard deviation."""
    settings, _ = _settings(meta=meta)
    bars = _prepare_bars(frame)
    close = bars["close"]
    window = int(settings["indicators"]["bb_window"])
    multiplier = float(settings["indicators"]["bb_k"])
    mid = close.rolling(window=window, min_periods=window).mean()
    deviation = close.rolling(window=window, min_periods=window).std(ddof=0)
    bars["vwap"] = compute_vwap(bars)
    bars["bb_mid"] = mid
    bars["bb_upper"] = mid + multiplier * deviation
    bars["bb_lower"] = mid - multiplier * deviation
    bars["z_score"] = (close - mid).div(deviation.where(deviation != 0))
    bars["vwap_dist"] = (close - bars["vwap"]).div(bars["vwap"].where(bars["vwap"] != 0))
    bars["bw"] = (bars["bb_upper"] - bars["bb_lower"]).div(
        bars["bb_mid"].where(bars["bb_mid"] != 0)
    )
    return bars


def compute_feature_frame(
    frame: pd.DataFrame,
    meta: Mapping[str, Any] | None = None,
    model_card: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Build model features in the exact order described by model_card.json."""
    settings, card = _settings(meta, model_card)
    bars = compute_indicators(frame, settings)
    close = bars["close"]
    mid = bars["bb_mid"]
    volume = bars["volume"].fillna(0.0).clip(lower=0.0)
    session = bars["datetime"].dt.normalize()
    window = int(settings["indicators"]["bb_window"])
    deviation = close.rolling(window=window, min_periods=window).std(ddof=0)
    bandwidth = bars["bw"]
    distance_from_vwap = (close - bars["vwap"]).div(
        bars["vwap"].where(bars["vwap"] != 0)
    )
    daily_volume = volume.groupby(session, sort=False).transform("sum")
    volume_window = _formula_period(card, "vol_ratio")
    volume_ratio = volume.div(
        volume.rolling(window=volume_window, min_periods=volume_window).mean()
    )
    volume_ratio = volume_ratio.where(daily_volume > 0, 1.0)

    calculated: dict[str, pd.Series] = {
        "d_vwap": distance_from_vwap,
        "d_vwap_chg3": distance_from_vwap.diff(
            _formula_period(card, "d_vwap_chg3")
        ),
        "vwap_slope6": bars["vwap"].pct_change(
            _formula_period(card, "vwap_slope6")
        ),
        "z": (close - mid).div(deviation.where(deviation != 0)),
        "BW": bandwidth,
        "BW_chg5": bandwidth.pct_change(_formula_period(card, "BW_chg5")),
        "ma_slope6": mid.pct_change(_formula_period(card, "ma_slope6")),
        "vol_ratio": volume_ratio,
        "mins_since_open": (
            bars["datetime"].dt.hour * 60
            + bars["datetime"].dt.minute
            - _market_open_offset(card)
        ).astype(float),
        "ret_1bar": close.pct_change(),
    }

    decisions = bars["decision"] if "decision" in bars else pd.Series("", index=bars.index)
    bars["side"] = np.select(
        [decisions.eq("BUY_CE"), decisions.eq("BUY_PE")],
        [1.0, -1.0],
        default=0.0,
    )
    for feature in card["feature_order"]:
        if feature == "side":
            bars[feature] = bars["side"].astype(float)
        elif feature in calculated:
            bars[feature] = calculated[feature].astype(float)
        else:
            raise ValueError(f"Unsupported model feature in model_card.json: {feature}")
    return bars


def detect_candidate_signal(
    frame: pd.DataFrame,
    meta: Mapping[str, Any] | None = None,
    model_card: Mapping[str, Any] | None = None,
) -> pd.Series:
    """Return BUY_CE, BUY_PE, or HOLD according to the supplied candidate rule."""
    settings, card = _settings(meta, model_card)
    bars = compute_feature_frame(frame, settings, card)
    decision_rule = card["candidate_rule"]
    if not {"BUY_CE", "BUY_PE"}.issubset(decision_rule):
        raise ValueError("model_card.json candidate_rule must define BUY_CE and BUY_PE.")

    threshold = float(settings["indicators"]["dual_pullback_z"])
    window_start, window_end = settings["entry_window"]
    times = bars["datetime"].dt.strftime("%H:%M")
    in_entry_window = times.between(window_start, window_end, inclusive="both")
    previous_z = bars["z"].shift(1)
    close = bars["close"]
    vwap = bars["vwap"]
    buy_ce = (
        close.gt(vwap)
        & previous_z.le(-threshold)
        & bars["z"].gt(-threshold)
        & in_entry_window
    )
    buy_pe = (
        close.lt(vwap)
        & previous_z.ge(threshold)
        & bars["z"].lt(threshold)
        & in_entry_window
    )
    return pd.Series(
        np.select([buy_ce, buy_pe], ["BUY_CE", "BUY_PE"], default="HOLD"),
        index=frame.index,
        name="decision",
    )


def feature_vector_for_row(
    frame: pd.DataFrame,
    row_index: int,
    feature_order: Sequence[str] | None = None,
    meta: Mapping[str, Any] | None = None,
    model_card: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Compute a model-ordered feature dictionary for one row."""
    _, card = _settings(meta, model_card)
    order = feature_order if feature_order is not None else card["feature_order"]
    features = compute_feature_frame(frame, meta, card)
    row = features.iloc[row_index]
    return {name: float(row[name]) for name in order}
