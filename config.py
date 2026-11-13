"""Environment-backed settings for live and replay data feeds."""

from __future__ import annotations

import os


def _positive_integer(name: str, default: str) -> int:
    """Read a positive integer feed interval from the environment."""
    try:
        value = int(os.environ.get(name, default))
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


LIVE_PROVIDER = os.environ.get("LIVE_PROVIDER", "yahoo").strip().lower()
if LIVE_PROVIDER not in {"yahoo", "replay"}:
    raise ValueError("LIVE_PROVIDER must be 'yahoo' or 'replay'.")

LIVE_REFRESH_SECONDS = _positive_integer("LIVE_REFRESH_SECONDS", "60")
REPLAY_SECONDS = _positive_integer("REPLAY_SECONDS", "20")
LIVE_LOOKBACK_DAYS = _positive_integer("LIVE_LOOKBACK_DAYS", "5")
LIVE_BAR_INTERVAL_MINUTES = 5
LIVE_SYMBOL = os.environ.get("LIVE_SYMBOL", "^NSEBANK").strip()
VIX_SYMBOL = os.environ.get("VIX_SYMBOL", "^INDIAVIX").strip()
IV_SOURCE = os.environ.get("IV_SOURCE", "vix").strip().lower()
if IV_SOURCE not in {"vix", "slider"}:
    raise ValueError("IV_SOURCE must be 'vix' or 'slider'.")
