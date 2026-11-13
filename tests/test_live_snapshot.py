from __future__ import annotations

import pandas as pd
import pytest

import app as app_module
import live
from data_loader import ensure_app_ready


def _csv_bars() -> pd.DataFrame:
    """Return a local fixture of closed research bars in the provider format."""
    signals = ensure_app_ready()["signals"]
    return signals[["datetime", "open", "high", "low", "close", "volume"]].copy()


def test_replay_snapshot_has_market_state_and_repriced_chain(monkeypatch) -> None:
    """The replay snapshot should expose bars and a fully priced illustrative chain."""
    bars = _csv_bars()
    monkeypatch.setattr(live, "_fetch_bars", lambda provider: (bars, None, 0.0))

    snapshot = live.get_live_snapshot("replay", iv=ensure_app_ready()["meta"]["option_pricing"]["iv"])

    assert snapshot["provider"] == "replay"
    assert isinstance(snapshot["spot"], (int, float))
    assert len(snapshot["option_chain"]) == 5
    assert sum(row["atm"] for row in snapshot["option_chain"]) == 1
    assert snapshot["bars"]


def test_zero_volume_snapshot_shows_equal_weight_fallback(monkeypatch) -> None:
    """An index feed without volume should use and disclose equal-weight VWAP."""
    bars = _csv_bars()
    bars["volume"] = 0
    monkeypatch.setattr(live, "_fetch_bars", lambda provider: (bars, None, 0.0))

    snapshot = live.get_live_snapshot("replay")

    assert snapshot["volume_unavailable"] is True
    assert any("VWAP uses equal weights" in badge["label"] for badge in snapshot["badges"])


def test_replay_option_chain_uses_the_replay_bar_time_for_pricing() -> None:
    """Replay order quotes and position marks share the same expiry horizon."""
    data = ensure_app_ready()
    live.reset_paper_accounts()
    live._replay_provider._cursor = min(
        len(live._replay_provider._frame) - 1,
        int(data["meta"]["indicators"]["bb_window"]),
    )
    expiry = live.default_expiry_date()
    snapshot = live.get_live_snapshot("replay", expiry=expiry)

    timestamp = pd.Timestamp(snapshot["last_bar_time"])
    atm = next(row for row in snapshot["option_chain"] if row["atm"])
    time_to_expiry = live._time_to_expiry(
        timestamp.to_pydatetime(),
        pd.Timestamp(expiry).date(),
    )
    expected = live.black_scholes_price(
        float(snapshot["spot"]),
        float(atm["strike"]),
        time_to_expiry,
        "CE",
        iv=float(snapshot["iv_used"]),
        rate=float(data["meta"]["option_pricing"]["rate"]),
        meta=data["meta"],
    )

    assert atm["ce"]["premium"] == pytest.approx(expected)


def test_feed_failure_keeps_last_good_bars_and_exposes_error(monkeypatch) -> None:
    """A failed refresh should return cached data with a visible retry badge."""
    bars = _csv_bars()
    monkeypatch.setattr(
        live,
        "_fetch_bars",
        lambda provider: (bars, "fixture feed failure", 0.0),
    )

    snapshot = live.get_live_snapshot("replay")

    assert snapshot["spot"] == float(bars.iloc[-1]["close"])
    assert any(badge["label"] == "Feed unavailable, retrying" for badge in snapshot["badges"])


def test_live_api_uses_replay_without_network_and_has_elapsed_time(monkeypatch) -> None:
    """The API should use a mocked provider and return a timed JSON snapshot."""
    monkeypatch.setattr(app_module, "_active_live_provider", "replay")
    bars = _csv_bars()
    monkeypatch.setattr(live, "_fetch_bars", lambda provider: (bars, None, 0.0))

    response = app_module.app.test_client().get("/api/live/snapshot")

    assert response.status_code == 200
    assert "elapsed_ms" in response.get_json()
    assert response.get_json()["snapshot"]["provider"] == "replay"


def test_paper_endpoints_validate_requests_and_report_timing() -> None:
    """Manual paper controls are exposed, validated, and explicitly timed."""
    client = app_module.app.test_client()

    assert client.post("/api/live/order", json={}).status_code == 400
    assert client.post("/api/live/close", json={}).status_code == 400
    settings = client.post(
        "/api/live/settings",
        json={"auto_exit": True, "enforce_rules": True, "copy_bot": False},
    )
    assert settings.status_code == 200
    assert "elapsed_ms" in settings.get_json()
    reset = client.post("/api/live/reset", json={})
    assert reset.status_code == 200
    assert "elapsed_ms" in reset.get_json()
    assert client.post("/api/live/replay", json={"action": "invalid"}).status_code == 400
    assert client.post("/api/live/provider", json={"provider": "invalid"}).status_code == 400
    assert client.post("/api/live/provider", json={"provider": "replay"}).status_code == 200
