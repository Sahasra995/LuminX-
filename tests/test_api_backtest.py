from __future__ import annotations

from data_loader import ensure_app_ready
from app import app


DATA = ensure_app_ready()


def test_default_backtest_api_matches_stored_results() -> None:
    """The all-days API response should match research-code expected results."""
    client = app.test_client()
    response = client.get("/api/backtest?date=all")

    assert response.status_code == 200
    payload = response.get_json()
    assert "elapsed_ms" in payload
    assert payload["metrics"]["Net P&L (Rs)"] == DATA["meta"]["expected"]["ai_model"]["Net P&L (Rs)"]
    assert payload["metrics"]["Trades taken"] == DATA["meta"]["expected"]["ai_model"]["Trades taken"]
    assert payload["no_ml_metrics"]["Net P&L (Rs)"] == DATA["meta"]["expected"]["no_ml"]["Net P&L (Rs)"]
    assert payload["timings"]["backtest_ms"] < 300


def test_api_backtest_clamps_controls_and_rejects_malformed_values() -> None:
    """Out-of-range controls are clamped while malformed values return JSON errors."""
    client = app.test_client()
    clamped = client.get("/api/backtest?conf=9&slip=20&iv=99")
    assert clamped.status_code == 200
    assert clamped.get_json()["conf"] == 0.8
    assert clamped.get_json()["slip_ticks"] == 10
    assert clamped.get_json()["iv"] == 0.6

    for query in ("conf=abc", "slip=1.5", "iv=nan", "date=not-a-date"):
        response = client.get(f"/api/backtest?{query}")
        assert response.status_code == 400
        assert "error" in response.get_json()
        assert "elapsed_ms" in response.get_json()


def test_chart_csv_and_meta_responses_include_required_data() -> None:
    """Auxiliary routes expose test days, detailed chart rows, and timed CSV."""
    client = app.test_client()
    meta = client.get("/api/meta").get_json()
    chart = client.get("/api/chart?date=all").get_json()
    csv_response = client.get("/api/trades.csv?date=all")

    assert meta["test_days"]
    assert "default_expiry" in meta
    assert chart["hint"] == "Pick a day for detail"
    assert chart["candles"]
    assert csv_response.status_code == 200
    assert "X-Elapsed-Ms" in csv_response.headers
    assert "exit_reason_label" in csv_response.get_data(as_text=True)
