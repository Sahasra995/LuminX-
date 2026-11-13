# LuminX

LuminX is a Flask dashboard for evaluating the out-of-sample Bank NIFTY options strategy and demonstrating live or replay signals with paper trading only. VWAP sets direction, a Bollinger Band z-score pullback sets timing, and a saved Random Forest approves or rejects live candidates.

## Run locally

Use Python 3.11 and the pinned dependencies:

```bash
pip install -r requirements.txt
flask --app app run
```

You can also start the application with `python app.py`. The production entry point is in `Procfile`.

## Tests

Run the offline test suite with:

```bash
pytest
```

Tests use replay bars or mocked provider responses and do not require market-data network access.

## Render deployment

1. Connect this GitHub repository as a Render Web Service.
2. Use the repository root as the service root directory.
3. Use `pip install -r requirements.txt` as the build command.
4. Use the `Procfile` start command, `gunicorn app:app --workers 1 --threads 4`.
5. Set the feed environment variables below in the Render dashboard.

The four research data artifacts in `data/` must be included with the deployment. Startup validates the artifacts and installed machine-learning package versions before serving requests.

## Feed environment

Copy `.env.example` for local configuration. Supported variables:

- `LIVE_PROVIDER`: `yahoo` or `replay`. Yahoo is the default provider; replay is available for demonstrations outside market hours.
- `LIVE_REFRESH_SECONDS`: shared live-provider refresh interval.
- `REPLAY_SECONDS`: replay bar advancement interval.
- `LIVE_LOOKBACK_DAYS`: intraday history window requested from the selected provider.
- `LIVE_SYMBOL`: Yahoo symbol for the Bank NIFTY index.
- `VIX_SYMBOL`: Yahoo symbol used when requesting India VIX.
- `IV_SOURCE`: `vix` to use Yahoo India VIX when available, otherwise the metadata IV; `slider` to use the dashboard IV control.

Yahoo's free intraday feed is delayed. When it cannot be fetched, LuminX keeps the last successful bars and shows a feed-status badge. Replay mode uses the supplied research bars and makes no market-data requests.

## Live paper ticket

Live mode includes a cash-checked manual order ticket, a separate bot paper account, optional bot copying, and metadata-driven risk enforcement and auto exits. Both accounts exist only in server memory; use **Reset paper account** to clear them. Replay mode supports play/pause, one-bar stepping, jumping to the next candidate, and paper buys/sells without market hours. Keyboard shortcuts C and P buy CE and PE while Live mode is active. The order, close, settings, reset, and replay endpoints are paper-only; no broker integration or real order submission is present.

## Data contract

The application reads and caches these four files at startup:

- `data/review3_signals.csv`: out-of-sample bars, indicators, candidate probabilities, and reference exit fixtures.
- `data/strategy_meta.json`: strategy, risk, transaction-cost, option-pricing, validation, and expected-result settings.
- `data/rf_model.joblib`: saved model used only to score live candidates; LuminX does not retrain it.
- `data/model_card.json`: live feature order and formulas, candidate rules, validation fixtures, and required library versions.

These artifacts are produced by the research/export code in this repository, including `export_review3_csv.py` and `export_live_model.py`. The dashboard does not modify or regenerate them.

## Important

Live option prices are Black-Scholes estimates, not exchange option-chain quotes. Yahoo market data may be delayed. All positions are simulated in server memory and reset on restart. LuminX never places real orders or connects to a broker.
