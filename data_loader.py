"""Load and validate the supplied LuminX research artifacts once per process."""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"

STRATEGY_META_PATH = DATA_DIR / "strategy_meta.json"
MODEL_CARD_PATH = DATA_DIR / "model_card.json"
MODEL_PATH = DATA_DIR / "rf_model.joblib"
SIGNALS_PATH = DATA_DIR / "review3_signals.csv"

REQUIRED_STRATEGY_META_KEYS = {
    "capital",
    "lot_size",
    "risk_per_trade",
    "target_pct",
    "stop_pct",
    "max_hold_bars",
    "max_trades_per_day",
    "daily_loss_cap_pct",
    "daily_loss_cap_rs",
    "entry_window",
    "square_off",
    "costs",
    "option_pricing",
    "indicators",
    "rf",
    "validation",
    "expected",
}
REQUIRED_META_NESTED_KEYS = {
    "costs": {
        "tick",
        "slippage_ticks_default",
        "brokerage_per_order",
        "stt_sell",
        "exchange_txn",
        "sebi_fee",
        "stamp_buy",
        "gst",
    },
    "option_pricing": {"iv", "rate", "days_to_expiry", "strike_step", "note"},
    "indicators": {"bb_window", "bb_k", "dual_pullback_z"},
    "rf": {"params", "tau_default", "label"},
    "validation": {
        "method",
        "warmup_sessions",
        "train_start",
        "test_start",
        "test_end",
        "sessions_total",
    },
    "expected": {"ai_model", "no_ml"},
}
REQUIRED_EXPECTED_RESULT_KEYS = {
    "ai_model": {
        "Trades taken",
        "Net P&L (Rs)",
        "ROI %",
        "Sharpe (ann.)",
        "Max drawdown %",
        "Win rate %",
        "Profit factor",
        "Accuracy rate %",
        "Precision %",
        "Recall %",
        "F1-score",
    },
    "no_ml": {
        "Trades taken",
        "Net P&L (Rs)",
        "ROI %",
        "Sharpe (ann.)",
        "Max drawdown %",
        "Win rate %",
        "Profit factor",
    },
}
REQUIRED_MODEL_CARD_KEYS = {
    "feature_order",
    "feature_formulas",
    "candidate_rule",
    "versions",
    "feature_fixture",
}
REQUIRED_SIGNAL_COLUMNS = {
    "datetime",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
    "bb_mid",
    "bb_upper",
    "bb_lower",
    "z_score",
    "vwap_dist",
    "decision",
    "rf_confidence",
    "strike",
    "entry_premium",
    "target_pct",
    "stop_pct",
    "target_premium",
    "stop_premium",
    "exit_datetime",
    "exit_premium",
    "exit_reason",
    "hit_target",
}


def _require_file(path: Path, label: str) -> None:
    """Raise a clear startup error when a required research artifact is absent."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing required {label}: {path}")


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    """Validate a JSON object or nested object."""
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return value


def _require_keys(
    mapping: dict[str, Any],
    required_keys: set[str],
    label: str,
) -> None:
    """Report required contract keys that are absent."""
    missing = sorted(required_keys.difference(mapping))
    if missing:
        raise ValueError(f"{label} is missing required keys: {', '.join(missing)}")


def _load_json(path: Path, label: str) -> dict[str, Any]:
    """Read one JSON artifact and validate its top-level shape."""
    _require_file(path, label)
    try:
        with path.open(encoding="utf-8") as source:
            data = json.load(source)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} contains invalid JSON: {exc}") from exc
    return _require_mapping(data, label)


def _validate_meta(meta: dict[str, Any]) -> None:
    """Validate required strategy settings and the expected-results contract."""
    _require_keys(meta, REQUIRED_STRATEGY_META_KEYS, "strategy_meta.json")
    for section, required_keys in REQUIRED_META_NESTED_KEYS.items():
        value = _require_mapping(meta[section], f"strategy_meta.json.{section}")
        _require_keys(value, required_keys, f"strategy_meta.json.{section}")
    expected = meta["expected"]
    for result_name, required_keys in REQUIRED_EXPECTED_RESULT_KEYS.items():
        result = _require_mapping(
            expected[result_name],
            f"strategy_meta.json.expected.{result_name}",
        )
        _require_keys(
            result,
            required_keys,
            f"strategy_meta.json.expected.{result_name}",
        )


def _validate_model_card(model_card: dict[str, Any]) -> None:
    """Validate feature metadata and check installed ML package versions."""
    _require_keys(model_card, REQUIRED_MODEL_CARD_KEYS, "model_card.json")
    feature_order = model_card["feature_order"]
    if not isinstance(feature_order, list) or not feature_order:
        raise ValueError("model_card.json.feature_order must be a non-empty list.")

    feature_formulas = _require_mapping(
        model_card["feature_formulas"],
        "model_card.json.feature_formulas",
    )
    if set(feature_order) != set(feature_formulas):
        raise ValueError(
            "model_card.json.feature_order and feature_formulas must contain "
            "the same feature names."
        )
    candidate_rule = _require_mapping(
        model_card["candidate_rule"],
        "model_card.json.candidate_rule",
    )
    _require_keys(
        candidate_rule,
        {"BUY_CE", "BUY_PE"},
        "model_card.json.candidate_rule",
    )

    fixtures = model_card["feature_fixture"]
    if not isinstance(fixtures, list):
        raise ValueError("model_card.json.feature_fixture must be a list.")
    for index, fixture in enumerate(fixtures):
        fixture = _require_mapping(fixture, f"feature_fixture[{index}]")
        _require_keys(
            fixture,
            {"datetime", "features", "expected_probability"},
            f"feature_fixture[{index}]",
        )
        features = _require_mapping(
            fixture["features"],
            f"feature_fixture[{index}].features",
        )
        if set(features) != set(feature_order):
            raise ValueError(
                f"feature_fixture[{index}].features does not match feature_order."
            )

    versions = _require_mapping(model_card["versions"], "model_card.json.versions")
    if not versions:
        raise ValueError("model_card.json.versions must not be empty.")
    for package_name, expected_version in versions.items():
        try:
            installed_version = metadata.version(package_name)
        except metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"Required package {package_name}={expected_version} is not installed."
            ) from exc
        if installed_version != expected_version:
            raise RuntimeError(
                f"Installed {package_name}={installed_version} does not match "
                f"model_card.json version {expected_version}."
            )


def _load_signals() -> pd.DataFrame:
    """Read the out-of-sample CSV and validate its required columns and dates."""
    _require_file(SIGNALS_PATH, "review3_signals.csv")
    try:
        signals = pd.read_csv(SIGNALS_PATH)
    except (OSError, pd.errors.ParserError, UnicodeError) as exc:
        raise ValueError(f"Could not read review3_signals.csv: {exc}") from exc

    missing = sorted(REQUIRED_SIGNAL_COLUMNS.difference(signals.columns))
    if missing:
        raise ValueError(
            f"review3_signals.csv is missing columns: {', '.join(missing)}"
        )
    for column in ("datetime", "exit_datetime"):
        timestamps = signals[column]
        if column == "exit_datetime":
            timestamps = timestamps.dropna()
        parsed_dates = pd.to_datetime(timestamps, errors="coerce")
        if parsed_dates.isna().any():
            raise ValueError(
                f"review3_signals.csv contains an invalid {column} value."
            )
    return signals


@lru_cache(maxsize=1)
def load_app_data() -> dict[str, Any]:
    """Load, validate, and cache all four supplied data artifacts."""
    meta = _load_json(STRATEGY_META_PATH, "strategy_meta.json")
    model_card = _load_json(MODEL_CARD_PATH, "model_card.json")
    _validate_meta(meta)
    _validate_model_card(model_card)

    _require_file(MODEL_PATH, "rf_model.joblib")
    model = joblib.load(MODEL_PATH)
    if not callable(getattr(model, "predict_proba", None)):
        raise ValueError("rf_model.joblib does not expose predict_proba.")
    model_feature_count = getattr(model, "n_features_in_", None)
    if model_feature_count is not None and model_feature_count != len(
        model_card["feature_order"]
    ):
        raise ValueError(
            "rf_model.joblib feature count does not match model_card.json."
        )

    return {
        "meta": meta,
        "model_card": model_card,
        "model": model,
        "signals": _load_signals(),
    }


def ensure_app_ready() -> dict[str, Any]:
    """Return the process-cached data after validating it on first use."""
    return load_app_data()


def load_strategy_meta() -> dict[str, Any]:
    """Return strategy metadata from the process cache."""
    return ensure_app_ready()["meta"]


def load_model_card() -> dict[str, Any]:
    """Return the model card from the process cache."""
    return ensure_app_ready()["model_card"]


def load_review3_signals() -> pd.DataFrame:
    """Return out-of-sample signals from the process cache."""
    return ensure_app_ready()["signals"]
