from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

from data_loader import load_model_card
from indicators import compute_feature_frame
from pricing import reprice_trade

MODEL_CARD = load_model_card()
MODEL = joblib.load("data/rf_model.joblib")


@pytest.fixture(scope="module")
def signals_frame() -> pd.DataFrame:
    frame = pd.read_csv("data/review3_signals.csv")
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    return frame


def test_feature_fixture_matches_model_and_expected_probability(signals_frame: pd.DataFrame) -> None:
    feature_frame = compute_feature_frame(signals_frame)
    feature_order = MODEL_CARD["feature_order"]
    model_features = MODEL_CARD["feature_fixture"]

    for fixture in model_features:
        timestamp = pd.Timestamp(fixture["datetime"])
        idx = int(np.where(feature_frame["datetime"] == timestamp)[0][0])
        row = feature_frame.iloc[idx]
        actual_features = {name: float(row[name]) for name in feature_order}
        actual_features["side"] = 1.0 if row["decision"] == "BUY_CE" else -1.0 if row["decision"] == "BUY_PE" else 0.0

        for name, expected in fixture["features"].items():
            assert abs(actual_features[name] - float(expected)) <= 1e-9, (
                fixture["datetime"],
                name,
                actual_features[name],
                expected,
            )

        feature_frame_input = pd.DataFrame([actual_features], columns=feature_order)
        probabilities = MODEL.predict_proba(feature_frame_input)[0, 1]
        assert abs(probabilities - float(fixture["expected_probability"])) <= 1e-6


def test_repricing_matches_csv_candidates(signals_frame: pd.DataFrame) -> None:
    mismatches: list[tuple[int, str, float, float]] = []
    for idx in signals_frame.index:
        row = signals_frame.iloc[idx]
        if row["decision"] == "HOLD":
            continue

        result = reprice_trade(signals_frame, idx)
        if (
            abs(float(row["entry_premium"]) - float(result["entry_premium"])) > 0.01
            or abs(float(row["exit_premium"]) - float(result["exit_premium"])) > 0.01
            or abs(float(row["strike"]) - float(result["strike"])) > 1e-9
            or str(row["exit_reason"]) != str(result["exit_reason"])
            or str(row["exit_datetime"]) != str(result["exit_datetime"])
        ):
            mismatches.append((idx, row["decision"], float(row["exit_premium"]), float(result["exit_premium"])))

    assert not mismatches, mismatches[:5]
