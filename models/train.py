"""
train.py
--------
Trains a Logistic Regression model to predict Over 2.5 goals.
Odds are used ONLY for edge calculation after prediction — not as training features.

Usage:
    python models/train.py
"""

import sys
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from data.processed.build_dataset import build_dataframe
from ingestion.odds_ingestion import load_odds, merge_odds_with_match_df

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "avg_goals_last5_home",
    "avg_goals_conceded_last5_home",
    "avg_goals_last5_away",
    "avg_goals_conceded_last5_away",
    "total_attack_form",
    "total_defense_form",
]

TRAIN_SEASONS = (2016, 2021)  # inclusive
TEST_SEASONS  = (2022, 2024)  # inclusive

EDGE_THRESHOLD = 0.03  # minimum perceived edge to flag a bet (3%)

MODEL_OUT  = project_root / "models" / "trained_model.pkl"
SCALER_OUT = project_root / "models" / "scaler.pkl"


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    print("Building match features from DB...")
    match_df = build_dataframe()

    print("Loading odds data...")
    odds_df = load_odds()
    df = merge_odds_with_match_df(match_df, odds_df)

    before = len(df)
    df = df.dropna(subset=FEATURE_COLS)
    print(f"Dropped {before - len(df)} rows missing features. {len(df)} rows remaining.")

    return df


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------
def split_data(df: pd.DataFrame):
    train = df[
        (df["season_start"] >= TRAIN_SEASONS[0]) &
        (df["season_start"] <= TRAIN_SEASONS[1])
    ]
    test = df[
        (df["season_start"] >= TEST_SEASONS[0]) &
        (df["season_start"] <= TEST_SEASONS[1])
    ]

    print(f"\nTrain: {len(train)} rows  "
          f"({TRAIN_SEASONS[0]}-{TRAIN_SEASONS[1]})")
    print(f"Test:  {len(test)} rows  "
          f"({TEST_SEASONS[0]}-{TEST_SEASONS[1]})")

    return train, test


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
def train_model(X_train, y_train):
    """
    Train logistic regression with post-hoc probability calibration.

    Calibration matters here because we're comparing model probabilities
    directly against market implied probabilities to compute edge.
    Poorly calibrated probabilities produce misleading edge estimates.

    CalibratedClassifierCV with cv='prefit' wraps the already-trained
    model and fits a calibration layer on the training data using
    isotonic regression.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    base_model = LogisticRegression(max_iter=1000, C=0.1)
    base_model.fit(X_scaled, y_train)

    # Calibrate on training data
    # Note: ideally you'd calibrate on a held-out validation fold.
    # Once you have enough data, split train into train/val and
    # calibrate on val to avoid overfitting the calibration layer.
    # cv=5 fits calibration using 5-fold cross-validation internally
    # which gives better calibration than fitting on the same training data
    calibrated = CalibratedClassifierCV(base_model, cv=5, method="isotonic")
    calibrated.fit(X_scaled, y_train)

    return scaler, base_model, calibrated


# ---------------------------------------------------------------------------
# Evaluate model performance
# ---------------------------------------------------------------------------
def evaluate_model(model, scaler, test: pd.DataFrame):
    X_test = scaler.transform(test[FEATURE_COLS])
    probs  = model.predict_proba(X_test)[:, 1]
    y_test = test["over_2_5"]

    auc    = roc_auc_score(y_test, probs)
    logloss = log_loss(y_test, probs)
    brier  = brier_score_loss(y_test, probs)

    print("\n--- Model Performance (test set) ---")
    print(f"  AUC:        {auc:.4f}")
    print(f"  Log Loss:   {logloss:.4f}")
    print(f"  Brier Score:{brier:.4f}  (lower = better calibrated, 0.25 = baseline)")

    return probs


# ---------------------------------------------------------------------------
# Edge calculation — odds used here, NOT in training
# ---------------------------------------------------------------------------
def evaluate_edge(test: pd.DataFrame, model_probs: np.ndarray) -> pd.DataFrame:
    """
    Compare model probabilities against vig-free market implied probabilities.
    Rows without odds data are excluded from edge analysis.
    """
    results = test.copy()
    results["model_prob"] = model_probs

    has_odds = results["market_prob_over"].notna()
    edge_df  = results[has_odds].copy()

    if len(edge_df) == 0:
        print("\nNo odds data available for edge calculation.")
        return pd.DataFrame()

    edge_df["edge"] = edge_df["model_prob"] - edge_df["market_prob_over"]

    # Flag bets where model sees edge above threshold
    edge_df["bet_over"] = edge_df["edge"] > EDGE_THRESHOLD

    n_bets      = edge_df["bet_over"].sum()
    n_available = len(edge_df)

    print(f"\n--- Edge Analysis (threshold: {EDGE_THRESHOLD:.0%}) ---")
    print(f"  Matches with odds:  {n_available}")
    print(f"  Flagged bets:       {n_bets} ({n_bets/n_available:.1%} of matches)")
    print(f"  Avg edge (all):     {edge_df['edge'].mean():.4f}")
    print(f"  Avg edge (bets):    {edge_df[edge_df['bet_over']]['edge'].mean():.4f}"
          if n_bets > 0 else "  Avg edge (bets):    N/A")

    # Hit rate on flagged bets — are we right more than the market expects?
    if n_bets > 0:
        bet_subset    = edge_df[edge_df["bet_over"]]
        hit_rate      = bet_subset["over_2_5"].mean()
        implied_rate  = bet_subset["market_prob_over"].mean()
        print(f"\n  Hit rate on flagged bets:    {hit_rate:.2%}")
        print(f"  Avg market implied prob:     {implied_rate:.2%}")
        print(f"  Outperformance vs market:    {hit_rate - implied_rate:+.2%}")

    return edge_df


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
def save_artifacts(scaler, model):
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model,  MODEL_OUT)
    joblib.dump(scaler, SCALER_OUT)
    print(f"\nModel saved to  {MODEL_OUT}")
    print(f"Scaler saved to {SCALER_OUT}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = load_data()

    train, test = split_data(df)

    X_train = train[FEATURE_COLS]
    y_train = train["over_2_5"]

    print("\nTraining model...")
    scaler, base_model, calibrated_model = train_model(X_train, y_train)

    print("\n-- Uncalibrated --")
    raw_probs = evaluate_model(base_model, scaler, test)

    print("\n-- Calibrated --")
    cal_probs = evaluate_model(calibrated_model, scaler, test)

    # Use calibrated probabilities for edge — these are more accurate
    # probability estimates which matters when comparing to market odds
    edge_df = evaluate_edge(test, cal_probs)

    save_artifacts(scaler, calibrated_model)