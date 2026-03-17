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

from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from data.processed.build_dataset import build_dataframe
from ingestion.odds_ingestion import load_odds, merge_odds_with_match_df
from core.features import FEATURE_COLS
from core.model import make_model, train_model_full as train_model
from core.data_loader import load_data

TRAIN_SEASONS = (2016, 2021)  # inclusive
TEST_SEASONS  = (2022, 2024)  # inclusive

EDGE_THRESHOLD = 0.11  # minimum perceived edge to flag a bet (3%)

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
    # XGBoost can handle NaNs natively, so we do NOT need to drop
    # every row that has at least one missing feature. We only drop
    # rows where *all* rolling features are missing, which corresponds
    # to teams with essentially no history.
    mask_all_missing = df[FEATURE_COLS].isna().all(axis=1)
    dropped = int(mask_all_missing.sum())
    df = df[~mask_all_missing].copy()

    print(
        f"Dropped {dropped} rows with all rolling features missing. "
        f"{len(df)} rows remaining (from {before})."
    )

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
# Evaluate model performance
# ---------------------------------------------------------------------------
def evaluate_model(model, test: pd.DataFrame):
    X_test = test[FEATURE_COLS]
    probs  = model.predict_proba(X_test)[:, 1]
    y_test = test["over_2_5"]

    auc     = roc_auc_score(y_test, probs)
    logloss = log_loss(y_test, probs)
    brier   = brier_score_loss(y_test, probs)

    print("\n--- Model Performance (test set) ---")
    print(f"  AUC:         {auc:.4f}")
    print(f"  Log Loss:    {logloss:.4f}")
    print(f"  Brier Score: {brier:.4f}  (baseline is 0.25)")

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
    MAX_ODDS = 1.75
    MIN_ODDS = 1.60
    edge_df["bet_over"] = (
    (edge_df["edge"] > EDGE_THRESHOLD) &
    (edge_df["odds_over_2_5"] >= MIN_ODDS) &
    (edge_df["odds_over_2_5"] <= MAX_ODDS)
)

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
def save_artifacts(model):
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_OUT)
    print(f"\nModel saved to {MODEL_OUT}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = load_data()

    train, test = split_data(df)

    X_train = train[FEATURE_COLS]
    y_train = train["over_2_5"]
    X_test  = test[FEATURE_COLS]
    y_test  = test["over_2_5"]

    print("\nTraining XGBoost model...")
    model = train_model(X_train, y_train)

    probs   = evaluate_model(model, test)
    edge_df = evaluate_edge(test, probs)

    save_artifacts(model)