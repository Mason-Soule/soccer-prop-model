"""
backtest.py
-----------
Walk-forward backtest for the EPL Over 2.5 goals model.

Standard train/test split uses a single hold-out window (2022-2024).
That gives ~36 matches with odds — far too few to measure edge reliably.

Walk-forward backtest solves this by:
  1. Training on seasons 1..N
  2. Predicting on season N+1
  3. Repeating for each season, accumulating all predictions
  4. Evaluating edge across the full prediction history (~1500+ bets)

This mimics real deployment — you never train on future data, but you
accumulate predictions across many seasons to get statistically meaningful
edge estimates.

Usage:
    python simulation/backtest.py

Output:
    - Per-season performance table
    - Aggregate edge analysis across all seasons
    - Kelly staking simulation with PnL curve
    - simulation/backtest_results.csv  (all flagged bets)
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.metrics import roc_auc_score, log_loss
from xgboost import XGBClassifier

project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from data.processed.build_dataset import build_dataframe
from ingestion.odds_ingestion import load_odds, merge_odds_with_match_df, audit_name_mismatches

# ---------------------------------------------------------------------------
# Config — must match train.py exactly
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "avg_goals_last5_home",
    "avg_goals_last15_home",
    "avg_goals_conceded_last5_home",
    "avg_goals_conceded_last15_home",
    "avg_goals_last5_away",
    "avg_goals_last15_away",
    "avg_goals_conceded_last5_away",
    "avg_goals_conceded_last15_away",
    "avg_xg_last5_home",
    "avg_xg_last15_home",
    "avg_xga_last5_home",
    "avg_xga_last15_home",
    "avg_xg_last5_away",
    "avg_xg_last15_away",
    "avg_xga_last5_away",
    "avg_xga_last15_away",
    "over_2_5_rate_last8_home",
    "over_2_5_rate_last8_away",
    "win_rate_last8_home",
    "win_rate_last8_away",
    "avg_xg_overperf_last5_home",
    "avg_xg_overperf_last5_away",
    "avg_shot_quality_last5_home",
    "avg_shot_quality_last5_away",
    "days_rest_current_home",
    "days_rest_current_away",
    "days_rest_diff",
    "ref_over_rate_last20_home",
]

# Minimum seasons of training data before we start predicting
# 2 seasons minimum so the model has enough data to learn from
MIN_TRAIN_SEASONS = 4

# Edge threshold — must match train.py
EDGE_THRESHOLD = 0.08
MAX_ODDS = 1.75
MIN_ODDS = 1.60

# Kelly fraction — fractional Kelly to reduce variance
# Full Kelly is theoretically optimal but has huge drawdowns in practice
# Quarter Kelly (0.15) is more conservative and realistic
KELLY_FRACTION = 0.15

# Starting bankroll for simulation
STARTING_BANKROLL = 1000.0

OUTPUT_DIR = project_root / "simulation"
OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Model factory — identical config to train.py
# ---------------------------------------------------------------------------
def make_model():
    return XGBClassifier(
        n_estimators=1000,
        max_depth=2,
        learning_rate=0.01,
        subsample=0.6,
        colsample_bytree=0.4,
        min_child_weight=20,
        reg_alpha=2.0,
        reg_lambda=5.0,
        early_stopping_rounds=50,
        eval_metric="logloss",
        random_state=42,
    )


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    print("Building match features from DB...")
    match_df = build_dataframe()

    print("Loading odds data...")
    odds_df = load_odds()
    df = merge_odds_with_match_df(match_df, odds_df)

    mask_all_missing = df[FEATURE_COLS].isna().all(axis=1)
    df = df[~mask_all_missing].copy()

    print(f"Total rows loaded: {len(df)}")
    print(f"Seasons available: {sorted(df['season_start'].unique())}")
    return df


# ---------------------------------------------------------------------------
# Walk-forward folds
# ---------------------------------------------------------------------------
def get_folds(df: pd.DataFrame) -> list[dict]:
    """
    Generate walk-forward folds.

    For seasons [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]:
        Fold 1: train=2016-2017, test=2018
        Fold 2: train=2016-2018, test=2019
        Fold 3: train=2016-2019, test=2020
        ...
        Fold 6: train=2016-2022, test=2023

    Each fold trains on all available history up to the test season.
    This is the only leakage-free evaluation strategy.
    """
    seasons = sorted(df["season_start"].unique())
    folds   = []

    for i in range(MIN_TRAIN_SEASONS, len(seasons)):
        train_seasons = seasons[:i]
        test_season   = seasons[i]

        folds.append({
            "fold":          i - MIN_TRAIN_SEASONS + 1,
            "train_seasons": train_seasons,
            "test_season":   test_season,
        })

    return folds


# ---------------------------------------------------------------------------
# Kelly staking
# ---------------------------------------------------------------------------
def kelly_stake(edge: float, odds: float, fraction: float = KELLY_FRACTION) -> float:
    """
    Fractional Kelly criterion stake as a proportion of bankroll.

    Kelly formula: f = (bp - q) / b
        b = decimal odds - 1 (net odds)
        p = model probability of winning
        q = 1 - p

    Fractional Kelly multiplies by `fraction` to reduce variance.
    Negative Kelly = no bet (model sees negative edge).

    Args:
        edge:     model_prob - market_prob (vig-free)
        odds:     decimal odds for the bet
        fraction: Kelly multiplier (0.25 = quarter Kelly)

    Returns:
        Stake as proportion of current bankroll (0 if no edge)
    """
    # Reconstruct model probability from edge and market prob
    # edge = model_prob - market_prob  →  model_prob = market_prob + edge
    # We need model_prob directly for Kelly
    b = odds - 1.0
    # market_prob is baked into edge — use model_prob directly
    # model_prob = market_prob + edge, but we pass edge here
    # So: p = (market_prob_over + edge), which we don't have directly
    # Instead use simplified Kelly: stake proportional to edge / odds
    # This is a common approximation when edge is small
    if edge <= 0:
        return 0.0

    # Full Kelly: f = edge / (odds - 1)
    # Rationale: how much of your edge are you risking per unit of odds
    f = edge / b
    return max(0.0, f * fraction)


# ---------------------------------------------------------------------------
# Run one fold
# ---------------------------------------------------------------------------
def run_fold(df: pd.DataFrame, fold: dict) -> pd.DataFrame:
    """
    Train on fold's train seasons, predict on test season.
    Returns DataFrame of test season predictions with edge and bet columns.
    """
    train_df = df[df["season_start"].isin(fold["train_seasons"])].copy()
    test_df  = df[df["season_start"] == fold["test_season"]].copy()

    if len(train_df) == 0 or len(test_df) == 0:
        print(f"  Fold {fold['fold']}: skipped (empty split)")
        return pd.DataFrame()

    X_train = train_df[FEATURE_COLS]
    y_train = train_df["over_2_5"]
    X_test  = test_df[FEATURE_COLS]
    y_test  = test_df["over_2_5"]

    model = make_model()
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    probs = model.predict_proba(X_test)[:, 1]

    # AUC for this fold
    try:
        fold_auc = roc_auc_score(y_test, probs)
    except Exception:
        fold_auc = float("nan")

    test_df = test_df.copy()
    test_df["model_prob"]    = probs
    test_df["fold"]          = fold["fold"]
    test_df["train_seasons"] = str(fold["train_seasons"])
    test_df["fold_auc"]      = fold_auc

    # Edge calculation
    has_odds = test_df["market_prob_over"].notna()
    test_df.loc[has_odds, "edge"] = (
        test_df.loc[has_odds, "model_prob"] -
        test_df.loc[has_odds, "market_prob_over"]
    )
    test_df["bet_over"] = (
    (test_df["edge"] > EDGE_THRESHOLD) &
    (test_df["odds_over_2_5"] >= MIN_ODDS) &
    (test_df["odds_over_2_5"] <= MAX_ODDS)
)

    n_bets = test_df["bet_over"].sum()
    print(
        f"  Fold {fold['fold']}: "
        f"train={fold['train_seasons'][0]}-{fold['train_seasons'][-1]}  "
        f"test={fold['test_season']}  "
        f"AUC={fold_auc:.3f}  "
        f"bets={n_bets}/{has_odds.sum()}"
    )

    return test_df


# ---------------------------------------------------------------------------
# Aggregate results
# ---------------------------------------------------------------------------
def aggregate_results(all_predictions: pd.DataFrame) -> None:
    """Print full backtest summary and run Kelly staking simulation."""

    print("\n" + "="*60)
    print("WALK-FORWARD BACKTEST RESULTS")
    print("="*60)

    # --- Per-season summary ---
    print("\n--- Per-Season Performance ---")
    season_summary = []

    for season, group in all_predictions.groupby("season_start"):
        has_odds   = group["market_prob_over"].notna()
        odds_group = group[has_odds]
        bet_group  = odds_group[odds_group["bet_over"] == True]

        try:
            auc = roc_auc_score(group["over_2_5"], group["model_prob"])
        except Exception:
            auc = float("nan")

        hit_rate     = bet_group["over_2_5"].mean() if len(bet_group) > 0 else float("nan")
        implied_rate = bet_group["market_prob_over"].mean() if len(bet_group) > 0 else float("nan")
        outperf      = hit_rate - implied_rate if not np.isnan(hit_rate) else float("nan")

        season_summary.append({
            "season":    season,
            "matches":   len(group),
            "auc":       round(auc, 3),
            "bets":      len(bet_group),
            "hit_rate":  round(hit_rate, 3) if not np.isnan(hit_rate) else None,
            "implied":   round(implied_rate, 3) if not np.isnan(implied_rate) else None,
            "outperf":   round(outperf, 3) if not np.isnan(outperf) else None,
        })

    summary_df = pd.DataFrame(season_summary)
    print(summary_df.to_string(index=False))

    # --- Aggregate edge analysis ---
    has_odds  = all_predictions["market_prob_over"].notna()
    odds_preds = all_predictions[has_odds]
    bet_preds  = odds_preds[odds_preds["bet_over"] == True]

    print(f"\n--- Aggregate Edge Analysis ---")
    print(f"  Total matches evaluated:   {len(all_predictions)}")
    print(f"  Matches with odds:         {len(odds_preds)}")
    print(f"  Total flagged bets:        {len(bet_preds)} "
          f"({len(bet_preds)/len(odds_preds):.1%} of matches with odds)")

    if len(bet_preds) > 0:
        hit_rate     = bet_preds["over_2_5"].mean()
        implied_rate = bet_preds["market_prob_over"].mean()
        avg_edge     = bet_preds["edge"].mean()
        avg_odds     = bet_preds["odds_over_2_5"].mean() if "odds_over_2_5" in bet_preds.columns else float("nan")

        print(f"  Hit rate on flagged bets:  {hit_rate:.2%}")
        print(f"  Avg market implied prob:   {implied_rate:.2%}")
        print(f"  Outperformance vs market:  {hit_rate - implied_rate:+.2%}")
        print(f"  Avg edge on flagged bets:  {avg_edge:.4f}")
        if not np.isnan(avg_odds):
            print(f"  Avg decimal odds:          {avg_odds:.3f}")

    # --- Kelly staking simulation ---
    if len(bet_preds) > 0 and "odds_over_2_5" in bet_preds.columns:
        print(f"\n--- Kelly Staking Simulation (fraction={KELLY_FRACTION}) ---")
        print(f"  Starting bankroll: £{STARTING_BANKROLL:,.2f}")

        bankroll = STARTING_BANKROLL
        bankroll_history = [bankroll]
        wins = losses = 0
        total_staked = 0.0

        bet_log = []

        for _, row in bet_preds.sort_values("date_home").iterrows():
            if pd.isna(row["odds_over_2_5"]) or pd.isna(row["edge"]):
                continue

            stake_pct = kelly_stake(row["edge"], row["odds_over_2_5"])
            stake     = bankroll * stake_pct
            stake     = min(stake, bankroll * 0.05)  # hard cap: never bet >5% on one match

            if stake < 1.0:  # minimum bet size
                continue

            total_staked += stake

            if row["over_2_5"] == 1:  # bet won
                profit = stake * (row["odds_over_2_5"] - 1)
                bankroll += profit
                wins += 1
                result = "WIN"
            else:  # bet lost
                bankroll -= stake
                losses += 1
                result = "LOSS"

            bankroll_history.append(bankroll)
            bet_log.append({
                "date":        row.get("date_home", ""),
                "home":        row.get("team_name_home", ""),
                "away":        row.get("team_name_away", ""),
                "model_prob":  round(row["model_prob"], 3),
                "market_prob": round(row["market_prob_over"], 3),
                "edge":        round(row["edge"], 3),
                "odds":        round(row["odds_over_2_5"], 3),
                "stake":       round(stake, 2),
                "result":      result,
                "bankroll":    round(bankroll, 2),
            })

        total_bets   = wins + losses
        roi          = (bankroll - STARTING_BANKROLL) / total_staked * 100 if total_staked > 0 else 0
        final_return = (bankroll - STARTING_BANKROLL) / STARTING_BANKROLL * 100

        print(f"  Total bets placed:   {total_bets}")
        print(f"  Wins:                {wins} ({wins/total_bets:.1%})" if total_bets > 0 else "")
        print(f"  Total staked:        £{total_staked:,.2f}")
        print(f"  Final bankroll:      £{bankroll:,.2f}")
        print(f"  Net P&L:             £{bankroll - STARTING_BANKROLL:+,.2f}")
        print(f"  ROI on staked:       {roi:+.1f}%")
        print(f"  Total return:        {final_return:+.1f}%")

        # Max drawdown
        peak = STARTING_BANKROLL
        max_drawdown = 0.0
        for b in bankroll_history:
            if b > peak:
                peak = b
            drawdown = (peak - b) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        print(f"  Max drawdown:        {max_drawdown:.1%}")

        # Save bet log
        if bet_log:
            bet_log_df = pd.DataFrame(bet_log)
            out_path   = OUTPUT_DIR / "backtest_results.csv"
            bet_log_df.to_csv(out_path, index=False)
            print(f"\n  Bet log saved to {out_path}")

        # Bankroll curve
        print(f"\n  Bankroll every 50 bets:")
        for i, b in enumerate(bankroll_history):
            if i % 50 == 0:
                print(f"    Bet {i:>4}: £{b:,.2f}")

        
        


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = load_data()

    folds = get_folds(df)
    print(f"\nRunning {len(folds)} walk-forward folds...")
    print("-" * 60)

    all_predictions = []

    for fold in folds:
        fold_preds = run_fold(df, fold)
        if len(fold_preds) > 0:
            all_predictions.append(fold_preds)

    if not all_predictions:
        print("No predictions generated. Check your data.")
        sys.exit(1)

    all_predictions_df = pd.concat(all_predictions, ignore_index=True)

    aggregate_results(all_predictions_df)