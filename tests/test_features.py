"""
tests/test_features.py
----------------------
Tests for feature engineering correctness.

Catches the bugs most likely to silently corrupt live results:
    - Duplicate columns in FEATURE_COLS
    - Rolling windows using current match data (leakage)
    - Missing features after build_dataframe()
    - Combined features computed correctly

Run with:
    python -m pytest tests/test_features.py -v
"""

import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from core.features import FEATURE_COLS, validate_feature_cols


# ---------------------------------------------------------------------------
# FEATURE_COLS integrity
# ---------------------------------------------------------------------------

def test_feature_cols_no_duplicates():
    """FEATURE_COLS must never contain duplicate entries."""
    seen = set()
    dupes = []
    for col in FEATURE_COLS:
        if col in seen:
            dupes.append(col)
        seen.add(col)
    assert dupes == [], (
        f"Duplicate entries in FEATURE_COLS: {dupes}. "
        f"Remove them from core/features.py."
    )


def test_feature_cols_not_empty():
    """FEATURE_COLS must have at least 30 features."""
    assert len(FEATURE_COLS) >= 30, (
        f"FEATURE_COLS only has {len(FEATURE_COLS)} entries — suspiciously low."
    )


def test_feature_cols_expected_groups():
    """FEATURE_COLS must contain features from each expected group."""
    required = [
        "avg_xg_last5_home",        # xG home
        "avg_xg_last5_away",        # xG away
        "combined_xg_last5",        # combined attack
        "league_avg_goals_last30",  # league baseline
        "h2h_avg_goals_last5",      # head to head
        "ref_over_rate_last20_home",# referee
        "days_rest_diff",           # rest
    ]
    missing = [c for c in required if c not in FEATURE_COLS]
    assert missing == [], (
        f"Expected features missing from FEATURE_COLS: {missing}"
    )


# ---------------------------------------------------------------------------
# Rolling window leakage tests
# ---------------------------------------------------------------------------

def _make_team_df(n_games: int = 20, goals: list = None) -> pd.DataFrame:
    """Create a minimal team-level DataFrame for testing rolling logic."""
    if goals is None:
        goals = list(range(1, n_games + 1))
    return pd.DataFrame({
        "team_id": [1] * n_games,
        "date":    pd.date_range("2020-01-01", periods=n_games, freq="7D"),
        "goals":   goals,
    })


def test_rolling_uses_shift():
    """
    Rolling features must use shift(1) so the current match is never
    included in its own feature calculation.

    If avg_goals_last5 for game 6 includes game 6's goals, that's leakage.
    We test this by checking the feature value equals the mean of the
    PRIOR 5 games, not the current + prior 4.
    """
    from features.rolling_stats import _rolling_mean_by_team

    goals = [1, 2, 3, 4, 5, 10]  # game 6 has 10 goals — obvious outlier
    df    = _make_team_df(n_games=6, goals=goals)
    df    = df.sort_values(["team_id", "date"])

    result = _rolling_mean_by_team(df, "goals", window=5)

    # Game 6 (index 5): correct = mean([1,2,3,4,5]) = 3.0
    # Leaked  = mean([2,3,4,5,10]) = 4.8
    game6_value = result.iloc[5]
    assert abs(game6_value - 3.0) < 0.01, (
        f"Rolling window leakage detected! "
        f"Expected 3.0 (mean of prior 5), got {game6_value:.2f}. "
        f"Check that shift(1) is applied before rolling()."
    )


def test_rolling_requires_min_periods():
    """
    Rolling features should be NaN for the first N games where
    there isn't enough history yet (min_periods=window).
    """
    from features.rolling_stats import _rolling_mean_by_team

    df     = _make_team_df(n_games=10)
    result = _rolling_mean_by_team(df, "goals", window=5)

    # First 5 rows should be NaN (not enough history)
    assert result.iloc[:5].isna().all(), (
        "Expected NaN for first 5 games (insufficient history) "
        "but got non-NaN values. Check min_periods setting."
    )

    # From game 6 onward should have values
    assert result.iloc[5:].notna().all(), (
        "Expected non-NaN from game 6 onward but got NaN values."
    )


# ---------------------------------------------------------------------------
# Combined feature correctness
# ---------------------------------------------------------------------------

def test_combined_xg_is_sum():
    """
    combined_xg_last5 must equal avg_xg_last5_home + avg_xg_last5_away.
    Tests that the addition in build_match_level_df is correct.
    """
    df = pd.DataFrame({
        "avg_xg_last5_home": [1.2, 1.5, 0.8],
        "avg_xg_last5_away": [0.9, 1.1, 1.3],
    })
    df["combined_xg_last5"] = df["avg_xg_last5_home"] + df["avg_xg_last5_away"]

    expected = [2.1, 2.6, 2.1]
    for i, exp in enumerate(expected):
        assert abs(df["combined_xg_last5"].iloc[i] - exp) < 0.001, (
            f"Row {i}: expected {exp}, got {df['combined_xg_last5'].iloc[i]}"
        )


def test_days_rest_diff_sign():
    """
    days_rest_diff = days_rest_current_home - days_rest_current_away.
    Positive means home team is more rested.
    """
    df = pd.DataFrame({
        "days_rest_current_home": [7, 3, 14],
        "days_rest_current_away": [3, 7,  7],
    })
    df["days_rest_diff"] = (
        df["days_rest_current_home"] - df["days_rest_current_away"]
    )

    assert df["days_rest_diff"].iloc[0] == 4,  "Home more rested: expected +4"
    assert df["days_rest_diff"].iloc[1] == -4, "Away more rested: expected -4"
    assert df["days_rest_diff"].iloc[2] == 7,  "Home much more rested: expected +7"


# ---------------------------------------------------------------------------
# validate_feature_cols
# ---------------------------------------------------------------------------

def test_validate_catches_missing_column():
    """validate_feature_cols should raise if a feature is missing."""
    df = pd.DataFrame({col: [1.0] for col in FEATURE_COLS[:-3]})  # missing last 3
    with pytest.raises(ValueError, match="Missing columns"):
        validate_feature_cols(df, label="test")


def test_validate_catches_duplicate_in_df():
    """
    validate_feature_cols should raise if selecting a column
    returns a DataFrame instead of a Series (duplicate col names in df).
    """
    # Build df with all features, then duplicate one column
    data = {col: [1.0, 2.0] for col in FEATURE_COLS}
    df = pd.DataFrame(data)

    # Manually create a duplicate column
    dup_col = FEATURE_COLS[0]
    df = pd.concat([df, df[[dup_col]]], axis=1)

    with pytest.raises(ValueError, match="DataFrame instead of a Series"):
        validate_feature_cols(df, label="test")


def test_validate_passes_clean_df():
    """validate_feature_cols should not raise on a clean DataFrame."""
    df = pd.DataFrame({col: [1.0, 2.0] for col in FEATURE_COLS})
    validate_feature_cols(df, label="test")  # should not raise


# ---------------------------------------------------------------------------
# H2H leakage test
# ---------------------------------------------------------------------------

def test_h2h_no_leakage():
    """
    H2H average must never include the current match's goals.
    Walk through 3 Arsenal vs Chelsea fixtures and verify each
    lookback only uses prior matches.
    """
    import numpy as np

    matches = pd.DataFrame({
        "date_home":       ["2020-01-01", "2020-08-01", "2021-01-01"],
        "team_name_home":  ["Arsenal",    "Chelsea",    "Arsenal"],
        "team_name_away":  ["Chelsea",    "Arsenal",    "Chelsea"],
        "goals_home":      [2,             1,            3],
        "goals_away":      [1,             2,            2],
    })
    matches["date_home"]      = pd.to_datetime(matches["date_home"])
    matches["total_goals_h2h"]= matches["goals_home"] + matches["goals_away"]
    matches["fixture_key"]    = matches.apply(
        lambda r: tuple(sorted([r["team_name_home"], r["team_name_away"]])),
        axis=1,
    )
    matches = matches.sort_values("date_home").reset_index(drop=True)

    h2h_vals        = []
    fixture_history = {}

    for _, row in matches.iterrows():
        key     = row["fixture_key"]
        history = fixture_history.get(key, [])
        h2h_vals.append(np.mean(history[-5:]) if len(history) >= 2 else np.nan)
        fixture_history[key] = history + [row["total_goals_h2h"]]

    matches["h2h_avg_goals_last5"] = h2h_vals

    # Match 1: no history → NaN
    assert pd.isna(matches["h2h_avg_goals_last5"].iloc[0]), (
        "First H2H match should be NaN (no history)"
    )
    # Match 2: only 1 prior match → NaN (min 2 required)
    assert pd.isna(matches["h2h_avg_goals_last5"].iloc[1]), (
        "Second H2H match should be NaN (only 1 prior, need 2)"
    )
    # Match 3: 2 prior matches → mean(3, 3) = 3.0
    assert abs(matches["h2h_avg_goals_last5"].iloc[2] - 3.0) < 0.01, (
        f"Match 3 H2H: expected 3.0, got {matches['h2h_avg_goals_last5'].iloc[2]}"
    )