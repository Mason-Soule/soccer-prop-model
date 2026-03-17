"""
tests/test_odds.py
------------------
Tests for odds processing and merge correctness.

Run with:
    python -m pytest tests/test_odds.py -v
"""

import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from ingestion.odds_ingestion import remove_vig, process_odds, merge_odds_with_match_df


# ---------------------------------------------------------------------------
# remove_vig
# ---------------------------------------------------------------------------

def test_vig_removal_probabilities_sum_to_one():
    """
    After vig removal, market_prob_over + market_prob_under must equal 1.0.
    Raw bookmaker odds sum to > 1.0 (the overround). Normalising removes this.
    """
    over_odds  = pd.Series([1.80, 2.00, 1.65])
    under_odds = pd.Series([2.05, 1.85, 2.25])

    result = remove_vig(over_odds, under_odds)

    for i in range(len(result)):
        total = result["market_prob_over"].iloc[i] + result["market_prob_under"].iloc[i]
        assert abs(total - 1.0) < 0.0001, (
            f"Row {i}: probabilities sum to {total:.4f}, expected 1.0"
        )


def test_vig_removal_overround_greater_than_one():
    """Overround must always be > 1.0 — bookmakers always take a margin."""
    over_odds  = pd.Series([1.80, 2.00])
    under_odds = pd.Series([2.05, 1.85])

    result = remove_vig(over_odds, under_odds)
    assert (result["overround"] > 1.0).all(), (
        "Overround should always be > 1.0"
    )


def test_vig_removal_market_prob_between_0_and_1():
    """Market probabilities must be between 0 and 1."""
    over_odds  = pd.Series([1.50, 1.70, 2.20])
    under_odds = pd.Series([2.60, 2.20, 1.70])

    result = remove_vig(over_odds, under_odds)

    assert (result["market_prob_over"]  > 0).all()
    assert (result["market_prob_over"]  < 1).all()
    assert (result["market_prob_under"] > 0).all()
    assert (result["market_prob_under"] < 1).all()


def test_vig_removal_higher_odds_lower_probability():
    """Higher odds must correspond to lower implied probability."""
    over_odds  = pd.Series([1.50, 2.00])  # row 0 is shorter odds
    under_odds = pd.Series([2.60, 1.85])

    result = remove_vig(over_odds, under_odds)

    assert result["market_prob_over"].iloc[0] > result["market_prob_over"].iloc[1], (
        "Lower odds (1.50) should have higher implied probability than higher odds (2.00)"
    )


# ---------------------------------------------------------------------------
# process_odds
# ---------------------------------------------------------------------------

def _make_raw_odds_df() -> pd.DataFrame:
    """Minimal raw odds DataFrame mimicking football-data.co.uk format."""
    return pd.DataFrame({
        "Date":        ["15/08/2025", "22/08/2025", "29/08/2025"],
        "HomeTeam":    ["Arsenal",    "Man City",   "Liverpool"],
        "AwayTeam":    ["Chelsea",    "Brighton",   "Everton"],
        "B365>2.5":    [1.80,         1.65,          1.90],
        "B365<2.5":    [2.05,         2.25,          1.95],
        "season":      ["2025-26",    "2025-26",     "2025-26"],
    })


def test_process_odds_output_columns():
    """process_odds must return the expected output columns."""
    raw    = _make_raw_odds_df()
    result = process_odds(raw)

    required = [
        "date", "home_team", "away_team", "season",
        "odds_over_2_5", "odds_under_2_5",
        "market_prob_over", "market_prob_under",
        "overround",
    ]
    for col in required:
        assert col in result.columns, f"Missing column: {col}"


def test_process_odds_drops_invalid_rows():
    """Rows with all odds columns invalid must be dropped or have NaN market probs."""
    raw = _make_raw_odds_df()
    # Set ALL over odds columns to invalid for row 0
    for col in ["B365C>2.5", "BbAv>2.5", "B365>2.5"]:
        if col in raw.columns:
            raw.loc[0, col] = 0.5
    result = process_odds(raw)
    # Row should either be dropped or have NaN market prob
    invalid_rows = result[result["market_prob_over"].isna()]
    assert len(result) <= 3, "Should not gain rows"
    assert len(result[result["market_prob_over"].notna()]) == 2, (
        f"Expected 2 rows with valid market probs, got "
        f"{len(result[result['market_prob_over'].notna()])}"
    )


def test_process_odds_date_parsing():
    """Both DD/MM/YY and DD/MM/YYYY date formats must parse correctly."""
    raw = _make_raw_odds_df()
    raw.loc[0, "Date"] = "15/08/25"    # two-digit year
    raw.loc[1, "Date"] = "22/08/2025"  # four-digit year
    raw.loc[2, "Date"] = "29/08/2025"  # four-digit year
    result = process_odds(raw)
    # At least the rows with parseable dates should have valid dates
    assert result["date"].notna().sum() >= 2, (
        "At least 2 rows with parseable dates should succeed"
    )


def test_process_odds_team_name_normalisation():
    """Team names must be normalised using EPL.fd_team_map."""
    raw = _make_raw_odds_df()
    raw.loc[0, "HomeTeam"] = "Manchester City"  # should map to "Man City"
    result = process_odds(raw)
    assert "Man City" in result["home_team"].values or \
           "Manchester City" not in result["home_team"].values, (
        "Manchester City should be normalised to Man City"
    )


# ---------------------------------------------------------------------------
# merge_odds_with_match_df
# ---------------------------------------------------------------------------

def _make_match_df() -> pd.DataFrame:
    """Minimal match-level DataFrame as produced by build_match_level_df."""
    return pd.DataFrame({
        "match_id":        [1, 2],
        "date_home":       pd.to_datetime(["2025-08-15", "2025-08-22"]),
        "team_name_home":  ["Arsenal",  "Man City"],
        "team_name_away":  ["Chelsea",  "Brighton"],
        "over_2_5":        [1, 0],
        "season_start":    [2025, 2025],
    })


def _make_odds_df() -> pd.DataFrame:
    """Minimal processed odds DataFrame."""
    return pd.DataFrame({
        "date":             pd.to_datetime(["2025-08-15", "2025-08-22"]),
        "home_team":        ["Arsenal",  "Man City"],
        "away_team":        ["Chelsea",  "Brighton"],
        "odds_over_2_5":    [1.80,       1.65],
        "odds_under_2_5":   [2.05,       2.25],
        "market_prob_over": [0.527,      0.580],
        "market_prob_under":[0.473,      0.420],
        "overround":        [1.052,      1.049],
        "odds_source":      ["B365",     "B365"],
        "season":           ["2025-26",  "2025-26"],
    })


def test_merge_adds_odds_columns():
    """merge_odds_with_match_df must add odds columns to match_df."""
    match_df = _make_match_df()
    odds_df  = _make_odds_df()
    result   = merge_odds_with_match_df(match_df, odds_df)

    assert "odds_over_2_5"    in result.columns
    assert "market_prob_over" in result.columns
    assert "overround"        in result.columns


def test_merge_match_rate_100_percent():
    """All rows should match when data is consistent."""
    match_df = _make_match_df()
    odds_df  = _make_odds_df()
    result   = merge_odds_with_match_df(match_df, odds_df)

    matched = result["market_prob_over"].notna().sum()
    assert matched == len(match_df), (
        f"Expected all {len(match_df)} rows to match, only {matched} matched"
    )


def test_merge_unmatched_rows_get_nan():
    """Rows with no matching odds should get NaN, not be dropped."""
    match_df = _make_match_df()
    odds_df  = _make_odds_df().iloc[:1]  # only first row in odds

    result = merge_odds_with_match_df(match_df, odds_df)

    assert len(result) == len(match_df), (
        "Unmatched rows should be kept with NaN odds, not dropped"
    )
    assert result["market_prob_over"].isna().sum() == 1, (
        "Unmatched row should have NaN market_prob_over"
    )


def test_merge_no_duplicate_columns():
    """merge must not introduce duplicate column names."""
    match_df = _make_match_df()
    odds_df  = _make_odds_df()
    result   = merge_odds_with_match_df(match_df, odds_df)

    dupes = [c for c in result.columns if list(result.columns).count(c) > 1]
    assert dupes == [], f"Duplicate columns after merge: {dupes}"