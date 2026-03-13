from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

DEFAULT_WINDOWS = (3, 5, 8, 15)
DEFAULT_STATS   = ("goals", "goals_conceded", "shots", "shots_conceded", "xg", "xga")


# ---------------------------------------------------------------------------
# Core rolling helpers
# ---------------------------------------------------------------------------

def _rolling_mean_by_team(df: pd.DataFrame, stat_col: str, window: int) -> pd.Series:
    """Overall rolling mean — all games regardless of venue."""
    return df.groupby("team_id")[stat_col].transform(
        lambda s: s.shift(1).rolling(window=window, min_periods=window).mean()
    )


def _rolling_std_by_team(df: pd.DataFrame, stat_col: str, window: int) -> pd.Series:
    """Rolling standard deviation — measures consistency/variance."""
    return df.groupby("team_id")[stat_col].transform(
        lambda s: s.shift(1).rolling(window=window, min_periods=window).std()
    )


# ---------------------------------------------------------------------------
# Venue-specific rolling helper
# ---------------------------------------------------------------------------

def _venue_rolling_mean(
    df: pd.DataFrame,
    stat_col: str,
    window: int,
    is_home: bool,
) -> pd.Series:
    """
    Rolling mean restricted to home-only or away-only games.

    For each team:
      1. Take only rows at the desired venue (home or away)
      2. Compute shift(1).rolling(window) on this compressed sequence
      3. Write results back only on those venue rows (NaN elsewhere)
    """
    df = df.sort_values(["team_id", "date"]).copy()

    is_home_norm = df["is_home"].map(
        lambda x: str(x).strip().lower() in ("true", "1", "t")
    )

    out = pd.Series(index=df.index, dtype="float64")

    for team_id, team_df in df.groupby("team_id", sort=False):
        venue_mask = is_home_norm.loc[team_df.index] == is_home
        venue_df   = team_df[venue_mask]
        if venue_df.empty:
            continue

        rolled = (
            venue_df[stat_col]
            .shift(1)
            .rolling(window=window, min_periods=window)
            .mean()
        )
        out.loc[venue_df.index] = rolled

    return out


# ---------------------------------------------------------------------------
# Standard rolling averages
# ---------------------------------------------------------------------------

def add_rolling_averages(
    df: pd.DataFrame,
    *,
    stats: Iterable[str] = DEFAULT_STATS,
    windows: Iterable[int] = DEFAULT_WINDOWS,
) -> pd.DataFrame:
    """
    Add overall rolling mean features for each stat x window combination.

    Output column naming: avg_{stat}_last{window}
    Examples: avg_goals_last5, avg_xg_last8, avg_goals_conceded_last15
    """
    df = df.sort_values(["team_id", "date"]).copy()

    missing = [s for s in stats if s not in df.columns]
    if missing:
        raise ValueError(f"Stats not found in DataFrame: {missing}")

    extra_windows = {
        "goals":          (15,),
        "goals_conceded": (15,),
        "xg":             (15,),
        "xga":            (15,),
    }

    for stat in stats:
        for w in windows:
            w = int(w)
            df[f"avg_{stat}_last{w}"] = _rolling_mean_by_team(df, stat, w)

        for w in extra_windows.get(stat, ()):
            w = int(w)
            col = f"avg_{stat}_last{w}"
            if col not in df.columns:
                df[col] = _rolling_mean_by_team(df, stat, w)

    return df


# ---------------------------------------------------------------------------
# Derived rolling features
# ---------------------------------------------------------------------------

def add_derived_features(
    df: pd.DataFrame,
    windows: Iterable[int] = (5, 8),
) -> pd.DataFrame:
    """
    Add higher-order rolling features that capture signals beyond raw averages.

    Requires add_rolling_averages() to have been called first.

    Features added:
        over_2_5_rate_last{w}       - % of last N games that went over 2.5
        win_rate_last{w}            - % of last N games won
        avg_goal_diff_last{w}       - rolling net goals per game
        avg_xg_overperf_last{w}     - rolling xG overperformance (goals - xG)
                                      positive = lucky, negative = unlucky/due
        avg_shot_quality_last{w}    - rolling xG per shot (chance quality)
        avg_goals_variance_last{w}  - rolling std dev of goals scored
        avg_days_rest_last1         - days since last match (single value, no window)
        ref_over_rate_last{10,20}   - referee rolling over 2.5 rate
    """
    df = df.sort_values(["team_id", "date"]).copy()

    for w in windows:
        w = int(w)

        # Over 2.5 rate — most directly predictive, mirrors the target variable
        if "over_2_5" in df.columns:
            df[f"over_2_5_rate_last{w}"] = _rolling_mean_by_team(df, "over_2_5", w)

        # Win rate — winning teams play more openly
        if "win" in df.columns:
            df[f"win_rate_last{w}"] = _rolling_mean_by_team(df, "win", w)

        # Goal difference — net goals per game, captures two-way quality
        if "goal_diff" in df.columns:
            df[f"avg_goal_diff_last{w}"] = _rolling_mean_by_team(df, "goal_diff", w)

        # xG overperformance — mean reversion signal
        # High positive = scoring more than deserved, expect regression
        # High negative = unlucky, expect more goals soon
        if "xg_overperformance" in df.columns:
            df[f"avg_xg_overperf_last{w}"] = _rolling_mean_by_team(
                df, "xg_overperformance", w
            )

        # Shot quality — xG per shot, distinguishes chance quality from volume
        if "shot_quality" in df.columns:
            df[f"avg_shot_quality_last{w}"] = _rolling_mean_by_team(
                df, "shot_quality", w
            )

        # Goals variance — high variance = unpredictable, more extreme scorelines
        if "goals" in df.columns:
            df[f"avg_goals_variance_last{w}"] = _rolling_std_by_team(df, "goals", w)

    # Days rest — current value only (not a rolling avg)
    # Fatigue affects defensive organization more than attacking instinct
    if "days_rest" in df.columns:
        df["days_rest_current"] = df["days_rest"]

    # Referee rolling over rate — some referees consistently produce more/fewer goals
    # min_periods=w//2 allows partial windows so fewer NaNs for rare referee combos
    if "referee" in df.columns and "over_2_5" in df.columns:
        for w in (10, 20):
            df[f"ref_over_rate_last{w}"] = (
                df.groupby("referee")["over_2_5"].transform(
                    lambda s: s.shift(1)
                               .rolling(window=w, min_periods=w // 2)
                               .mean()
                )
            )

    # Short-window momentum — how much the team's last 3 games deviate
    # from their 15-game baseline. Captures heating up / cooling down.
    # Positive = more dangerous than usual, negative = going cold.
    if "avg_xg_last3" not in df.columns and "xg" in df.columns:
        df["avg_xg_last3"] = _rolling_mean_by_team(df, "xg", 3)
    if "avg_goals_last3" not in df.columns and "goals" in df.columns:
        df["avg_goals_last3"] = _rolling_mean_by_team(df, "goals", 3)

    if "avg_xg_last3" in df.columns and "avg_xg_last15" in df.columns:
        df["xg_momentum_last3"] = df["avg_xg_last3"] - df["avg_xg_last15"]

    if "avg_goals_last3" in df.columns and "avg_goals_last15" in df.columns:
        df["goals_momentum_last3"] = df["avg_goals_last3"] - df["avg_goals_last15"]

    # Tighter over 2.5 rate window — more responsive to recent form
    if "over_2_5" in df.columns:
        df["over_2_5_rate_last5"] = _rolling_mean_by_team(df, "over_2_5", 5)
    
    # Referee short-window over rate — more responsive than last20
    if "referee" in df.columns and "over_2_5" in df.columns:
        df["ref_over_rate_last10"] = (
            df.groupby("referee")["over_2_5"].transform(
                lambda s: s.shift(1)
                           .rolling(window=10, min_periods=5)
                           .mean()
            )
        )

    # Referee foul rate — high foul refs disrupt flow and suppress goals
    if "referee" in df.columns and "fouls" in df.columns:
        df["ref_foul_rate_last20"] = (
            df.groupby("referee")["fouls"].transform(
                lambda s: s.shift(1)
                           .rolling(window=20, min_periods=10)
                           .mean()
            )
        )

    return df


# ---------------------------------------------------------------------------
# Venue-specific rolling averages
# ---------------------------------------------------------------------------

def add_venue_rolling_averages(
    df: pd.DataFrame,
    *,
    stats: Iterable[str] = ("goals", "goals_conceded", "xg", "xga"),
    windows: Iterable[int] = (3, 5, 8),
) -> pd.DataFrame:
    """
    Add venue-specific rolling mean features (home games only / away games only).

    Output column naming:
        avg_{stat}_home_last{window}  - computed from home games only
        avg_{stat}_away_last{window}  - computed from away games only

    Requires `is_home` column (bool or bool-like from PostgreSQL).
    """
    if "is_home" not in df.columns:
        raise ValueError(
            "'is_home' column not found. "
            "Must exist before calling add_venue_rolling_averages()."
        )

    df = df.sort_values(["team_id", "date"]).copy()

    missing = [s for s in stats if s not in df.columns]
    if missing:
        raise ValueError(f"Stats not found in DataFrame: {missing}")

    for stat in stats:
        for w in windows:
            w = int(w)
            df[f"avg_{stat}_home_last{w}"] = _venue_rolling_mean(
                df, stat, w, is_home=True
            )
            df[f"avg_{stat}_away_last{w}"] = _venue_rolling_mean(
                df, stat, w, is_home=False
            )

    return df


# ---------------------------------------------------------------------------
# Backwards-compatible wrappers
# ---------------------------------------------------------------------------

def avg_goals_last5(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    df = df.sort_values(["team_id", "date"]).copy()
    df["avg_goals_last5"] = _rolling_mean_by_team(df, "goals", int(window))
    return df


def avg_goals_conceded_last5(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    df = df.sort_values(["team_id", "date"]).copy()
    df["avg_goals_conceded_last5"] = _rolling_mean_by_team(df, "goals_conceded", int(window))
    return df


def avg_shots_last5(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    df = df.sort_values(["team_id", "date"]).copy()
    df["avg_shots_last5"] = _rolling_mean_by_team(df, "shots", int(window))
    return df


def avg_shots_conceded_last5(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    df = df.sort_values(["team_id", "date"]).copy()
    df["avg_shots_conceded_last5"] = _rolling_mean_by_team(df, "shots_conceded", int(window))
    return df