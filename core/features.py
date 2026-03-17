"""
core/features.py
----------------
Single source of truth for FEATURE_COLS.

Every script that needs features imports from here:
    from core.features import FEATURE_COLS

Never define FEATURE_COLS locally in backtest.py, train.py, or predict.py.
Add new features here and they automatically appear everywhere.
"""

FEATURE_COLS = [
    # --- Goals rolling averages ---
    # Short (last 5) and long (last 15) windows for both teams
    "avg_goals_last5_home",
    "avg_goals_last15_home",
    "avg_goals_conceded_last5_home",
    "avg_goals_conceded_last15_home",
    "avg_goals_last5_away",
    "avg_goals_last15_away",
    "avg_goals_conceded_last5_away",
    "avg_goals_conceded_last15_away",

    # --- xG rolling averages ---
    # Most predictive features — keep two windows
    "avg_xg_last5_home",
    "avg_xg_last15_home",
    "avg_xga_last5_home",
    "avg_xga_last15_home",
    "avg_xg_last5_away",
    "avg_xg_last15_away",
    "avg_xga_last5_away",
    "avg_xga_last15_away",

    # --- Over 2.5 rate ---
    # How often each team's games go over in recent windows
    "over_2_5_rate_last8_home",
    "over_2_5_rate_last8_away",
    "over_2_5_rate_last5_home",
    "over_2_5_rate_last5_away",

    # --- Win rate ---
    # Winning teams play more open football
    "win_rate_last8_home",
    "win_rate_last8_away",

    # --- xG overperformance ---
    # Mean reversion signal: high = scoring more than chances deserve
    "avg_xg_overperf_last5_home",
    "avg_xg_overperf_last5_away",

    # --- Shot quality ---
    # xG per shot — distinguishes chance quality from volume
    "avg_shot_quality_last5_home",
    "avg_shot_quality_last5_away",

    # --- Rest ---
    # Fatigue affects defensive organisation more than attack
    "days_rest_current_home",
    "days_rest_current_away",
    "days_rest_diff",

    # --- Combined attack strength ---
    # Sum of both teams' rolling xG/goals
    # Encodes total goal expectation directly — model doesn't have to infer addition
    "combined_xg_last5",
    "combined_xg_last15",
    "combined_goals_last5",

    # --- League baseline ---
    # Rolling league average goals — captures seasonal goal rate drift
    "league_avg_goals_last30",

    # --- Head-to-head ---
    # Average total goals in last 5 meetings between this exact pairing
    "h2h_avg_goals_last5",

    # --- Form momentum ---
    # Short-term deviation from baseline — captures heating up / cooling down
    "combined_xg_momentum",
    "combined_goals_momentum",

    # --- Referee ---
    # Some referees consistently produce more or fewer goals
    "ref_over_rate_last10_home",
    "ref_over_rate_last20_home",
    "ref_foul_rate_last20_home",
]


def validate_feature_cols(df, label: str = "") -> None:
    """
    Check that all FEATURE_COLS exist in df and none are duplicated.
    Call this after loading data to catch issues early.

    Args:
        df:    DataFrame to validate against
        label: optional label for the error message e.g. "backtest load_data"
    """
    prefix = f"[{label}] " if label else ""

    # Check for duplicates in the list itself
    seen = set()
    dupes = []
    for col in FEATURE_COLS:
        if col in seen:
            dupes.append(col)
        seen.add(col)
    if dupes:
        raise ValueError(
            f"{prefix}Duplicate entries in FEATURE_COLS: {dupes}. "
            f"Remove the duplicates from core/features.py."
        )

    # Check all cols exist in the DataFrame
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{prefix}Missing columns in DataFrame: {missing}. "
            f"Check that rolling_stats.py and build_dataset.py are up to date."
        )

    # Check none of the selected columns are DataFrames (duplicate col names in df)
    import pandas as pd
    bad = [c for c in FEATURE_COLS if isinstance(df[c], pd.DataFrame)]
    if bad:
        raise ValueError(
            f"{prefix}These columns returned a DataFrame instead of a Series "
            f"(duplicate column names in df): {bad}."
        )