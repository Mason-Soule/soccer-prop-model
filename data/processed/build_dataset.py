import pandas as pd
import numpy as np
import sys
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
load_dotenv()

from pathlib import Path
project_root = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(project_root))

from features.rolling_stats import add_rolling_averages, add_venue_rolling_averages, add_derived_features


def build_match_level_df(df):
    home = df[df["is_home"] == True].copy()
    away = df[df["is_home"] == False].copy()

    # Drop match-level columns from the away side before merging.
    # These columns are identical for both sides of the same match,
    # so keeping both creates _home/_away duplicates that break XGBoost.
    match_level_cols = ["league_avg_goals_last30"]
    away = away.drop(columns=[c for c in match_level_cols if c in away.columns])

    match_df = home.merge(
        away,
        on="match_id",
        suffixes=("_home", "_away")
    )

    # Rename the _home suffix off match-level columns — they're not home-specific
    for col in match_level_cols:
        if f"{col}_home" in match_df.columns:
            match_df = match_df.rename(columns={f"{col}_home": col})



    # Target should be identical from either side
    match_df["over_2_5"] = match_df["over_2_5_home"]
    match_df["season_start"] = match_df["season_home"].str[:4].astype(int)

    # --- Match-level derived features ---
    # These are computed after the home/away merge since they combine both sides

    # Days rest differential — positive means home team more rested
    match_df["days_rest_diff"] = (
        match_df["days_rest_home"] - match_df["days_rest_away"]
    )

    # --- Combined attack features ---
    # Sum of both teams' rolling xG/goals — a direct proxy for total goals
    # More predictive than using home + away separately because the model
    # no longer has to discover the additive relationship itself.
    match_df["combined_xg_last5"] = (
        match_df["avg_xg_last5_home"] + match_df["avg_xg_last5_away"]
    )
    match_df["combined_xg_last15"] = (
        match_df["avg_xg_last15_home"] + match_df["avg_xg_last15_away"]
    )
    match_df["combined_goals_last5"] = (
        match_df["avg_goals_last5_home"] + match_df["avg_goals_last5_away"]
    )

# --- Head-to-head average goals ---
    # Average total goals in the last 5 meetings between this exact pairing.
    # fixture_key sorts both team names so A-vs-B and B-vs-A are the same key.
    # We iterate sorted by date and look back — no leakage.
    match_df = match_df.sort_values("date_home").reset_index(drop=True)
    match_df = match_df.loc[:, ~match_df.columns.duplicated()]

    match_df["total_goals_h2h"] = (
        match_df["goals_home"] + match_df["goals_away"]
    )
    match_df["fixture_key"] = match_df.apply(
        lambda r: tuple(sorted([r["team_name_home"], r["team_name_away"]])),
        axis=1,
    )

    h2h_vals = []
    # Build a dict of fixture -> list of past total goals as we walk forward
    fixture_history: dict = {}

    for idx, row in match_df.iterrows():
        key = row["fixture_key"]
        history = fixture_history.get(key, [])

        if len(history) >= 2:
            h2h_vals.append(np.mean(history[-5:]))
        else:
            h2h_vals.append(np.nan)

        # Append current match result AFTER recording the lookback value
        # so the current match never appears in its own average
        fixture_history[key] = history + [row["total_goals_h2h"]]

    match_df["h2h_avg_goals_last5"] = h2h_vals
    match_df = match_df.drop(columns=["total_goals_h2h", "fixture_key"])

    return match_df


def build_dataframe():
    query = """
    SELECT
        m.match_id,
        m.date,
        m.season,
        m.league,
        m.referee,

        tms.team_id,
        t.name AS team_name,
        tms.is_home,

        -- Team stats
        tms.goals,
        tms.shots,
        tms.shots_on_target,
        tms.fouls,
        tms.corners,
        tms.yellow_cards,
        tms.red_cards,
        tms.xg,
        tms.xga,

        -- Opponent stats
        opp.team_id AS opponent_id,
        opp.goals AS goals_conceded,
        opp.shots AS shots_conceded,
        opp.shots_on_target AS shots_on_target_conceded,
        opp.fouls AS fouls_drawn,
        opp.corners AS corners_conceded,
        opp.yellow_cards AS opponent_yellow_cards,
        opp.red_cards AS opponent_red_cards

    FROM team_match_stats tms

    JOIN team_match_stats opp
        ON tms.match_id = opp.match_id
        AND tms.team_id != opp.team_id

    JOIN matches m
        ON tms.match_id = m.match_id

    JOIN teams t
        ON tms.team_id = t.team_id

    ORDER BY tms.team_id, m.date;
    """

    engine = create_engine(
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@localhost:5432/{os.getenv('DB_NAME')}"
    )
    df = pd.read_sql(query, engine)

    # Sort by team and date — required for all rolling calculations
    df = df.sort_values(["team_id", "date"])

    # --- Base target and derived columns ---
    df["goal_diff"]  = df["goals"] - df["goals_conceded"]
    df["win"]        = (df["goal_diff"] > 0).astype(int)
    df["over_2_5"]   = ((df["goals"] + df["goals_conceded"]) > 2).astype(int)

    # xG overperformance — positive means scoring more than chances deserved
    # Used as a mean-reversion signal: high values predict future regression
    df["xg_overperformance"] = df["goals"] - df["xg"]

    # Shot quality — xG per shot, measures chance quality not just volume
    df["shot_quality"] = df["xg"] / df["shots"].replace(0, np.nan)

    # Shot quality conceded — how dangerous the chances the team gives up are
    df["shot_quality_conceded"] = df["xga"] / df["shots_conceded"].replace(0, np.nan)

    # Days rest — how many days since the team's last match
    # Shift(1) so we don't use the current match date
    df["days_rest"] = (
        df.groupby("team_id")["date"]
          .transform(lambda s: s.diff().dt.days)
    )

    # --- Rolling averages for all standard stats + xG ---
    df = add_rolling_averages(df)

    # --- Derived rolling features ---
    # These are computed after add_rolling_averages so all avg_ columns exist
    df = add_derived_features(df)

    # Drop rows where core rolling features are NaN
    # (first N games of each team's history)
    core_cols = ["avg_goals_last15", "avg_goals_conceded_last15"]
    df = df.dropna(subset=core_cols)

    # --- League-wide rolling average goals ---
    # Computed across all matches (not per team) to capture era-level goal rates.
    # We sort by date, deduplicate to one row per match (using only home rows
    # to avoid double-counting), compute the rolling mean, then join back.
    # shift(1) ensures we never include the current match in its own average.
    match_goals = (
        df[df["is_home"] == True][["match_id", "date", "goals", "goals_conceded"]]
        .copy()
        .sort_values("date")
        .reset_index(drop=True)
    )
    match_goals["total_goals"] = match_goals["goals"] + match_goals["goals_conceded"]
    match_goals["league_avg_goals_last30"] = (
        match_goals["total_goals"]
        .shift(1)
        .rolling(window=30, min_periods=15)
        .mean()
    )
    # Join the league average back onto all rows (home + away) by match_id
    df = df.merge(
        match_goals[["match_id", "league_avg_goals_last30"]],
        on="match_id",
        how="left",
    )


    return build_match_level_df(df)