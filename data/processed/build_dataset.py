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

    match_df = home.merge(
        away,
        on="match_id",
        suffixes=("_home", "_away")
    )

    # Target should be identical from either side
    match_df["over_2_5"] = match_df["over_2_5_home"]
    match_df["season_start"] = match_df["season_home"].str[:4].astype(int)

    # --- Match-level derived features ---
    # These are computed after the home/away merge since they combine both sides

    # Days rest differential — positive means home team more rested
    match_df["days_rest_diff"] = (
        match_df["days_rest_home"] - match_df["days_rest_away"]
    )

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

    return build_match_level_df(df)


