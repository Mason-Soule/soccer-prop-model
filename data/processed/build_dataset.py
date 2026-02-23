import pandas as pd
import sys
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
load_dotenv()

# Get the parent directory (your_project/)
# Current file: data/processed/build_dataset.py
# Go up two levels: data/processed/ -> data/ -> your_project/
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Or using pathlib (cleaner)
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.absolute()

# Add to Python path
sys.path.insert(0, str(project_root))

# Now import from features (sibling directory)
from features.rolling_stats import avg_goals_last5, avg_goals_conceded_last5, avg_shots_last5, avg_shots_conceded_last5

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

    match_df["season_start"] = df["season"].str[:4].astype(int)

    #current attack and defense form of both teams
    match_df["total_attack_form"] = (match_df["avg_goals_last5_home"] + match_df["avg_goals_last5_away"])
    match_df["total_defense_form"] = (match_df["avg_goals_conceded_last5_home"] + match_df["avg_goals_conceded_last5_away"])
    return match_df

def build_dataframe():
    query = """
    SELECT
        m.match_id,
        m.date,
        m.season,
        m.league,

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

    #sorts by team id and date
    df = df.sort_values(["team_id", "date"])

    #adds goal difference, win/loss, and over 2.5 goals
    df["goal_diff"] = df["goals"] - df["goals_conceded"]
    df["win"] = (df["goal_diff"] > 0).astype(int)
    df["over_2_5"] = ((df["goals"] + df["goals_conceded"]) > 2).astype(int)

    #window = 5 by default
    #Add rolling goals to the data frame and drope first 5 games of each team
    df = avg_goals_last5(df)
    df = avg_goals_conceded_last5(df)
    df = avg_shots_last5(df)
    df = avg_shots_conceded_last5(df)
    df = df.dropna()

    return build_match_level_df(df)
    
    

