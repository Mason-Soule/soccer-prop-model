import psycopg2
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()

# --- Connect to PostgreSQL ---
conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host="localhost",
    port="5432"
)

year = "2025-26"

cursor = conn.cursor()

# --- Load CSV ---
df = pd.read_csv(f"data/raw/epl/{year}/match_data.csv")

# --- Insert Teams ---
teams = set(df["HomeTeam"]).union(set(df["AwayTeam"]))

for team in teams:
    cursor.execute(
        """
        INSERT INTO teams (name, league)
        VALUES (%s, %s)
        ON CONFLICT (name, league) DO NOTHING;
        """,
        (team, "EPL")
    )

conn.commit()

# --- Build Team Lookup ---
cursor.execute("SELECT team_id, name FROM teams;")
team_lookup = {name: id for id, name in cursor.fetchall()}

# --- Insert Matches ---
for _, row in df.iterrows():
    for fmt in ("%d/%m/%y", "%d/%m/%Y"):
        try:
            match_date = pd.to_datetime(row["Date"], format=fmt).date()
            break
        except ValueError:
            continue
    home_id = team_lookup[row["HomeTeam"]]
    away_id = team_lookup[row["AwayTeam"]]
    cursor.execute(
        """
        INSERT INTO matches (
            date,
            home_team_id,
            away_team_id,
            season,
            league,
            referee
        )
        VALUES (%s, %s, %s, %s, %s, %s);
        """,
        (
            match_date,
            team_lookup[row["HomeTeam"]],
            team_lookup[row["AwayTeam"]],
            year,
            "EPL",
            row["Referee"]
        )
    )
    cursor.execute(
    """
    SELECT match_id
    FROM matches
    WHERE date = %s
      AND home_team_id = %s
      AND away_team_id = %s;
    """,
    (
        match_date,
        team_lookup[row["HomeTeam"]],
        team_lookup[row["AwayTeam"]],
    )
    )

    match_id = cursor.fetchone()[0]

    # --- Insert Home Team Stats ---
    cursor.execute(
        """
        INSERT INTO team_match_stats (
            team_id,
            match_id,
            is_home,
            goals,
            shots,
            shots_on_target,
            fouls,
            corners,
            yellow_cards,
            red_cards
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """,
        (
            home_id,
            match_id,
            True,
            row["FTHG"],
            row["HS"],
            row["HST"],
            row["HF"],
            row["HC"],
            row["HY"],
            row["HR"]
        )
    )

    # --- Insert Away Team Stats ---
    cursor.execute(
        """
        INSERT INTO team_match_stats (
            team_id,
            match_id,
            is_home,
            goals,
            shots,
            shots_on_target,
            fouls,
            corners,
            yellow_cards,
            red_cards
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """,
        (
            away_id,
            match_id,
            False,
            row["FTAG"],
            row["AS"],
            row["AST"],
            row["AF"],
            row["AC"],
            row["AY"],
            row["AR"]
        )
    )

conn.commit()
cursor.close()
conn.close()

print("Matches loaded successfully.")