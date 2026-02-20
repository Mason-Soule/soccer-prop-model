import psycopg2
import pandas as pd

# --- Connect to PostgreSQL ---
conn = psycopg2.connect(
    dbname="player_props",
    user="grownp",
    password="Mase3806",
    host="localhost",
    port="5432"
)

cursor = conn.cursor()

# --- Load CSV ---
df = pd.read_csv("data/raw/epl/2022-2023/match_data.csv")

# --- Insert Teams ---
teams = set(df["HomeTeam"]).union(set(df["AwayTeam"]))

for team in teams:
    cursor.execute(
        """
        INSERT INTO teams (name)
        VALUES (%s)
        ON CONFLICT (name) DO NOTHING;
        """,
        (team,)
    )

conn.commit()

# --- Build Team Lookup ---
cursor.execute("SELECT id, name FROM teams;")
team_lookup = {name: id for id, name in cursor.fetchall()}

# --- Insert Matches ---
for _, row in df.iterrows():
    cursor.execute(
        """
        INSERT INTO matches (
            match_date,
            season,
            home_team_id,
            away_team_id,
            home_goals,
            away_goals
        )
        VALUES (%s, %s, %s, %s, %s, %s);
        """,
        (
            pd.to_datetime(row["Date"], format="%d/%m/%y").date(),
            "2022-23",
            team_lookup[row["HomeTeam"]],
            team_lookup[row["AwayTeam"]],
            row["FTHG"],
            row["FTAG"]
        )
    )

conn.commit()
cursor.close()
conn.close()

print("Matches loaded successfully.")