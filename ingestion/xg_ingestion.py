"""
xg_ingestion.py
---------------
Downloads historical EPL xG data from Understat and writes it into
your existing team_match_stats table as `xg` and `xga` columns.

Prerequisites:
    1. Add columns to your DB first:
       ALTER TABLE team_match_stats ADD COLUMN IF NOT EXISTS xg FLOAT;
       ALTER TABLE team_match_stats ADD COLUMN IF NOT EXISTS xga FLOAT;

    2. Install dependency:
       pip install understatapi

Usage:
    python ingestion/xg_ingestion.py

After running, add "xg" and "xga" to DEFAULT_STATS in rolling_stats.py
and add the new feature columns to FEATURE_COLS in train.py.
"""

import sys
import logging
import time
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import os

load_dotenv()

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from config.leagues.epl import EPL

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Season config
# Understat season parameter is the START year of the season
# e.g. 2016 = 2016/17 season
# ---------------------------------------------------------------------------

SEASONS = EPL.understat_seasons

# ---------------------------------------------------------------------------
# Team name map — Understat full name -> your DB name
#
# Understat uses full English names with underscores for spaces.
# Your DB uses the same short names as football-data.co.uk.
# ---------------------------------------------------------------------------
UNDERSTAT_TO_DB = {
    "Manchester_United":      "Man United",
    "Manchester_City":        "Man City",
    "Tottenham":              "Tottenham",
    "Arsenal":                "Arsenal",
    "Chelsea":                "Chelsea",
    "Liverpool":              "Liverpool",
    "Leicester":              "Leicester",
    "Everton":                "Everton",
    "West_Ham":               "West Ham",
    "Southampton":            "Southampton",
    "Crystal_Palace":         "Crystal Palace",
    "Burnley":                "Burnley",
    "Watford":                "Watford",
    "West_Bromwich_Albion":   "West Brom",
    "Stoke":                  "Stoke",
    "Swansea":                "Swansea",
    "Hull":                   "Hull",
    "Middlesbrough":          "Middlesbrough",
    "Sunderland":             "Sunderland",
    "Bournemouth":            "Bournemouth",
    "Newcastle_United":       "Newcastle",
    "Brighton":               "Brighton",
    "Huddersfield":           "Huddersfield",
    "Cardiff":                "Cardiff",
    "Fulham":                 "Fulham",
    "Wolves":                 "Wolves",
    "Norwich":                "Norwich",
    "Sheffield_United":       "Sheffield United",
    "Leeds":                  "Leeds",
    "Aston_Villa":            "Aston Villa",
    "Brentford":              "Brentford",
    "Nottingham_Forest":      "Nott'm Forest",
    "Luton":                  "Luton",
    "Ipswich":                "Ipswich",
}


# ---------------------------------------------------------------------------
# Download xG from Understat
# ---------------------------------------------------------------------------

def fetch_season_xg(season: int) -> pd.DataFrame:
    """
    Fetch all EPL match xG for a single season from Understat.

    Uses the league endpoint which returns one row per match with:
        h (home team name), a (away team name),
        xg (home xG), xga (away xG), datetime, isResult

    Args:
        season: Start year of season (e.g. 2016 for 2016/17)

    Returns:
        DataFrame with columns:
            date, home_team, away_team, home_xg, away_xg
    """
    from understatapi import UnderstatClient

    logger.info(f"  Fetching {season}/{str(season+1)[2:]} xG from Understat...")

    with UnderstatClient() as understat:
        raw = understat.league(league="EPL").get_match_data(season=str(season))

    df = pd.DataFrame(raw)

    # Filter to completed matches only
    df = df[df["isResult"].astype(str).str.lower() == "true"].copy()

    # Parse relevant fields
    df["date"]      = pd.to_datetime(df["datetime"]).dt.date
    df["home_team"] = df["h"].apply(lambda x: x["title"] if isinstance(x, dict) else x)
    df["away_team"] = df["a"].apply(lambda x: x["title"] if isinstance(x, dict) else x)
    df["home_xg"] = pd.to_numeric(df["xG"].apply(lambda x: x["h"]), errors="coerce")
    df["away_xg"] = pd.to_numeric(df["xG"].apply(lambda x: x["a"]), errors="coerce")

    # Normalize team names to match your DB
    df["home_team"] = df["home_team"].map(
        lambda x: UNDERSTAT_TO_DB.get(x.replace(" ", "_"), x)
    )
    df["away_team"] = df["away_team"].map(
        lambda x: UNDERSTAT_TO_DB.get(x.replace(" ", "_"), x)
    )

    df = df[["date", "home_team", "away_team", "home_xg", "away_xg"]].copy()
    df["season"] = f"{season}-{str(season+1)[2:]}"

    logger.info(f"  {season}/{str(season+1)[2:]}: {len(df)} matches fetched")
    return df


def fetch_all_seasons(seasons: list = SEASONS) -> pd.DataFrame:
    """Download xG for all seasons and return combined DataFrame."""
    logger.info("Fetching xG data from Understat...")
    dfs = []

    for season in seasons:
        try:
            df = fetch_season_xg(season)
            dfs.append(df)
            # Be polite to Understat — avoid hammering their server
            time.sleep(1)
        except Exception as e:
            logger.warning(f"  Failed to fetch {season}: {e}")
            continue

    if not dfs:
        raise RuntimeError("No xG data fetched. Check network or Understat availability.")

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total matches fetched: {len(combined)}")
    return combined


# ---------------------------------------------------------------------------
# Match Understat rows to your DB match IDs
# ---------------------------------------------------------------------------

def build_match_lookup(engine) -> pd.DataFrame:
    """
    Load matches + team names from your DB to create a lookup table.

    Returns DataFrame with:
        match_id, date, home_team, away_team, home_team_id, away_team_id
    """
    query = """
    SELECT
        m.match_id,
        m.date::date AS date,
        ht.name AS home_team,
        at.name AS away_team,
        m.home_team_id,
        m.away_team_id
    FROM matches m
    JOIN teams ht ON m.home_team_id = ht.team_id
    JOIN teams at ON m.away_team_id = at.team_id
    ORDER BY m.date
    """
    df = pd.read_sql(query, engine)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def merge_xg_with_matches(
    xg_df: pd.DataFrame,
    match_lookup: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join Understat xG onto your DB matches by date + team names.

    Returns DataFrame with:
        match_id, home_team_id, away_team_id, home_xg, away_xg
    """
    merged = xg_df.merge(
        match_lookup,
        on=["date", "home_team", "away_team"],
        how="inner",
    )

    matched = len(merged)
    total   = len(xg_df)
    pct     = matched / total * 100 if total > 0 else 0

    logger.info(f"xG merge: {matched}/{total} matches joined ({pct:.1f}%)")

    if pct < 90:
        # Find unmatched rows to help debug team name issues
        unmatched = xg_df[
            ~xg_df.set_index(["date", "home_team", "away_team"]).index.isin(
                merged.set_index(["date", "home_team", "away_team"]).index
            )
        ]
        unmatched_teams = sorted(
            set(unmatched["home_team"].tolist() + unmatched["away_team"].tolist())
        )
        logger.warning(f"Unmatched team names in xG data: {unmatched_teams}")
        logger.warning("Add missing names to UNDERSTAT_TO_DB map and rerun.")

    return merged[["match_id", "home_team_id", "away_team_id", "home_xg", "away_xg"]]


# ---------------------------------------------------------------------------
# Write xG to team_match_stats
# ---------------------------------------------------------------------------

def write_xg_to_db(merged: pd.DataFrame, engine) -> None:
    """
    Update team_match_stats with xg and xga values.

    For each match:
        - home team row gets xg = home_xg, xga = away_xg
        - away team row gets xg = away_xg, xga = home_xg

    Uses INSERT ... ON CONFLICT DO UPDATE (upsert) so reruns are safe.
    """
    updated = 0

    with engine.begin() as conn:
        for _, row in merged.iterrows():
            # Update home team row
            conn.execute(text("""
                UPDATE team_match_stats
                SET xg  = :xg,
                    xga = :xga
                WHERE match_id = :match_id
                  AND team_id  = :team_id
            """), {
                "xg":       float(row["home_xg"]),
                "xga":      float(row["away_xg"]),
                "match_id": int(row["match_id"]),
                "team_id":  int(row["home_team_id"]),
            })

            # Update away team row
            conn.execute(text("""
                UPDATE team_match_stats
                SET xg  = :xg,
                    xga = :xga
                WHERE match_id = :match_id
                  AND team_id  = :team_id
            """), {
                "xg":       float(row["away_xg"]),
                "xga":      float(row["home_xg"]),
                "match_id": int(row["match_id"]),
                "team_id":  int(row["away_team_id"]),
            })

            updated += 1

    logger.info(f"Updated xG for {updated} matches ({updated * 2} team rows)")


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------

def verify_xg_coverage(engine) -> None:
    """Print coverage stats to confirm xG was written correctly."""
    query = """
    SELECT
        m.season,
        COUNT(*) AS total_rows,
        COUNT(tms.xg) AS rows_with_xg,
        ROUND(AVG(tms.xg)::numeric, 3) AS avg_xg,
        ROUND(AVG(tms.xga)::numeric, 3) AS avg_xga
    FROM team_match_stats tms
    JOIN matches m ON tms.match_id = m.match_id
    GROUP BY m.season
    ORDER BY m.season
    """
    df = pd.read_sql(query, engine)
    print("\n--- xG Coverage by Season ---")
    print(df.to_string(index=False))

    total     = df["total_rows"].sum()
    with_xg   = df["rows_with_xg"].sum()
    pct       = with_xg / total * 100 if total > 0 else 0
    print(f"\nOverall: {with_xg}/{total} rows have xG ({pct:.1f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_xg_ingestion() -> None:
    """Full pipeline: fetch from Understat → match to DB → write xG."""

    engine = create_engine(
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@localhost:5432/{os.getenv('DB_NAME')}"
    )

    # Step 1 — fetch xG from Understat
    xg_df = fetch_all_seasons()

    # Step 2 — load your DB match lookup
    logger.info("Loading match lookup from DB...")
    match_lookup = build_match_lookup(engine)

    # Step 3 — join xG onto your matches
    merged = merge_xg_with_matches(xg_df, match_lookup)

    if len(merged) == 0:
        logger.error("No matches joined — check team name mapping and DB connection.")
        return

    # Step 4 — write to DB
    logger.info("Writing xG to team_match_stats...")
    write_xg_to_db(merged, engine)

    # Step 5 — verify
    verify_xg_coverage(engine)

    logger.info("\nDone. Next steps:")


if __name__ == "__main__":
    run_xg_ingestion()