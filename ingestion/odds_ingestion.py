"""
odds_ingestion.py
-----------------
Downloads EPL odds from football-data.co.uk, strips bookmaker vig,
and merges implied probabilities onto the match-level DataFrame
produced by build_dataset.py.

Your existing pipeline produces a match-level df where home/away columns
are suffixed (_home, _away) after the merge in build_match_level_df().
This module joins onto: date_home, team_name_home, team_name_away.

Usage:
    from ingestion.odds_ingestion import load_odds, merge_odds_with_match_df

    odds_df     = load_odds()
    full_df     = merge_odds_with_match_df(match_df, odds_df)

Or run standalone to download and save odds CSV:
    python ingestion/odds_ingestion.py
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import requests
from io import StringIO
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Path setup (mirrors pattern in build_dataset.py)
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
# Season config — mirrors the seasons you load via load_matches.py
# ---------------------------------------------------------------------------
SEASONS  = EPL.fd_seasons
BASE_URL = "https://www.football-data.co.uk/mmz4281/{code}/" + EPL.fd_division + ".csv"

# Pinnacle closing odds are the sharpest signal.
# PSC columns only appear consistently from ~2019-20 onward.
# For earlier seasons the pipeline automatically falls back to BbAv (avg market).
OVER_COLS_PRIORITY  = ["B365C>2.5", "BbAv>2.5", "B365>2.5"]
UNDER_COLS_PRIORITY = ["B365C<2.5", "BbAv<2.5", "B365<2.5"]

# Save path mirrors your data/raw/epl/<season>/ structure
RAW_DIR = project_root / "data" / "raw" / "epl"

# ---------------------------------------------------------------------------
# Team name map — football-data.co.uk name -> your DB name
# ---------------------------------------------------------------------------
# Your DB names came from football-data.co.uk CSVs so they already match
# for most teams. These mappings cover cases where football-data.co.uk
# uses different names across seasons (e.g. "Nott'm Forest" vs "Nottm Forest").
# If audit_name_mismatches() reports new mismatches, extend this map.

TEAM_NAME_MAP = EPL.fd_team_map

# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _download_season(season_label: str, season_code: str) -> pd.DataFrame | None:
    """Download one season CSV from football-data.co.uk."""
    url = BASE_URL.format(code=season_code)
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        df = pd.concat([df, pd.Series([season_label] * len(df), name="season")], axis=1)
        logger.info(f"  {season_label}: {len(df)} rows")
        return df
    except requests.RequestException as e:
        logger.warning(f"  {season_label} download failed: {e}")
        return None


def download_all_seasons(seasons: dict = SEASONS) -> pd.DataFrame:
    """Download all seasons and return a single concatenated raw DataFrame."""
    logger.info("Downloading odds from football-data.co.uk...")
    dfs = [_download_season(label, code) for label, code in seasons.items()]
    dfs = [d for d in dfs if d is not None]

    if not dfs:
        raise RuntimeError("No seasons downloaded. Check network connection.")

    raw = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total raw rows: {len(raw)}")
    return raw


# ---------------------------------------------------------------------------
# Vig removal
# ---------------------------------------------------------------------------

def remove_vig(over_odds: pd.Series, under_odds: pd.Series) -> pd.DataFrame:
    """
    Strip bookmaker margin from raw decimal odds.

    Raw implied probabilities (1/odds) sum to > 1.0 — the overround.
    Dividing each side by the total normalizes them to sum to exactly 1.0,
    giving the market's true probability estimate for each outcome.

    Returns DataFrame with: market_prob_over, market_prob_under, overround
    """
    raw_over  = 1.0 / over_odds
    raw_under = 1.0 / under_odds
    overround = raw_over + raw_under

    return pd.DataFrame({
        "market_prob_over":  raw_over  / overround,
        "market_prob_under": raw_under / overround,
        "overround":         overround,
    })


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def process_odds(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw odds data and compute vig-free implied probabilities.

    Odds columns vary by season on football-data.co.uk — earlier seasons
    may lack BbAv but have B365, or vice versa. This function resolves
    the best available odds on a row-by-row basis across the priority list
    so no rows are dropped purely due to a missing preferred column.

    Output columns:
        date, home_team, away_team, season,
        odds_over_2_5, odds_under_2_5,
        market_prob_over, market_prob_under, overround, odds_source
    """
    df = raw.copy()

    # --- Dates ---
    # football-data.co.uk uses mixed DD/MM/YY and DD/MM/YYYY across seasons
    # Try both formats explicitly to avoid dateutil fallback warning
    def parse_dates(series):
        for fmt in ("%d/%m/%y", "%d/%m/%Y"):
            parsed = pd.to_datetime(series, format=fmt, errors="coerce")
            if parsed.notna().mean() > 0.9:
                return parsed
        return pd.to_datetime(series, dayfirst=True, errors="coerce")

    df["Date"] = parse_dates(df["Date"])
    n_before = len(df)
    df = df.dropna(subset=["Date"])
    if len(df) < n_before:
        logger.warning(f"Dropped {n_before - len(df)} rows: unparseable dates")

    # --- Rename to match your build_dataset column naming style ---
    df = df.rename(columns={
        "Date":     "date",
        "HomeTeam": "home_team",
        "AwayTeam": "away_team",
    })

    # --- Standardize team names to match your DB ---
    df["home_team"] = df["home_team"].replace(TEAM_NAME_MAP)
    df["away_team"] = df["away_team"].replace(TEAM_NAME_MAP)

    # --- Resolve best available odds row-by-row ---
    # The problem: BbAv>2.5 exists as a column in all seasons but is NaN
    # for earlier seasons. Picking one column globally drops those rows.
    # Instead coalesce across the priority list — first non-null wins per row.
    available_over  = [c for c in OVER_COLS_PRIORITY  if c in df.columns]
    available_under = [c for c in UNDER_COLS_PRIORITY if c in df.columns]

    if not available_over or not available_under:
        raise ValueError(
            f"No usable over/under odds columns found. "
            f"Available columns: {list(df.columns)}"
        )

    logger.info(f"Over cols available:  {available_over}")
    logger.info(f"Under cols available: {available_under}")

    # Convert all candidate columns to numeric first
    for c in available_over + available_under:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Coalesce: bfill across axis=1 then take first column = first non-null per row
    df["odds_over_2_5"]  = df[available_over].bfill(axis=1).iloc[:, 0]
    df["odds_under_2_5"] = df[available_under].bfill(axis=1).iloc[:, 0]

    # Track which source was used per row
    def source_col(row, cols):
        for c in cols:
            if pd.notna(row[c]):
                return c
        return None

    df["odds_source"] = df.apply(lambda r: source_col(r, available_over), axis=1)

    logger.info("Odds source breakdown by season:\n" +
                df.groupby(["season", "odds_source"]).size().to_string())

    # --- Drop missing / invalid odds ---
    n_before = len(df)
    df = df.dropna(subset=["odds_over_2_5", "odds_under_2_5"])
    invalid = (df["odds_over_2_5"] <= 1.0) | (df["odds_under_2_5"] <= 1.0)
    df = df[~invalid]
    dropped = n_before - len(df)
    if dropped:
        logger.warning(f"Dropped {dropped} rows: missing or invalid odds")

    # --- Vig removal ---
    vig = remove_vig(df["odds_over_2_5"], df["odds_under_2_5"])
    df  = pd.concat([df.reset_index(drop=True), vig], axis=1)

    # --- Keep only columns needed downstream ---
    keep = [
        "date", "season", "home_team", "away_team",
        "odds_over_2_5", "odds_under_2_5",
        "market_prob_over", "market_prob_under",
        "overround", "odds_source",
    ]
    df = df[[c for c in keep if c in df.columns]]
    df = df.sort_values("date").reset_index(drop=True)

    logger.info(f"Clean odds rows: {len(df)}")
    logger.info(f"Avg overround: {df['overround'].mean():.4f}  "
                f"(Pinnacle is ~1.02, soft books ~1.05-1.07)")

    return df


# ---------------------------------------------------------------------------
# Merge onto your existing match-level DataFrame
# ---------------------------------------------------------------------------

def merge_odds_with_match_df(
    match_df: pd.DataFrame,
    odds_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left-join odds onto the DataFrame produced by build_match_level_df().

    Your build_match_level_df() merges home/away rows using suffixes
    (_home, _away), so the relevant join keys are:
        date_home      <- match date
        team_name_home <- home team name
        team_name_away <- away team name

    Adds these columns to match_df:
        odds_over_2_5    : decimal odds for over 2.5 (use for Kelly staking)
        odds_under_2_5   : decimal odds for under 2.5
        market_prob_over : vig-free market implied P(over 2.5)
        market_prob_under: vig-free market implied P(under 2.5)
        overround        : bookmaker margin (1.0 = no margin)
        odds_source      : which odds column was used (PSC vs BbAv)

    Args:
        match_df: Output of build_dataframe() from build_dataset.py
        odds_df:  Output of process_odds() or load_odds()

    Returns:
        match_df with odds columns appended. Unmatched rows keep NaN in
        odds columns — run audit_name_mismatches() if match rate is low.
    """
    match_df = match_df.copy()
    odds_df  = odds_df.copy()

    # psycopg2 returns timezone-aware Python datetime objects stored as object
    # dtype. Converting to UTC then extracting .date() strips both the timezone
    # and the time component, leaving a clean calendar date on both sides.
    # odds dates are already midnight naive timestamps so .dt.date works directly.
    match_df["date_home"] = pd.to_datetime(
        pd.to_datetime(match_df["date_home"], utc=True).dt.date
    )
    odds_df["date"] = pd.to_datetime(
        pd.to_datetime(odds_df["date"]).dt.date
    )

    merged = match_df.merge(
        odds_df,
        left_on=["date_home", "team_name_home", "team_name_away"],
        right_on=["date",      "home_team",      "away_team"],
        how="left",
    )

    # Drop redundant keys brought in from odds_df
    merged = merged.drop(columns=["date", "home_team", "away_team"], errors="ignore")

    matched = merged["market_prob_over"].notna().sum()
    total   = len(merged)
    pct     = matched / total * 100

    logger.info(f"Odds merge: {matched}/{total} rows matched ({pct:.1f}%)")

    if pct < 95:
        logger.warning(
            f"Match rate below 95% — likely team name mismatches. "
            f"Run audit_name_mismatches(match_df, odds_df) to inspect."
        )

    return merged


# ---------------------------------------------------------------------------
# Debugging helper
# ---------------------------------------------------------------------------

def audit_name_mismatches(
    match_df: pd.DataFrame,
    odds_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Find team names in your match_df that don't appear in odds_df.

    Since both datasets originate from football-data.co.uk CSVs loaded
    via load_matches.py, mismatches should be rare — but can occur if
    team names were manually edited in the DB.

    Returns a DataFrame of unmatched names so you can update your DB
    or add a rename mapping before merging.
    """
    match_teams = (
        set(match_df["team_name_home"].dropna()) |
        set(match_df["team_name_away"].dropna())
    )
    odds_teams = (
        set(odds_df["home_team"].dropna()) |
        set(odds_df["away_team"].dropna())
    )

    unmatched = sorted(match_teams - odds_teams)

    if not unmatched:
        logger.info("No team name mismatches — all names align.")
        return pd.DataFrame()

    logger.warning(f"{len(unmatched)} unmatched team names: {unmatched}")
    return pd.DataFrame({"unmatched_in_match_df": unmatched})


# ---------------------------------------------------------------------------
# Save / Load helpers
# ---------------------------------------------------------------------------

def save_odds(df: pd.DataFrame, path: Path = None) -> Path:
    """Save processed odds to CSV. Mirrors your data/raw/epl/ structure."""
    if path is None:
        path = RAW_DIR / "odds_processed.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Saved odds to {path}")
    return path


def load_odds(path: Path = None) -> pd.DataFrame:
    """
    Load processed odds CSV if it exists, otherwise download and process.
    Avoids re-downloading on every run.
    """
    if path is None:
        path = RAW_DIR / "odds_processed.csv"

    if path.exists():
        logger.info(f"Loading cached odds from {path}")
        df = pd.read_csv(path, parse_dates=["date"])
        return df

    logger.info("No cached odds found — downloading now...")
    raw = download_all_seasons()
    df  = process_odds(raw)
    save_odds(df, path)
    return df


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    odds = load_odds()

    print("\n--- Sample rows ---")
    print(odds.head())

    print("\n--- Column types ---")
    print(odds.dtypes)

    print("\n--- Implied probability stats ---")
    print(odds[["market_prob_over", "market_prob_under", "overround"]].describe())

    print("\n--- Odds source breakdown (PSC = Pinnacle, BbAv = avg market) ---")
    print(odds["odds_source"].value_counts())