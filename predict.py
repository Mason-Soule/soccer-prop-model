"""
predict.py
----------
Generates over 2.5 goals bet recommendations for upcoming EPL fixtures.

Workflow:
    1. Load all historical data (2014-2025) and train the model
    2. Fetch upcoming fixtures + live odds from The Odds API
    3. Normalise team names to match DB
    4. Build rolling features using recent 2025-26 form
    5. Predict and calculate edge
    6. Output bet shortlist filtered to calibrated edge zones

Usage:
    python predict.py

Requirements:
    ODDS_API_KEY in .env file
    2025-26 match data loaded into DB via load_matches.py
    2025-26 xG loaded via xg_ingestion.py
"""

import os
import sys
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
from xgboost import XGBClassifier

load_dotenv()

project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

from data.processed.build_dataset import build_dataframe
from ingestion.odds_ingestion import load_odds, merge_odds_with_match_df
from core.features import FEATURE_COLS
from core.model import train_model_full
from core.staking import is_value_bet, suggested_stake
from config.leagues.epl import EPL
from core.data_loader import load_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# Edge zones — calibrated from backtest analysis
# Bets outside these zones showed negative outperformance historically
MIN_ODDS       = 1.60
MAX_ODDS       = 2.00
KELLY_FRACTION = 0.15
KELLY_FRACTION_WIDE = 0.10 
STARTING_BANKROLL = 1000.0

# The Odds API config
ODDS_API_KEY    = os.getenv("ODDS_API_KEY")
ODDS_API_BASE   = "https://api.the-odds-api.com/v4"
SPORT_KEY       = "soccer_epl"
MARKETS         = "totals"       # over/under markets
REGIONS         = "eu"           # European decimal odds
ODDS_FORMAT     = "decimal"

# ---------------------------------------------------------------------------
# Team name mapping — The Odds API names → your DB names
# ---------------------------------------------------------------------------
ODDS_API_TO_DB = {
    "Manchester City":        "Man City",
    "Manchester United":      "Man United",
    "Newcastle United":       "Newcastle",
    "Tottenham Hotspur":      "Tottenham",
    "Wolverhampton Wanderers":"Wolves",
    "Nottingham Forest":      "Nott'm Forest",
    "West Ham United":        "West Ham",
    "Brighton & Hove Albion": "Brighton",
    "Leicester City":         "Leicester",
    "Leeds United":           "Leeds",
    "West Bromwich Albion":   "West Brom",
    "Sheffield United":       "Sheffield United",
    "Aston Villa":            "Aston Villa",
    "AFC Bournemouth":        "Bournemouth",
    "Ipswich Town":           "Ipswich",
    "Huddersfield Town":      "Huddersfield",
    "Cardiff City":           "Cardiff",
    "Hull City":              "Hull",
    "Swansea City":           "Swansea",
    "Norwich City":           "Norwich",
    "Stoke City":             "Stoke",
    "Burnley":                "Burnley",
    "Brentford":              "Brentford",
    "Fulham":                 "Fulham",
    "Crystal Palace":         "Crystal Palace",
    "Everton":                "Everton",
    "Southampton":            "Southampton",
    "Arsenal":                "Arsenal",
    "Chelsea":                "Chelsea",
    "Liverpool":              "Liverpool",
    "Sunderland":             "Sunderland",
    "Middlesbrough":          "Middlesbrough",
    "Watford":                "Watford",
    "Brighton and Hove Albion": "Brighton",
    "Sunderland AFC":           "Sunderland",
}


# ---------------------------------------------------------------------------
# Step 1 — Train model on all historical data
# ---------------------------------------------------------------------------
def train_model(df: pd.DataFrame) -> XGBClassifier:
    train_df = df[df["season_start"] <= 2024].copy()
    mask = train_df[FEATURE_COLS].isna().all(axis=1)
    train_df = train_df[~mask]
    X_train = train_df[FEATURE_COLS]
    y_train = train_df["over_2_5"]
    logger.info(f"Training on {len(train_df)} matches "
                f"({train_df['season_start'].min()}–{train_df['season_start'].max()})")
    model = train_model_full(X_train, y_train)
    logger.info("Model trained.")
    return model


# ---------------------------------------------------------------------------
# Step 2 — Fetch upcoming fixtures + odds from The Odds API
# ---------------------------------------------------------------------------
def fetch_upcoming_odds() -> pd.DataFrame:
    """
    Fetch upcoming EPL fixtures with over/under 2.5 odds from The Odds API.

    Returns DataFrame with columns:
        fixture_id, date, home_team, away_team,
        odds_over_2_5, odds_under_2_5, market_prob_over, market_prob_under
    """
    if not ODDS_API_KEY:
        raise ValueError("ODDS_API_KEY not found in .env file")

    url = (
        f"{ODDS_API_BASE}/sports/{SPORT_KEY}/odds"
        f"?apiKey={ODDS_API_KEY}"
        f"&regions={REGIONS}"
        f"&markets={MARKETS}"
        f"&oddsFormat={ODDS_FORMAT}"
    )

    logger.info("Fetching upcoming fixtures from The Odds API...")
    resp = requests.get(url, timeout=15)

    if resp.status_code != 200:
        raise RuntimeError(
            f"Odds API error {resp.status_code}: {resp.text}"
        )

    remaining = resp.headers.get("x-requests-remaining", "?")
    logger.info(f"Odds API requests remaining this month: {remaining}")

    games = resp.json()
    logger.info(f"Fixtures returned: {len(games)}")

    rows = []
    for game in games:
        home_raw = game["home_team"]
        away_raw = game["away_team"]

        # Normalise to DB names
        home = ODDS_API_TO_DB.get(home_raw, home_raw)
        away = ODDS_API_TO_DB.get(away_raw, away_raw)

        date = pd.to_datetime(game["commence_time"]).tz_convert("UTC")

        # Find best over/under 2.5 odds across available bookmakers
        # Priority: Pinnacle > Bet365 > first available
        best_over  = None
        best_under = None

        bookmaker_priority = ["pinnacle", "bet365", "betfair_ex_eu"]
        bookmakers_by_key  = {b["key"]: b for b in game.get("bookmakers", [])}

        for bk_key in bookmaker_priority + list(bookmakers_by_key.keys()):
            bk = bookmakers_by_key.get(bk_key)
            if not bk:
                continue
            for market in bk.get("markets", []):
                if market["key"] != "totals":
                    continue
                for outcome in market.get("outcomes", []):
                    if outcome.get("point") == 2.5:
                        if outcome["name"] == "Over" and best_over is None:
                            best_over = outcome["price"]
                        if outcome["name"] == "Under" and best_under is None:
                            best_under = outcome["price"]
            if best_over and best_under:
                break

        if not best_over or not best_under:
            logger.warning(f"  No over/under odds found for {home} vs {away} — skipping")
            continue

        # Strip vig
        raw_over  = 1.0 / best_over
        raw_under = 1.0 / best_under
        overround = raw_over + raw_under
        market_prob_over  = raw_over  / overround
        market_prob_under = raw_under / overround

        rows.append({
            "fixture_id":        game["id"],
            "date":              date,
            "home_team":         home,
            "away_team":         away,
            "home_team_raw":     home_raw,
            "away_team_raw":     away_raw,
            "odds_over_2_5":     best_over,
            "odds_under_2_5":    best_under,
            "market_prob_over":  round(market_prob_over, 4),
            "market_prob_under": round(market_prob_under, 4),
            "overround":         round(overround, 4),
        })

    fixtures_df = pd.DataFrame(rows)

    if fixtures_df.empty:
        logger.warning("No upcoming fixtures with over/under odds found.")
        return fixtures_df

    logger.info(f"Fixtures with odds: {len(fixtures_df)}")
    logger.info("\n" + fixtures_df[["date","home_team","away_team",
                                    "odds_over_2_5","market_prob_over"]].to_string(index=False))
    return fixtures_df


# ---------------------------------------------------------------------------
# Step 3 — Build features for upcoming fixtures
# ---------------------------------------------------------------------------
def build_features_for_fixtures(
    full_df: pd.DataFrame,
    fixtures: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each upcoming fixture, look up the most recent rolling features
    for both the home and away team from the full_df.

    We use the last completed match for each team as their current
    feature state — this is the same data the model saw during training.

    Returns DataFrame with FEATURE_COLS populated for each fixture.
    """
    # Get latest feature row per team from 2025-26 data
    # full_df is match-level (one row per fixture) with _home/_away suffixes
    # We need to reconstruct team-level latest features

    # Build a lookup: team -> their most recent row as home team
    # and their most recent row as away team, then pick the latest

    current_season = full_df[full_df["season_start"] == 2025].copy()

    if current_season.empty:
        logger.warning("No 2025-26 data found — features will be NaN")

    # Latest home features per team
    home_latest = (
        current_season
        .sort_values("date_home")
        .groupby("team_name_home")
        .last()
        .reset_index()
    )

    # Latest away features per team
    away_latest = (
        current_season
        .sort_values("date_home")
        .groupby("team_name_away")
        .last()
        .reset_index()
    )

    # Home feature columns
    home_feat_cols = [c for c in FEATURE_COLS if c.endswith("_home")]
    away_feat_cols = [c for c in FEATURE_COLS if c.endswith("_away")]
    shared_cols    = [c for c in FEATURE_COLS
                      if not c.endswith("_home") and not c.endswith("_away")]

    rows = []
    for _, fixture in fixtures.iterrows():
        home_team = fixture["home_team"]
        away_team = fixture["away_team"]

        # Find latest home features for this home team
        home_row = home_latest[home_latest["team_name_home"] == home_team]
        away_row = away_latest[away_latest["team_name_away"] == away_team]

        # Fall back to away-side data if team hasn't played at home recently
        if home_row.empty:
            home_row = away_latest[away_latest["team_name_away"] == home_team]
            if not home_row.empty:
                # Rename away cols to home cols for this team
                rename_map = {c: c.replace("_away", "_home")
                              for c in away_feat_cols if c in home_row.columns}
                home_row = home_row.rename(columns=rename_map)

        if away_row.empty:
            away_row = home_latest[home_latest["team_name_home"] == away_team]
            if not away_row.empty:
                rename_map = {c: c.replace("_home", "_away")
                              for c in home_feat_cols if c in away_row.columns}
                away_row = away_row.rename(columns=rename_map)

        row = {
            "date_home":      fixture["date"],
            "team_name_home": home_team,
            "team_name_away": away_team,
            "odds_over_2_5":  fixture["odds_over_2_5"],
            "odds_under_2_5": fixture["odds_under_2_5"],
            "market_prob_over":  fixture["market_prob_over"],
            "market_prob_under": fixture["market_prob_under"],
        }

        # Pull home features
        for col in home_feat_cols:
            if not home_row.empty and col in home_row.columns:
                row[col] = home_row[col].values[0]
            else:
                row[col] = np.nan

        # Pull away features
        for col in away_feat_cols:
            if not away_row.empty and col in away_row.columns:
                row[col] = away_row[col].values[0]
            else:
                row[col] = np.nan

        # Shared/combined features — recalculate from home+away
        row["days_rest_diff"] = (
            row.get("days_rest_current_home", np.nan) -
            row.get("days_rest_current_away", np.nan)
        )
        row["combined_xg_last5"] = (
            row.get("avg_xg_last5_home", np.nan) +
            row.get("avg_xg_last5_away", np.nan)
        )
        row["combined_xg_last15"] = (
            row.get("avg_xg_last15_home", np.nan) +
            row.get("avg_xg_last15_away", np.nan)
        )
        row["combined_goals_last5"] = (
            row.get("avg_goals_last5_home", np.nan) +
            row.get("avg_goals_last5_away", np.nan)
        )
        row["combined_xg_momentum"] = (
            row.get("xg_momentum_last3_home", np.nan) +
            row.get("xg_momentum_last3_away", np.nan)
        )
        row["combined_goals_momentum"] = (
            row.get("goals_momentum_last3_home", np.nan) +
            row.get("goals_momentum_last3_away", np.nan)
        )

        # league_avg_goals_last30 — use the latest value from current season
        if not current_season.empty and "league_avg_goals_last30" in current_season.columns:
            row["league_avg_goals_last30"] = (
                current_season["league_avg_goals_last30"].dropna().iloc[-1]
                if not current_season["league_avg_goals_last30"].dropna().empty
                else np.nan
            )
        else:
            row["league_avg_goals_last30"] = np.nan

        # h2h_avg_goals_last5 — look up fixture history
        fixture_key = tuple(sorted([home_team, away_team]))
        h2h_matches = current_season[
            current_season.apply(
                lambda r: tuple(sorted([r["team_name_home"], r["team_name_away"]])) == fixture_key,
                axis=1
            )
        ]
        if len(h2h_matches) >= 2:
            recent_h2h = h2h_matches.sort_values("date_home").tail(5)
            row["h2h_avg_goals_last5"] = (
                (recent_h2h["goals_home"] + recent_h2h["goals_away"]).mean()
            )
        else:
            # Fall back to full history
            all_h2h = full_df[
                full_df.apply(
                    lambda r: tuple(sorted([r["team_name_home"], r["team_name_away"]])) == fixture_key,
                    axis=1
                )
            ]
            if len(all_h2h) >= 2:
                recent = all_h2h.sort_values("date_home").tail(5)
                row["h2h_avg_goals_last5"] = (
                    (recent["goals_home"] + recent["goals_away"]).mean()
                )
            else:
                row["h2h_avg_goals_last5"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 4 — Apply model and calculate edge
# ---------------------------------------------------------------------------
def predict_and_rank(
    model: XGBClassifier,
    fixtures_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run model on fixture features, calculate edge, apply zone filter.
    Returns ranked bet recommendations.
    """
    if fixtures_df.empty:
        logger.warning("No fixtures to predict on.")
        return pd.DataFrame()

    X = fixtures_df[FEATURE_COLS]
    fixtures_df = fixtures_df.copy()
    fixtures_df["model_prob"] = model.predict_proba(X)[:, 1]

    fixtures_df["edge"] = (
        fixtures_df["model_prob"] - fixtures_df["market_prob_over"]
    )

    fixtures_df["bet_recommended"] = fixtures_df.apply(
        lambda r: is_value_bet(r["edge"], r["odds_over_2_5"], EPL),
        axis=1,
    )

    fixtures_df["suggested_stake"] = fixtures_df.apply(
        lambda r: suggested_stake(r["edge"], r["odds_over_2_5"], STARTING_BANKROLL, EPL)
        if r["bet_recommended"] else 0.0,
        axis=1,
    )

    return fixtures_df.sort_values("edge", ascending=False)


# ---------------------------------------------------------------------------
# Step 5 — Print output
# ---------------------------------------------------------------------------
def print_recommendations(df: pd.DataFrame) -> None:
    bets = df[df["bet_recommended"] == True]

    print("\n" + "="*60)
    print("EPL OVER 2.5 GOALS — BET RECOMMENDATIONS")
    print(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("="*60)

    if bets.empty:
        print("\nNo bets recommended for upcoming fixtures.")
        print("(No fixtures in calibrated edge zones 0.11-0.13 or 0.15+)")
    else:
        print(f"\n{len(bets)} bet(s) recommended:\n")
        for _, row in bets.iterrows():
            date_str = pd.to_datetime(row["date_home"]).strftime("%a %d %b %H:%M")
            print(f"  {row['team_name_home']} vs {row['team_name_away']}")
            print(f"    Date:          {date_str} UTC")
            print(f"    Odds (over):   {row['odds_over_2_5']:.2f}")
            print(f"    Model prob:    {row['model_prob']:.1%}")
            print(f"    Market prob:   {row['market_prob_over']:.1%}")
            print(f"    Edge:          {row['edge']:+.1%}")
            print(f"    Suggested stake (£1k bank): £{row['suggested_stake']:.2f}")
            print()

    print("-"*60)
    print("\nAll upcoming fixtures:\n")
    display_cols = [
        "team_name_home", "team_name_away",
        "odds_over_2_5", "model_prob", "market_prob_over",
        "edge", "bet_recommended"
    ]
    display = df[display_cols].copy()
    display["model_prob"]       = display["model_prob"].map("{:.1%}".format)
    display["market_prob_over"] = display["market_prob_over"].map("{:.1%}".format)
    display["edge"]             = display["edge"].map("{:+.1%}".format)
    display.columns = ["Home", "Away", "Odds", "Model%", "Market%", "Edge", "Bet?"]
    print(display.to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Step 1 — Load all data and train
    logger.info("Loading match data and building features...")
    full_df = load_data(live=True)

    logger.info("Training model on all historical seasons...")
    model = train_model(full_df)

    # Step 2 — Fetch upcoming fixtures
    fixtures = fetch_upcoming_odds()

    if fixtures.empty:
        print("No upcoming fixtures found. Check ODDS_API_KEY and try again.")
        sys.exit(0)

    # Step 3 — Build features for upcoming fixtures
    logger.info("Building features for upcoming fixtures...")
    fixture_features = build_features_for_fixtures(full_df, fixtures)

    # Step 4 — Predict and rank
    logger.info("Running model predictions...")
    results = predict_and_rank(model, fixture_features)

    # Step 5 — Print recommendations
    print_recommendations(results)

    # Save latest
    out_path = project_root / "predictions" / "latest.csv"
    out_path.parent.mkdir(exist_ok=True)
    results.to_csv(out_path, index=False)

    # Archive with today's date so tracker.py can find historical predictions
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    archive_dir = project_root / "predictions" / "history"
    archive_dir.mkdir(exist_ok=True)
    archive_path = archive_dir / f"{today}.csv"
    results.to_csv(archive_path, index=False)

    logger.info(f"Predictions saved to {out_path}")
    logger.info(f"Archived to {archive_path}")