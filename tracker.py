"""
tracker.py
----------
Tracks live bet results against model predictions.

Workflow:
    1. Loads all saved predictions from predictions/history/
    2. Looks up actual results from the DB for completed fixtures
    3. Calculates running P&L, hit rate, ROI, and edge validation
    4. Saves updated results to predictions/live_results.csv
    5. Prints a summary dashboard

Run after each gameweek once results are in:
    python tracker.py

Predictions are archived automatically by predict.py each run.
To enable archiving, predict.py saves to predictions/history/YYYY-MM-DD.csv
as well as predictions/latest.csv.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from rich import box

load_dotenv()

project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

PREDICTIONS_DIR  = project_root / "predictions"
HISTORY_DIR      = PREDICTIONS_DIR / "history"
LIVE_RESULTS_CSV = PREDICTIONS_DIR / "live_results.csv"
STARTING_BANKROLL = 1000.0
KELLY_FRACTION    = 0.15
KELLY_FRACTION_WIDE = 0.10  # for bets with odds > 1.75


# ---------------------------------------------------------------------------
# Load all saved predictions
# ---------------------------------------------------------------------------
def load_all_predictions() -> pd.DataFrame:
    """
    Load all prediction CSVs from predictions/history/.
    Falls back to predictions/latest.csv if no history exists yet.
    """
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    history_files = sorted(HISTORY_DIR.glob("*.csv"))

    if history_files:
        dfs = []
        for f in history_files:
            df = pd.read_csv(f)
            df["prediction_date"] = f.stem  # filename is the date
            dfs.append(df)
        combined = pd.concat(dfs, ignore_index=True)
        combined = (
            combined
            .sort_values("prediction_date")
            .drop_duplicates(
                subset=["team_name_home", "team_name_away"],
                keep="last"
            )
            .reset_index(drop=True)
        )

        logger.info(f"Loaded {len(history_files)} prediction files, "
                    f"{len(combined)} total fixture rows after dedup")
        return combined

    # Fall back to latest.csv
    latest = PREDICTIONS_DIR / "latest.csv"
    if latest.exists():
        logger.info("No history found — loading predictions/latest.csv")
        df = pd.read_csv(latest)
        df["prediction_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return df

    logger.warning("No prediction files found.")
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Look up actual results from DB
# ---------------------------------------------------------------------------
def fetch_results_from_db() -> pd.DataFrame:
    """
    Load completed 2025-26 match results from the DB.
    Returns one row per match with home/away team names and total goals.
    """
    engine = create_engine(
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@localhost:5432/{os.getenv('DB_NAME')}"
    )

    query = """
    SELECT
        m.date::date                AS match_date,
        ht.name                     AS home_team,
        at.name                     AS away_team,
        home_stats.goals            AS home_goals,
        away_stats.goals            AS away_goals,
        home_stats.goals + away_stats.goals AS total_goals,
        CASE WHEN home_stats.goals + away_stats.goals > 2
             THEN 1 ELSE 0 END      AS over_2_5
    FROM matches m
    JOIN teams ht         ON m.home_team_id = ht.team_id
    JOIN teams at         ON m.away_team_id = at.team_id
    JOIN team_match_stats home_stats
        ON home_stats.match_id = m.match_id
        AND home_stats.is_home = TRUE
    JOIN team_match_stats away_stats
        ON away_stats.match_id = m.match_id
        AND away_stats.is_home = FALSE
    WHERE m.season = '2025-26'
    ORDER BY m.date
    """

    df = pd.read_sql(query, engine)
    df["match_date"] = pd.to_datetime(df["match_date"]).dt.date
    logger.info(f"Loaded {len(df)} completed 2025-26 matches from DB")
    return df


# ---------------------------------------------------------------------------
# Match predictions to results
# ---------------------------------------------------------------------------
def match_predictions_to_results(
    predictions: pd.DataFrame,
    results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join predictions onto actual results by date + home team + away team.
    Only returns rows where bet_recommended == True.
    """
    # Only care about recommended bets
    bets = predictions[predictions["bet_recommended"] == True].copy()

    if bets.empty:
        logger.warning("No recommended bets found in predictions.")
        return pd.DataFrame()

    # Normalise dates
    bets["match_date"] = pd.to_datetime(bets["date_home"]).dt.date
    results["match_date"] = pd.to_datetime(results["match_date"]).dt.date

    # Merge on date + teams
    merged = bets.merge(
        results,
        left_on=["match_date", "team_name_home", "team_name_away"],
        right_on=["match_date", "home_team",      "away_team"],
        how="left",
    )

    total  = len(merged)
    settled = merged["over_2_5"].notna().sum()
    pending = total - settled

    logger.info(f"Bets: {total} total, {settled} settled, {pending} pending")
    return merged


# ---------------------------------------------------------------------------
# Calculate running P&L
# ---------------------------------------------------------------------------
def calculate_pnl(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate stake, result, and running P&L for each settled bet.
    Uses the same Kelly formula as predict.py.
    """
    df = df.copy()
    settled = df[df["over_2_5"].notna()].sort_values("match_date").copy()

    bankroll = STARTING_BANKROLL
    rows = []

    for _, row in settled.iterrows():
        edge = row["edge"]
        odds = row["odds_over_2_5"]

        if edge <= 0:
            continue

        b = odds - 1.0
        fraction = KELLY_FRACTION if odds <= 1.75 else KELLY_FRACTION_WIDE
        stake = bankroll * (edge / b) * fraction
        stake = min(stake, bankroll * 0.05)  # hard cap 5%
        stake = max(stake, 1.0)

        won = int(row["over_2_5"]) == 1
        if won:
            profit  = stake * (odds - 1)
            bankroll += profit
            result   = "WIN"
        else:
            bankroll -= stake
            result    = "LOSS"

        rows.append({
            "date":          row["match_date"],
            "home":          row["team_name_home"],
            "away":          row["team_name_away"],
            "odds":          round(odds, 2),
            "model_prob":    round(row["model_prob"], 3),
            "market_prob":   round(row["market_prob_over"], 3),
            "edge":          round(edge, 3),
            "stake":         round(stake, 2),
            "result":        result,
            "total_goals":   int(row["total_goals"]) if pd.notna(row.get("total_goals")) else None,
            "bankroll":      round(bankroll, 2),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Rich dashboard
# ---------------------------------------------------------------------------
def print_dashboard(
    pnl_df: pd.DataFrame,
    all_bets: pd.DataFrame,
) -> None:
    console = Console()
    today   = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    console.print()
    console.rule("[bold white]EPL Over 2.5 · Live Bet Tracker[/]")
    console.print(f"[dim]Updated: {today}[/]\n")

    # ------------------------------------------------------------------
    # Pending bets table
    # ------------------------------------------------------------------
    pending = all_bets[all_bets["over_2_5"].isna()]
    if not pending.empty:
        pt = Table(
            title=f"Pending Bets ({len(pending)})",
            box=box.SIMPLE_HEAD,
            show_lines=False,
            title_style="bold yellow",
            header_style="dim",
        )
        pt.add_column("Match",     style="white",  no_wrap=True)
        pt.add_column("Date",      style="dim",    no_wrap=True)
        pt.add_column("Odds",      justify="right")
        pt.add_column("Model",     justify="right")
        pt.add_column("Market",    justify="right")
        pt.add_column("Edge",      justify="right")
        pt.add_column("Stake",     justify="right")

        for _, row in pending.sort_values("date_home").iterrows():
            date_str = pd.to_datetime(row["date_home"]).strftime("%a %d %b %H:%M")
            edge     = row["edge"]
            edge_str = f"{edge:+.1%}"
            # colour code edge by strength
            if edge >= 0.15:
                edge_col = f"[bold green]{edge_str}[/]"
            elif edge >= 0.11:
                edge_col = f"[green]{edge_str}[/]"
            else:
                edge_col = f"[yellow]{edge_str}[/]"

            stake_str = (
            f"£{row['suggested_stake']:.2f}"
            if "suggested_stake" in row and pd.notna(row.get("suggested_stake"))
            else "-"
        )
            pt.add_row(
                f"{row['team_name_home']} vs {row['team_name_away']}",
                date_str,
                f"{row['odds_over_2_5']:.2f}",
                f"{row['model_prob']:.1%}",
                f"{row['market_prob_over']:.1%}",
                edge_col,
                stake_str,
            )
        console.print(pt)

    # ------------------------------------------------------------------
    # No settled bets yet
    # ------------------------------------------------------------------
    if pnl_df.empty:
        console.print("[dim]No settled bets yet — results will appear here after games are played.[/]\n")
        return

    # ------------------------------------------------------------------
    # Summary stats
    # ------------------------------------------------------------------
    wins       = (pnl_df["result"] == "WIN").sum()
    losses     = (pnl_df["result"] == "LOSS").sum()
    total      = wins + losses
    hit_rate   = wins / total if total > 0 else 0
    final_bank = pnl_df["bankroll"].iloc[-1]
    net_pnl    = final_bank - STARTING_BANKROLL
    total_staked = pnl_df["stake"].sum()
    roi        = net_pnl / total_staked * 100 if total_staked > 0 else 0
    avg_edge   = pnl_df["edge"].mean()
    avg_market = pnl_df["market_prob"].mean()
    outperf    = hit_rate - avg_market

    # Max drawdown
    peak   = STARTING_BANKROLL
    max_dd = 0.0
    for bank in pnl_df["bankroll"]:
        if bank > peak:
            peak = bank
        dd = (peak - bank) / peak
        if dd > max_dd:
            max_dd = dd

    pnl_colour  = "green" if net_pnl >= 0 else "red"
    roi_colour  = "green" if roi >= 0 else "red"
    out_colour  = "green" if outperf >= 0 else "red"

    # Four summary panels side by side
    panels = [
        Panel(
            f"[bold]{wins}W / {losses}L[/]\n[dim]{hit_rate:.1%} hit rate[/]",
            title="Record", border_style="dim"
        ),
        Panel(
            f"[bold {pnl_colour}]£{net_pnl:+,.2f}[/]\n[dim]ROI {roi:+.1f}%[/]",
            title="P&L", border_style="dim"
        ),
        Panel(
            f"[bold {out_colour}]{outperf:+.1%}[/]\n[dim]vs {avg_market:.1%} implied[/]",
            title="Outperformance", border_style="dim"
        ),
        Panel(
            f"[bold]{max_dd:.1%}[/]\n[dim]bank: £{final_bank:,.2f}[/]",
            title="Max Drawdown", border_style="dim"
        ),
    ]
    console.print(Columns(panels))

    # ------------------------------------------------------------------
    # Settled bet log table
    # ------------------------------------------------------------------
    bt = Table(
        title="Bet Log",
        box=box.SIMPLE_HEAD,
        show_lines=False,
        title_style="bold white",
        header_style="dim",
    )
    bt.add_column("Date",   no_wrap=True)
    bt.add_column("Home",   style="white")
    bt.add_column("Away",   style="white")
    bt.add_column("Odds",   justify="right")
    bt.add_column("Edge",   justify="right")
    bt.add_column("Stake",  justify="right")
    bt.add_column("Goals",  justify="center")
    bt.add_column("Result", justify="center")
    bt.add_column("Bank",   justify="right")

    for _, row in pnl_df.iterrows():
        goals      = str(int(row["total_goals"])) if pd.notna(row["total_goals"]) else "-"
        won        = row["result"] == "WIN"
        result_str = "[bold green]WIN[/]" if won else "[bold red]LOSS[/]"
        bank_col   = "green" if row["bankroll"] >= STARTING_BANKROLL else "red"

        bt.add_row(
            str(row["date"]),
            row["home"],
            row["away"],
            f"{row['odds']:.2f}",
            f"{row['edge']:+.1%}",
            f"£{row['stake']:.2f}",
            goals,
            result_str,
            f"[{bank_col}]£{row['bankroll']:,.2f}[/]",
        )
    console.print(bt)

    # ------------------------------------------------------------------
    # Edge validation
    # ------------------------------------------------------------------
    expected_wins = pnl_df["market_prob"].sum()
    diff          = wins - expected_wins
    if diff >= 0:
        status = f"[green]Outperforming market by {diff:.1f} wins[/]"
    else:
        status = f"[red]Underperforming market by {abs(diff):.1f} wins[/]"

    if total < 10:
        caution = f"[yellow]Only {total} settled bets — too small for conclusions[/]"
    elif total < 30:
        caution = f"[yellow]{total} settled bets — early signal, keep tracking[/]"
    else:
        caution = f"[green]{total} settled bets — meaningful sample[/]"

    console.print(Panel(
        f"Expected wins (market): [bold]{expected_wins:.1f}[/]   "
        f"Actual wins: [bold]{wins}[/]\n"
        f"{status}\n{caution}",
        title="Edge Validation",
        border_style="dim",
    ))
    console.print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Load predictions
    logging.getLogger().setLevel(logging.WARNING)  # suppress INFO during dashboard
    
    predictions = load_all_predictions()
    if predictions.empty:
        print("No predictions found. Run predict.py first.")
        sys.exit(0)

    # Fetch actual results from DB
    results = fetch_results_from_db()

    # Match predictions to results
    all_bets = match_predictions_to_results(predictions, results)
    if all_bets.empty:
        print("No recommended bets to track yet.")
        sys.exit(0)

    # Calculate P&L on settled bets
    pnl_df = calculate_pnl(all_bets)

    # Print dashboard
    print_dashboard(pnl_df, all_bets)

    # Save results
    if not pnl_df.empty:
        LIVE_RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
        pnl_df.to_csv(LIVE_RESULTS_CSV, index=False)
        logger.info(f"Results saved to {LIVE_RESULTS_CSV}")