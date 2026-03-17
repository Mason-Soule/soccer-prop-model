"""
core/staking.py
---------------
Single source of truth for Kelly staking and edge zone filtering.

Every script that calculates stakes or filters bets imports from here:
    from core.staking import kelly_stake, is_value_bet

Never define staking logic locally in backtest.py or predict.py.
Change Kelly fraction or edge zones in config/leagues/epl.py and
they flow through automatically.
"""

import pandas as pd
from config.leagues.epl import LeagueConfig


def is_value_bet(
    edge: float,
    odds: float,
    league: LeagueConfig,
) -> bool:
    """
    Return True if this bet passes the edge zone and odds filters.

    Edge zones are calibrated from backtest analysis — only zones with
    demonstrated positive outperformance are included. Bets outside
    these zones showed negative outperformance historically.

    Args:
        edge:   model_prob - market_prob (vig-free)
        odds:   decimal odds for the bet
        league: LeagueConfig instance with edge_zones, min_odds, max_odds

    Returns:
        True if the bet should be placed
    """
    if pd.isna(edge) or pd.isna(odds):
        return False

    # Odds range filter
    if odds < league.min_odds or odds > league.max_odds:
        return False

    # Edge zone filter — must fall in at least one calibrated zone
    for low, high in league.edge_zones:
        if high is None:
            # Open-ended zone e.g. (0.15, None) means edge >= 0.15
            if edge >= low:
                return True
        else:
            if low <= edge < high:
                return True

    return False


def kelly_stake(
    edge: float,
    odds: float,
    league: LeagueConfig,
) -> float:
    """
    Calculate fractional Kelly stake as a proportion of bankroll.

    Kelly formula: f = edge / (odds - 1)
    Then multiplied by the Kelly fraction to reduce variance.

    Uses a reduced fraction for bets outside the originally calibrated
    odds range (min_odds to max_odds) to reflect lower confidence.

    Args:
        edge:   model_prob - market_prob (vig-free)
        odds:   decimal odds for the bet
        league: LeagueConfig with kelly_fraction, kelly_fraction_wide

    Returns:
        Stake as proportion of current bankroll (0.0 if no edge)
    """
    if edge <= 0 or pd.isna(edge) or pd.isna(odds):
        return 0.0

    b = odds - 1.0
    if b <= 0:
        return 0.0

    # Use reduced fraction outside the calibrated odds range
    fraction = (
        league.kelly_fraction
        if league.min_odds <= odds <= league.max_odds
        else league.kelly_fraction_wide
    )

    return max(0.0, (edge / b) * fraction)


def suggested_stake(
    edge: float,
    odds: float,
    bankroll: float,
    league: LeagueConfig,
) -> float:
    """
    Calculate the suggested stake in £ given the current bankroll.

    Args:
        edge:     model_prob - market_prob
        odds:     decimal odds
        bankroll: current bankroll in £
        league:   LeagueConfig

    Returns:
        Stake in £, rounded to 2 decimal places
    """
    fraction = kelly_stake(edge, odds, league)
    return round(bankroll * fraction, 2)