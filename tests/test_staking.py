"""
tests/test_staking.py
---------------------
Tests for Kelly staking and edge zone filtering.

Run with:
    python -m pytest tests/test_staking.py -v
"""

import sys
import pytest
import math
from pathlib import Path

project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from core.staking import is_value_bet, kelly_stake, suggested_stake
from config.leagues.epl import EPL


# ---------------------------------------------------------------------------
# is_value_bet
# ---------------------------------------------------------------------------

def test_value_bet_in_zone_1():
    """Edge 0.11-0.13 with odds in range should be a value bet."""
    assert is_value_bet(edge=0.12, odds=1.65, league=EPL) is True


def test_value_bet_in_zone_2():
    """Edge >= 0.15 with odds in range should be a value bet."""
    assert is_value_bet(edge=0.18, odds=1.70, league=EPL) is True


def test_value_bet_dead_zone():
    """Edge 0.13-0.15 should NOT be a value bet — dead zone."""
    assert is_value_bet(edge=0.14, odds=1.68, league=EPL) is False


def test_value_bet_below_threshold():
    """Edge below 0.11 should NOT be a value bet."""
    assert is_value_bet(edge=0.09, odds=1.65, league=EPL) is False


def test_value_bet_odds_too_low():
    """Odds below MIN_ODDS should be rejected even with strong edge."""
    assert is_value_bet(edge=0.20, odds=1.55, league=EPL) is False


def test_value_bet_odds_too_high():
    """Odds above MAX_ODDS should be rejected even with strong edge."""
    assert is_value_bet(edge=0.20, odds=1.80, league=EPL) is False


def test_value_bet_negative_edge():
    """Negative edge is never a value bet."""
    assert is_value_bet(edge=-0.05, odds=1.65, league=EPL) is False


def test_value_bet_nan_edge():
    """NaN edge should return False gracefully."""
    import math
    assert is_value_bet(edge=float("nan"), odds=1.65, league=EPL) is False


def test_value_bet_nan_odds():
    """NaN odds should return False gracefully."""
    assert is_value_bet(edge=0.15, odds=float("nan"), league=EPL) is False


def test_value_bet_exact_boundary_low():
    """Edge exactly at 0.11 should be a value bet (zone starts at 0.11)."""
    assert is_value_bet(edge=0.11, odds=1.65, league=EPL) is True


def test_value_bet_exact_boundary_high():
    """Edge exactly at 0.13 should NOT be a value bet (zone ends before 0.13)."""
    assert is_value_bet(edge=0.13, odds=1.65, league=EPL) is False


def test_value_bet_exact_zone2_start():
    """Edge exactly at 0.15 should be a value bet (zone 2 starts at 0.15)."""
    assert is_value_bet(edge=0.15, odds=1.65, league=EPL) is True


# ---------------------------------------------------------------------------
# kelly_stake
# ---------------------------------------------------------------------------

def test_kelly_stake_positive_edge():
    """Kelly stake should be positive for positive edge."""
    stake = kelly_stake(edge=0.12, odds=1.68, league=EPL)
    assert stake > 0


def test_kelly_stake_zero_edge():
    """Kelly stake should be 0 for zero edge."""
    assert kelly_stake(edge=0.0, odds=1.68, league=EPL) == 0.0


def test_kelly_stake_negative_edge():
    """Kelly stake should be 0 for negative edge."""
    assert kelly_stake(edge=-0.05, odds=1.68, league=EPL) == 0.0


def test_kelly_stake_formula():
    """
    Kelly stake = (edge / (odds - 1)) * fraction.
    For edge=0.12, odds=1.68, fraction=0.15:
    f = (0.12 / 0.68) * 0.15 = 0.02647...
    """
    stake = kelly_stake(edge=0.12, odds=1.68, league=EPL)
    expected = (0.12 / 0.68) * EPL.kelly_fraction
    assert abs(stake - expected) < 0.0001, (
        f"Expected {expected:.4f}, got {stake:.4f}"
    )


def test_kelly_stake_wide_odds_uses_reduced_fraction():
    """
    Bets with odds above MAX_ODDS should use kelly_fraction_wide (0.10)
    not the standard kelly_fraction (0.15).
    """
    odds_wide = EPL.max_odds + 0.10  # just above calibrated range
    stake_wide = kelly_stake(edge=0.12, odds=odds_wide, league=EPL)

    odds_normal = EPL.min_odds + 0.05  # inside calibrated range
    stake_normal = kelly_stake(edge=0.12, odds=odds_normal, league=EPL)

    # Wide odds stake should be smaller due to reduced fraction
    assert stake_wide < stake_normal, (
        "Stake for out-of-range odds should be smaller than in-range odds "
        f"(wide: {stake_wide:.4f}, normal: {stake_normal:.4f})"
    )


# ---------------------------------------------------------------------------
# suggested_stake
# ---------------------------------------------------------------------------

def test_suggested_stake_scales_with_bankroll():
    """Suggested stake should scale proportionally with bankroll."""
    stake_1k = suggested_stake(edge=0.12, odds=1.68, bankroll=1000.0, league=EPL)
    stake_2k = suggested_stake(edge=0.12, odds=1.68, bankroll=2000.0, league=EPL)
    assert abs(stake_2k - stake_1k * 2) < 0.01, (
        f"Stake should double when bankroll doubles: {stake_1k} vs {stake_2k}"
    )


def test_suggested_stake_is_rounded():
    """Suggested stake should be rounded to 2 decimal places."""
    stake = suggested_stake(edge=0.12, odds=1.68, bankroll=1000.0, league=EPL)
    assert stake == round(stake, 2), "Stake should be rounded to 2dp"


def test_suggested_stake_zero_for_no_edge():
    """Suggested stake should be 0 for negative edge."""
    stake = suggested_stake(edge=-0.05, odds=1.68, bankroll=1000.0, league=EPL)
    assert stake == 0.0