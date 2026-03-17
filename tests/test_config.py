"""
tests/test_config.py
--------------------
Tests for league config integrity.

Run with:
    python -m pytest tests/test_config.py -v
"""

import sys
import pytest
from pathlib import Path

project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from config.leagues.epl import EPL, LeagueConfig


# ---------------------------------------------------------------------------
# EPL config sanity checks
# ---------------------------------------------------------------------------

def test_epl_identity():
    """EPL config must have correct identity fields."""
    assert EPL.name == "English Premier League"
    assert EPL.key  == "epl"
    assert EPL.db_league == "EPL"


def test_epl_has_seasons():
    """EPL must have at least 10 seasons configured."""
    assert len(EPL.fd_seasons) >= 10, (
        f"Expected 10+ seasons, got {len(EPL.fd_seasons)}"
    )


def test_epl_season_codes_format():
    """Season codes must be 4-digit strings like '1617'."""
    for label, code in EPL.fd_seasons.items():
        assert len(code) == 4 and code.isdigit(), (
            f"Season code '{code}' for '{label}' should be 4 digits e.g. '1617'"
        )


def test_epl_season_labels_format():
    """Season labels must follow 'YYYY-YY' format e.g. '2024-25'."""
    for label in EPL.fd_seasons.keys():
        parts = label.split("-")
        assert len(parts) == 2, f"Label '{label}' should be 'YYYY-YY'"
        assert len(parts[0]) == 4, f"Label '{label}' year should be 4 digits"
        assert len(parts[1]) == 2, f"Label '{label}' suffix should be 2 digits"


def test_epl_understat_seasons_are_ints():
    """Understat seasons must be integers."""
    for s in EPL.understat_seasons:
        assert isinstance(s, int), f"Season {s} should be int not {type(s)}"


def test_epl_understat_seasons_start_at_2014():
    """EPL understat data starts at 2014 — earlier seasons have no xG."""
    assert min(EPL.understat_seasons) >= 2014, (
        "Understat xG data is not available before 2014-15"
    )


def test_epl_odds_range_valid():
    """MIN_ODDS must be less than MAX_ODDS."""
    assert EPL.min_odds < EPL.max_odds, (
        f"min_odds ({EPL.min_odds}) must be less than max_odds ({EPL.max_odds})"
    )


def test_epl_odds_range_sensible():
    """Odds range must be within sensible betting bounds."""
    assert EPL.min_odds >= 1.30, "min_odds below 1.30 is unrealistically low"
    assert EPL.max_odds <= 3.00, "max_odds above 3.00 is outside typical range"


def test_epl_kelly_fraction_sensible():
    """Kelly fraction must be between 0 and 1."""
    assert 0 < EPL.kelly_fraction < 1.0, (
        f"kelly_fraction {EPL.kelly_fraction} must be between 0 and 1"
    )
    assert 0 < EPL.kelly_fraction_wide < EPL.kelly_fraction, (
        f"kelly_fraction_wide should be smaller than kelly_fraction"
    )


def test_epl_edge_zones_valid():
    """Edge zones must be non-overlapping and well-formed."""
    for i, (low, high) in enumerate(EPL.edge_zones):
        assert low >= 0, f"Zone {i} low bound {low} must be >= 0"
        if high is not None:
            assert high > low, (
                f"Zone {i}: high ({high}) must be greater than low ({low})"
            )


def test_epl_edge_zones_no_overlap():
    """Edge zones must not overlap each other."""
    zones = [(l, h if h is not None else 999) for l, h in EPL.edge_zones]
    for i in range(len(zones) - 1):
        assert zones[i][1] <= zones[i+1][0], (
            f"Zone {i} overlaps with zone {i+1}: {zones[i]} and {zones[i+1]}"
        )


def test_epl_team_maps_not_empty():
    """All three team name maps must have entries."""
    assert len(EPL.fd_team_map) > 0,        "fd_team_map is empty"
    assert len(EPL.understat_team_map) > 0, "understat_team_map is empty"
    assert len(EPL.odds_api_team_map) > 0,  "odds_api_team_map is empty"


def test_epl_is_frozen():
    """LeagueConfig must be immutable — accidental overrides raise errors."""
    with pytest.raises((AttributeError, TypeError)):
        EPL.min_odds = 1.50  # type: ignore


def test_league_config_is_dataclass():
    """EPL must be a LeagueConfig instance."""
    assert isinstance(EPL, LeagueConfig)