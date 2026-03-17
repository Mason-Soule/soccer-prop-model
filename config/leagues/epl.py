"""
config/leagues/epl.py
---------------------
Single source of truth for all EPL-specific constants.

When adding a new league (e.g. Bundesliga), create a new file:
    config/leagues/bundesliga.py

with the same structure but different values. Every script imports
from here instead of hardcoding constants locally.

Usage:
    from config.leagues.epl import EPL
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LeagueConfig:
    """
    All constants needed to run the full pipeline for one league.
    frozen=True means these are immutable — no accidental overrides.
    """

    # --- Identity ---
    name: str        # "English Premier League"
    key: str         # "epl" — used in file paths
    db_league: str   # value stored in DB `league` column

    # --- Data sources ---
    fd_seasons: dict         = field(default_factory=dict)   # {"2024-25": "2425"}
    fd_division: str         = "E0"                          # "E0" EPL, "D1" Bundesliga
    understat_league: str    = "EPL"                         # Understat league name
    understat_seasons: list  = field(default_factory=list)   # [2014, 2015, ...]
    odds_api_sport_key: str  = "soccer_epl"                  # The Odds API key

    # --- Team name maps ---
    fd_team_map: dict        = field(default_factory=dict)   # football-data -> DB
    understat_team_map: dict = field(default_factory=dict)   # Understat -> DB
    odds_api_team_map: dict  = field(default_factory=dict)   # Odds API -> DB

    # --- Model config ---
    min_train_seasons: int   = 5
    min_odds: float          = 1.60
    max_odds: float          = 1.75
    kelly_fraction: float    = 0.15
    kelly_fraction_wide: float = 0.10   # for bets outside calibrated odds range
    edge_zones: list         = field(default_factory=lambda: [
        (0.11, 0.13),   # moderate edge zone
        (0.15, None),   # high edge zone — None means open ended
    ])
    starting_bankroll: float = 1000.0


# ---------------------------------------------------------------------------
# EPL instance — import this in your scripts
# ---------------------------------------------------------------------------
EPL = LeagueConfig(
    name      = "English Premier League",
    key       = "epl",
    db_league = "EPL",

    fd_seasons = {
        "2014-15": "1415",
        "2015-16": "1516",
        "2016-17": "1617",
        "2017-18": "1718",
        "2018-19": "1819",
        "2019-20": "1920",
        "2020-21": "2021",
        "2021-22": "2122",
        "2022-23": "2223",
        "2023-24": "2324",
        "2024-25": "2425",
        "2025-26": "2526",
    },

    fd_division        = "E0",
    understat_league   = "EPL",
    understat_seasons  = [2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024,2025],
    odds_api_sport_key = "soccer_epl",

    fd_team_map = {
        "Nottm Forest":             "Nott'm Forest",
        "Nottingham Forest":        "Nott'm Forest",
        "Manchester United":        "Man United",
        "Manchester City":          "Man City",
        "Newcastle United":         "Newcastle",
        "Wolverhampton":            "Wolves",
        "Wolverhampton Wanderers":  "Wolves",
        "West Bromwich Albion":     "West Brom",
        "West Bromwich":            "West Brom",
        "Sheffield Utd":            "Sheffield United",
        "Leicester City":           "Leicester",
        "Leeds United":             "Leeds",
        "Spurs":                    "Tottenham",
        "Tottenham Hotspur":        "Tottenham",
        "Brighton & Hove Albion":   "Brighton",
        "Brighton and Hove Albion": "Brighton",
        "Stoke City":               "Stoke",
        "Norwich City":             "Norwich",
        "Huddersfield Town":        "Huddersfield",
        "Cardiff City":             "Cardiff",
        "Hull City":                "Hull",
        "Swansea City":             "Swansea",
        "Ipswich Town":             "Ipswich",
    },

    understat_team_map = {
        "Manchester_City":          "Man City",
        "Manchester_United":        "Man United",
        "Newcastle_United":         "Newcastle",
        "Tottenham":                "Tottenham",
        "Wolverhampton_Wanderers":  "Wolves",
        "Nottingham_Forest":        "Nott'm Forest",
        "West_Ham":                 "West Ham",
        "Brighton":                 "Brighton",
        "Leicester":                "Leicester",
        "Leeds":                    "Leeds",
        "West_Brom":                "West Brom",
        "Sheffield_United":         "Sheffield United",
        "Huddersfield":             "Huddersfield",
        "Cardiff":                  "Cardiff",
        "Hull":                     "Hull",
        "Swansea":                  "Swansea",
        "Norwich":                  "Norwich",
        "Stoke":                    "Stoke",
        "Sunderland":               "Sunderland",
        "Middlesbrough":            "Middlesbrough",
        "Watford":                  "Watford",
        "Ipswich":                  "Ipswich",
    },

    odds_api_team_map = {
        "Manchester City":          "Man City",
        "Manchester United":        "Man United",
        "Newcastle United":         "Newcastle",
        "Tottenham Hotspur":        "Tottenham",
        "Wolverhampton Wanderers":  "Wolves",
        "Nottingham Forest":        "Nott'm Forest",
        "West Ham United":          "West Ham",
        "Brighton & Hove Albion":   "Brighton",
        "Brighton and Hove Albion": "Brighton",
        "Leicester City":           "Leicester",
        "Leeds United":             "Leeds",
        "West Bromwich Albion":     "West Brom",
        "Sheffield United":         "Sheffield United",
        "Aston Villa":              "Aston Villa",
        "AFC Bournemouth":          "Bournemouth",
        "Ipswich Town":             "Ipswich",
        "Huddersfield Town":        "Huddersfield",
        "Cardiff City":             "Cardiff",
        "Hull City":                "Hull",
        "Swansea City":             "Swansea",
        "Norwich City":             "Norwich",
        "Stoke City":               "Stoke",
        "Sunderland AFC":           "Sunderland",
    },

    min_train_seasons   = 5,
    min_odds            = 1.60,
    max_odds            = 1.75,
    kelly_fraction      = 0.15,
    kelly_fraction_wide = 0.10,
    edge_zones          = [(0.11, 0.13), (0.15, None)],
    starting_bankroll   = 1000.0,
)