"""
core/data_loader.py
-------------------
Single source of truth for loading and preparing match data.

Every script that needs data imports from here:
    from core.data_loader import load_data

Never reimplement load_data() locally in backtest.py, train.py,
or predict.py.

Two modes:
    load_data()              — all seasons, used by backtest and train
    load_data(live=True)     — includes current incomplete season,
                               used by predict.py for live features
"""

import logging
import pandas as pd
from core.features import FEATURE_COLS, validate_feature_cols
from data.processed.build_dataset import build_dataframe
from ingestion.odds_ingestion import load_odds, merge_odds_with_match_df

logger = logging.getLogger(__name__)


def load_data(
    live: bool = False,
    validate: bool = True,
) -> pd.DataFrame:
    """
    Load match features and odds for all completed historical seasons.

    Steps:
        1. Build rolling features from DB via build_dataframe()
        2. Merge vig-free odds from football-data.co.uk
        3. Drop rows where all features are missing (early season warmup)
        4. Optionally exclude the current incomplete season (live=False)
        5. Optionally validate FEATURE_COLS against the loaded df

    Args:
        live:     if True, include the current season (2025-26) for
                  live feature lookups in predict.py.
                  if False (default), exclude it — incomplete seasons
                  should never be used as backtest test folds.
        validate: if True, run validate_feature_cols() to catch issues
                  early. Set False only if you need raw data for debugging.

    Returns:
        Match-level DataFrame with all features and odds columns merged.
    """
    logger.info("Building match features from DB...")
    match_df = build_dataframe()

    logger.info("Loading odds data...")
    odds_df = load_odds()
    df = merge_odds_with_match_df(match_df, odds_df)

    # Deduplicate any columns that crept in during the odds merge
    df = df.loc[:, ~df.columns.duplicated()]

    # Drop rows where every feature is NaN
    # These are the first N games of each team's history where rolling
    # windows haven't warmed up yet — XGBoost can't do anything with them
    mask_all_missing = df[FEATURE_COLS].isna().all(axis=1)
    dropped = int(mask_all_missing.sum())
    df = df[~mask_all_missing].copy()

    if dropped:
        logger.info(f"Dropped {dropped} rows with all features missing (rolling warmup)")

    # Exclude current incomplete season from historical backtest/training
    # 2025 = 2025-26 season start year
    # predict.py passes live=True to keep these rows for feature lookups
    if not live:
        before = len(df)
        df = df[df["season_start"] < 2025].copy()
        excluded = before - len(df)
        if excluded:
            logger.info(f"Excluded {excluded} rows from current incomplete season")

    logger.info(f"Total rows loaded: {len(df)}")
    logger.info(f"Seasons: {sorted(df['season_start'].unique())}")

    if validate:
        validate_feature_cols(df, label="load_data")

    return df