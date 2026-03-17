"""
scripts/migrate_schema.py
-------------------------
Safe schema improvements for the soccer props database.

Changes made:
    1. Add NOT NULL constraints to team_match_stats.team_id and match_id
    2. Add primary key to team_match_stats
    3. Add index on team_match_stats.team_id (speeds up rolling feature queries)
    4. Add index on team_match_stats.match_id (speeds up match lookups)
    5. Add index on matches.season (speeds up season filtering in build_dataset)
    6. Widen matches.referee to varchar(100) — some names exceed 50 chars
    7. Add index on matches(home_team_id, away_team_id) for H2H lookups

Each change is wrapped in a transaction so failures roll back cleanly.
Safe to run multiple times — uses IF NOT EXISTS and DO NOTHING patterns.

Usage:
    python scripts/migrate_schema.py
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv
import psycopg2

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def get_conn():
    return psycopg2.connect(
        dbname   = os.getenv("DB_NAME"),
        user     = os.getenv("DB_USER"),
        password = os.getenv("DB_PASSWORD"),
        host     = "localhost",
        port     = "5432",
    )


def run_migration(conn, description: str, sql: str) -> bool:
    """
    Run a single migration step.
    Returns True if successful, False if skipped (already applied).
    Rolls back on failure so the DB stays consistent.
    """
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
        logger.info(f"  ✓ {description}")
        return True
    except psycopg2.errors.DuplicateTable:
        conn.rollback()
        logger.info(f"  — {description} (already exists, skipped)")
        return False
    except psycopg2.errors.DuplicateObject:
        conn.rollback()
        logger.info(f"  — {description} (already exists, skipped)")
        return False
    except psycopg2.errors.NotNullViolation as e:
        conn.rollback()
        logger.warning(f"  ✗ {description} — NOT NULL violation: {e}")
        logger.warning("    Fix the data first, then rerun.")
        return False
    except Exception as e:
        conn.rollback()
        logger.error(f"  ✗ {description} — {e}")
        return False


def check_nulls(conn) -> bool:
    """
    Check for NULL team_id or match_id in team_match_stats before
    adding NOT NULL constraints. Reports any problems found.
    Returns True if safe to proceed.
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) FROM team_match_stats
            WHERE team_id IS NULL OR match_id IS NULL
        """)
        null_count = cur.fetchone()[0]

    if null_count > 0:
        logger.warning(
            f"Found {null_count} rows with NULL team_id or match_id "
            f"in team_match_stats. Fix these before adding NOT NULL constraints."
        )
        return False

    logger.info(f"  ✓ No NULL team_id or match_id found — safe to add NOT NULL")
    return True


def check_pk_conflict(conn) -> bool:
    """
    Check if the existing unique constraint would conflict with a PK.
    Returns True if safe to add primary key.
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) FROM (
                SELECT team_id, match_id, COUNT(*)
                FROM team_match_stats
                GROUP BY team_id, match_id
                HAVING COUNT(*) > 1
            ) dups
        """)
        dup_count = cur.fetchone()[0]

    if dup_count > 0:
        logger.warning(
            f"Found {dup_count} duplicate (team_id, match_id) combinations. "
            f"Cannot add primary key until duplicates are resolved."
        )
        return False

    logger.info(f"  ✓ No duplicate (team_id, match_id) pairs — safe to add PK")
    return True


def run_all_migrations():
    conn = get_conn()
    logger.info("Connected to database.")
    logger.info(f"Database: {os.getenv('DB_NAME')}\n")

    # ------------------------------------------------------------------
    # Pre-flight checks
    # ------------------------------------------------------------------
    logger.info("Running pre-flight checks...")

    nulls_ok  = check_nulls(conn)
    pk_ok     = check_pk_conflict(conn)

    print()

    # ------------------------------------------------------------------
    # Migration 1 — NOT NULL constraints on team_match_stats
    # ------------------------------------------------------------------
    logger.info("Step 1: Adding NOT NULL constraints to team_match_stats...")

    if nulls_ok:
        run_migration(
            conn,
            "NOT NULL on team_match_stats.team_id",
            """
            ALTER TABLE team_match_stats
            ALTER COLUMN team_id SET NOT NULL
            """
        )
        run_migration(
            conn,
            "NOT NULL on team_match_stats.match_id",
            """
            ALTER TABLE team_match_stats
            ALTER COLUMN match_id SET NOT NULL
            """
        )
    else:
        logger.warning("  Skipping NOT NULL constraints — fix NULLs first")

    # ------------------------------------------------------------------
    # Migration 2 — Primary key on team_match_stats
    # ------------------------------------------------------------------
    logger.info("\nStep 2: Adding primary key to team_match_stats...")

    if pk_ok and nulls_ok:
        # Drop the existing unique constraint first — PK replaces it
        run_migration(
            conn,
            "Drop existing unique constraint (will be replaced by PK)",
            """
            ALTER TABLE team_match_stats
            DROP CONSTRAINT IF EXISTS team_match_stats_team_id_match_id_key
            """
        )
        run_migration(
            conn,
            "Add PRIMARY KEY (team_id, match_id) to team_match_stats",
            """
            ALTER TABLE team_match_stats
            ADD CONSTRAINT team_match_stats_pkey
            PRIMARY KEY (team_id, match_id)
            """
        )
    else:
        logger.warning("  Skipping primary key — pre-flight checks failed")

    # ------------------------------------------------------------------
    # Migration 3 — Indexes on team_match_stats
    # ------------------------------------------------------------------
    logger.info("\nStep 3: Adding indexes to team_match_stats...")

    run_migration(
        conn,
        "Index on team_match_stats.team_id",
        """
        CREATE INDEX IF NOT EXISTS idx_team_match_stats_team_id
        ON team_match_stats(team_id)
        """
    )

    run_migration(
        conn,
        "Index on team_match_stats.match_id",
        """
        CREATE INDEX IF NOT EXISTS idx_team_match_stats_match_id
        ON team_match_stats(match_id)
        """
    )

    run_migration(
        conn,
        "Index on team_match_stats(team_id, match_id) for join queries",
        """
        CREATE INDEX IF NOT EXISTS idx_team_match_stats_team_match
        ON team_match_stats(team_id, match_id)
        """
    )

    # ------------------------------------------------------------------
    # Migration 4 — Index on matches.season
    # ------------------------------------------------------------------
    logger.info("\nStep 4: Adding index on matches.season...")

    run_migration(
        conn,
        "Index on matches.season",
        """
        CREATE INDEX IF NOT EXISTS idx_matches_season
        ON matches(season)
        """
    )

    # ------------------------------------------------------------------
    # Migration 5 — Index on matches(home_team_id, away_team_id)
    # ------------------------------------------------------------------
    logger.info("\nStep 5: Adding H2H index on matches...")

    run_migration(
        conn,
        "Index on matches(home_team_id, away_team_id) for H2H lookups",
        """
        CREATE INDEX IF NOT EXISTS idx_matches_teams
        ON matches(home_team_id, away_team_id)
        """
    )

    # ------------------------------------------------------------------
    # Migration 6 — Widen referee column
    # ------------------------------------------------------------------
    logger.info("\nStep 6: Widening matches.referee to varchar(100)...")

    run_migration(
        conn,
        "Widen matches.referee from varchar(50) to varchar(100)",
        """
        ALTER TABLE matches
        ALTER COLUMN referee TYPE varchar(100)
        """
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info("\nVerifying final schema...")

    with conn.cursor() as cur:
        # Show indexes on team_match_stats
        cur.execute("""
            SELECT indexname, indexdef
            FROM pg_indexes
            WHERE tablename = 'team_match_stats'
            ORDER BY indexname
        """)
        tms_indexes = cur.fetchall()

        # Show indexes on matches
        cur.execute("""
            SELECT indexname, indexdef
            FROM pg_indexes
            WHERE tablename = 'matches'
            ORDER BY indexname
        """)
        match_indexes = cur.fetchall()

        # Check referee column width
        cur.execute("""
            SELECT character_maximum_length
            FROM information_schema.columns
            WHERE table_name = 'matches'
            AND column_name = 'referee'
        """)
        ref_width = cur.fetchone()[0]

    print("\n--- team_match_stats indexes ---")
    for name, defn in tms_indexes:
        print(f"  {name}")

    print("\n--- matches indexes ---")
    for name, defn in match_indexes:
        print(f"  {name}")

    print(f"\n--- matches.referee width: varchar({ref_width}) ---")

    conn.close()
    logger.info("\nMigration complete.")


if __name__ == "__main__":
    run_all_migrations()