#!/bin/bash
# refresh.sh
# ----------
# Weekly data refresh for the EPL over 2.5 goals model.
# Run this once per gameweek before running predict.py.
#
# What it does:
#   1. Downloads latest match results from football-data.co.uk
#   2. Loads new results into the DB
#   3. Fetches fresh xG data from Understat
#   4. Regenerates the odds cache with latest closing odds
#   5. Runs predict.py and outputs bet recommendations
#
# Usage:
#   chmod +x refresh.sh   (first time only)
#   ./refresh.sh

set -e  # Stop immediately if any command fails

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$PROJECT_DIR/venv-soccer"
PYTHON="$VENV/bin/python"
SEASON="2025-26"
SEASON_CODE="2526"
CSV_PATH="$PROJECT_DIR/data/raw/epl/$SEASON/match_data.csv"
ODDS_CACHE="$PROJECT_DIR/data/raw/epl/odds_processed.csv"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

step() { echo -e "\n${GREEN}==>${NC} $1"; }
warn() { echo -e "${YELLOW}WARNING:${NC} $1"; }
fail() { echo -e "${RED}ERROR:${NC} $1"; exit 1; }

# ---------------------------------------------------------------------------
# Step 0 — Sanity checks
# ---------------------------------------------------------------------------
step "Checking environment..."

[ -f "$PYTHON" ] || fail "Python not found at $PYTHON — check VENV path"
[ -f "$PROJECT_DIR/.env" ] || warn ".env file not found — DB credentials may be missing"

step "Running test suite..."

$PYTHON -m pytest tests/ -q --tb=short || fail "Tests failed — aborting refresh. Fix the issue and rerun."

echo "  All tests passed."
cd "$PROJECT_DIR"

# ---------------------------------------------------------------------------
# Step 1 — Download latest match CSV from football-data.co.uk
# ---------------------------------------------------------------------------
step "Downloading latest $SEASON match data..."

mkdir -p "$PROJECT_DIR/data/raw/epl/$SEASON"

curl -s -o "$CSV_PATH" \
    "https://www.football-data.co.uk/mmz4281/$SEASON_CODE/E0.csv"

if [ ! -s "$CSV_PATH" ]; then
    fail "Downloaded CSV is empty — check football-data.co.uk is accessible"
fi

ROW_COUNT=$(tail -n +2 "$CSV_PATH" | grep -c "E0" || true)
echo "  Downloaded $ROW_COUNT matches for $SEASON"

# ---------------------------------------------------------------------------
# Step 2 — Load new results into DB
# ---------------------------------------------------------------------------
step "Loading match results into database..."

LOAD_SCRIPT="$PROJECT_DIR/scripts/load_matches.py"
[ -f "$LOAD_SCRIPT" ] || fail "load_matches.py not found at $LOAD_SCRIPT"

# Back up, update season, run, restore
cp "$LOAD_SCRIPT" "$LOAD_SCRIPT.bak"
sed -i "s/year = \".*\"/year = \"$SEASON\"/" "$LOAD_SCRIPT"
$PYTHON "$LOAD_SCRIPT"
EXIT_CODE=$?
mv "$LOAD_SCRIPT.bak" "$LOAD_SCRIPT"
[ $EXIT_CODE -eq 0 ] || fail "load_matches.py failed"

# ---------------------------------------------------------------------------
# Step 3 — Fetch fresh xG from Understat
# ---------------------------------------------------------------------------
step "Fetching xG data from Understat..."

$PYTHON ingestion/xg_ingestion.py || fail "xg_ingestion.py failed"

# ---------------------------------------------------------------------------
# Step 4 — Regenerate odds cache
# ---------------------------------------------------------------------------
step "Regenerating odds cache..."

if [ -f "$ODDS_CACHE" ]; then
    echo "  Deleting stale odds cache..."
    rm "$ODDS_CACHE"
fi

$PYTHON ingestion/odds_ingestion.py || fail "odds_ingestion.py failed"

# ---------------------------------------------------------------------------
# Step 5 — Run predictions
# ---------------------------------------------------------------------------
step "Generating bet recommendations..."

$PYTHON predict.py

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo -e "\n${GREEN}Refresh complete.${NC}"
echo "Predictions saved to: $PROJECT_DIR/predictions/latest.csv"
echo ""
echo "Next steps:"
echo "  - Check predictions/latest.csv for full fixture table"
echo "  - Run again after each gameweek to keep data current"