-- Minimal schema for data validation stage
-- Focus: Support get_player_history function

-- Teams table
CREATE TABLE IF NOT EXISTS teams (
    team_id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    league VARCHAR(50) NOT NULL,
    UNIQUE(name, league)
);

-- Matches table (date used for temporal filtering)
CREATE TABLE IF NOT EXISTS matches (
    match_id SERIAL PRIMARY KEY,
    date TIMESTAMP WITH TIME ZONE NOT NULL,
    home_team_id INTEGER NOT NULL REFERENCES teams(team_id),
    away_team_id INTEGER NOT NULL REFERENCES teams(team_id),
    season VARCHAR(20) NOT NULL,
    league VARCHAR(50) NOT NULL,
    referee VARCHAR(50) NOT NULL
);

CREATE INDEX idx_matches_date ON matches(date);

-- Players table
CREATE TABLE IF NOT EXISTS players (
    player_id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    team_id INTEGER REFERENCES teams(team_id)
);

-- Player matches table (aggregated stats per match)
CREATE TABLE IF NOT EXISTS player_matches (
    player_match_id SERIAL PRIMARY KEY,
    player_id INTEGER NOT NULL REFERENCES players(player_id),
    match_id INTEGER NOT NULL REFERENCES matches(match_id),
    minutes_played INTEGER NOT NULL DEFAULT 0,
    shots INTEGER NOT NULL DEFAULT 0,
    shots_on_target INTEGER NOT NULL DEFAULT 0,
    goals INTEGER NOT NULL DEFAULT 0,
    assists INTEGER DEFAULT 0,
    UNIQUE(player_id, match_id)
);

-- Create team match stats
CREATE TABLE IF NOT  EXISTS team_match_stats (
    team_id INTEGER REFERENCES teams(team_id),
    match_id INTEGER REFERENCES matches(match_id),
    is_home BOOLEAN NOT NULL,
    goals INTEGER,
    shots INTEGER,
    shots_on_target INTEGER,
    fouls INTEGER,
    corners INTEGER,
    yellow_cards INTEGER,
    red_cards INTEGER,
    UNIQUE(team_id, match_id)
);

CREATE INDEX idx_player_matches_player ON player_matches(player_id);
CREATE INDEX idx_player_matches_match ON player_matches(match_id);
