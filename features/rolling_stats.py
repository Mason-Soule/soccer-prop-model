def avg_goals_last5(df, window=5):
    df = df.sort_values(["team_id", "date"])

    df["avg_goals_last5"] = (
        df.groupby("team_id")["goals"]
          .shift(1)
          .rolling(window)
          .mean()
    )

    return df

def avg_goals_conceded_last5(df, window=5):
    df = df.sort_values(["team_id", "date"])

    df["avg_goals_conceded_last5"] = (
        df.groupby("team_id")["goals_conceded"]
          .shift(1)
          .rolling(window)
          .mean()
    )

    return df

def avg_shots_last5(df, window=5):
    df = df.sort_values(["team_id", "date"])

    df["avg_shots_last5"] = (
        df.groupby("team_id")["shots"]
          .shift(1)
          .rolling(window)
          .mean()
    )

    return df

def avg_shots_conceded_last5(df, window=5):
    df = df.sort_values(["team_id", "date"])

    df["avg_shots_conceded_last5"] = (
        df.groupby("team_id")["shots_conceded"]
          .shift(1)
          .rolling(window)
          .mean()
    )

    return df
def avg_shots_conceded_last5(df, window=5):
    df = df.sort_values(["team_id", "date"])

    df["avg_shots_conceded_last5"] = (
        df.groupby("team_id")["shots_conceded"]
          .shift(1)
          .rolling(window)
          .mean()
    )

    return df
