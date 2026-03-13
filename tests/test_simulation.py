import pandas as pd

bets = pd.read_csv("/mnt/c/Users/soule/soccer-prop-model/simulation/backtest_results.csv")

# Edge buckets — does higher edge actually predict better outcomes?
bets["edge_bucket"] = pd.cut(bets["edge"], bins=[0, 0.09, 0.11, 0.13, 0.15, 1.0],
                              labels=["0.08-0.09", "0.09-0.11", "0.11-0.13", "0.13-0.15", "0.15+"])

summary = bets.groupby("edge_bucket", observed=True).agg(
    bets       = ("result", "count"),
    wins       = ("result", lambda x: (x == "WIN").sum()),
    hit_rate   = ("result", lambda x: (x == "WIN").mean()),
    avg_odds   = ("odds",   "mean"),
    avg_edge   = ("edge",   "mean"),
).round(3)

summary["expected_hit"] = (1 / summary["avg_odds"]).round(3)
summary["outperf"]      = (summary["hit_rate"] - summary["expected_hit"]).round(3)

print(summary.to_string())

# Also show by season
print("\n--- By season ---")
season_summary = bets.groupby("date").apply(lambda x: x["date"].str[:4]).reset_index()
bets["season"] = bets["date"].str[:4]
print(bets.groupby("season").agg(
    bets     = ("result", "count"),
    hit_rate = ("result", lambda x: (x == "WIN").mean()),
    avg_edge = ("edge",   "mean"),
    avg_odds = ("odds",   "mean"),
).round(3).to_string())