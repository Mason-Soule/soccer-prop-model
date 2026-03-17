import pandas as pd
odds = pd.read_csv("data/raw/epl/odds_processed.csv")
print(odds[["odds_over_2_5", "odds_under_2_5"]].describe())
print(odds["odds_under_2_5"].isna().sum(), "missing under odds")