import pandas as pd
odds = pd.read_csv("data/raw/epl/odds_processed.csv")

# Drop NaN before sorting
print("NaN home_team rows:", odds["home_team"].isna().sum())
print(sorted(odds["home_team"].dropna().unique()))
print("Total rows:", len(odds))