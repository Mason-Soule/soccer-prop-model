import pandas as pd
df = pd.read_csv("simulation/backtest_results.csv")
print(df.groupby(pd.cut(df["odds"], bins=[1.0, 1.6, 1.7, 1.8, 1.9]))
       .agg(bets=("result","count"),
            hit_rate=("result", lambda x: (x=="WIN").mean()),
            pnl=("stake", lambda x: (
                df.loc[x.index].apply(
                    lambda r: r["stake"]*(r["odds"]-1) 
                    if r["result"]=="WIN" else -r["stake"], axis=1
                ).sum()
            ))).round(3))