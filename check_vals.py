import pandas as pd
import common
import numpy as np

A = []
dfs = [None]
df_simples = [None]

gts, preds = [], []
for i in range(1, 61):
    df0 = pd.read_csv(f'./vals/{i:02d}.csv', header=None)
    df1 = pd.read_csv(f'./vals/{i:02d}_simple.csv', header=None)

    a = common.smape(df0.dropna()[1], df0.dropna()[3])
    b = common.smape(df1.dropna()[1], df1.dropna()[2])
    assert df0[1].equals(df1[1])
    gts.append(df0.dropna()[1])
    if a<b:
        preds.append(df0.dropna()[3])
    else:
        preds.append(df1.dropna()[2])
    A.append(a if a<b else b)
    #A.append(b)
    #print(i, f"{a:.3f}", f"{b:.3f}", "rec" if a<b else "simple")
    print(i, "rec" if a<b else "simple")
    dfs.append(df0)
    df_simples.append(df1)

#print(np.array(A).mean())
