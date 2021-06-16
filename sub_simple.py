import xgboost as xgb
import common
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
from common import smape, prep_submission
from train import train_xgb
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer

parser = argparse.ArgumentParser()
parser.add_argument('--num', '-n', type=int, default=-1, help='building number')
parser.add_argument('--tag', '-t', default='test', type=str)
args = parser.parse_args()

dataroot = "./data"

df = None
scores = []

if args.num == -1:
    R = range(1,61)
else:
    R = range(args.num, args.num+1)

for num in R:
    X_train, y_train, X_test, test_idx = common.prep_full(dataroot, num)
    y_train = np.log1p(y_train)

    with open(f"./res_{args.tag}/{num:02d}.json", "r") as inf:
        o = json.load(inf)

    cv = KFold(n_splits=5)
    def smape_expm1(A, F):
        return smape(np.expm1(A[:,None]), np.expm1(F[:,None]))
    scorer = make_scorer(smape_expm1, greater_is_better=False)

    res = train_xgb(o['param'], X_train, y_train, cv, scorer=scorer)
    print(num, res['loss'])
    scores.append(res['loss'])

    preds = []
    for m in res['models']:
        pred = np.expm1(m.predict(X_test))
        preds.append(np.array(pred))
    preds = np.stack(preds).mean(axis=0)

    sub_df = pd.DataFrame(preds, index=test_idx, columns=['answer']).reset_index()
    sub_df['num_date_time'] = sub_df['datetime'].apply(lambda t: t.strftime(f"{num} %Y-%m-%d %H"))
    sub_df.drop('datetime', axis=1)
    sub_df = sub_df[['num_date_time', 'answer']]

    if df is None:
        df = sub_df.copy()
    else:
        df = df.append(sub_df)

print("avg score:", np.stack(scores).mean())
df.to_csv(f'sub_{args.tag}_{common.now()}.csv', index=False)

