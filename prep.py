import argparse
import pathlib
import torch
import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from common import read_df, create_lag_features, smape, TargetTransformer, smapem1
from train import optimize_xgb, train_xgb
from hyperopt import STATUS_OK
from sklearn.metrics import make_scorer

parser = argparse.ArgumentParser()
parser.add_argument('--num', type=int, default=3, help='building number')
parser.add_argument('--dataroot', type=str, default="./data")
parser.add_argument('--nweek', type=int, default=3)
parser.add_argument('--pacfth', type=float, default=0.2)
parser.add_argument('--logtr', type=bool, default=False)
args = parser.parse_args()

print(args)

train_df, test_df = read_df(pathlib.Path(args.dataroot))

df = train_df[train_df.num==args.num].set_index('datetime').asfreq('1H', 'bfill')
df = df.drop(['date','num','nelec_cool_flag','solar_flag'], axis=1)
df['target'] = df['target']

nlags = 24*7*args.nweek

lags = create_lag_features(df.target, nlags, 0.2)
print("lags feature size=", lags.shape[1])
features = df.join(lags, how="outer").dropna()
target = features.target
features = features.drop('target', axis=1)
X_train, X_test, y_train, y_test = train_test_split(features,
                                                    target,
                                                    test_size=0.3,
                                                    shuffle=False)
pickle.dump((X_train, X_test, y_train, y_test), 'prep.pkl')

y_train_trf = TargetTransformer(log=args.logtr, detrend=False)
y_train = y_train_trf.transform(y_train.index, y_train.values)

y_test_trf = TargetTransformer(log=args.logtr, detrend=False)
y_test = y_test_trf.transform(y_test.index, y_test.values)

if args.logtr:
    scorer = make_scorer(smapem1, greater_is_better=False)
else:
    scorer = make_scorer(smape, greater_is_better=False)

best, trials = optimize_xgb(X_train, y_train, max_evals=50, scorer=scorer)


# evaluate the best model on the test set
res = train_xgb(best, X_test, y_test, scorer)
xgb_model = res["model"]
xgb_model.save_model(f"model_{args.num}.json")
preds = xgb_model.predict(X_test)
cv_score = min([f["loss"] for f in trials.results if f["status"] == STATUS_OK])
score_fn = smape if not args.logtr else smapem1
smape_score = score_fn(y_test.values, preds)

print(f"num({args.num}) RMSE CV/test: {cv_score:.4f}/{smape_score:.4f}")
