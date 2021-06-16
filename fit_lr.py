import argparse
import pathlib
import numpy as np
import pandas as pd
import json

import common
from common import read_df, create_lag_features, smape, TargetTransformer
from train import optimize_xgb, train_xgb
from hyperopt import STATUS_OK
from sklearn.metrics import make_scorer
import pickle
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str, default="./data")
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--n_splits', type=int, default=5)
parser.add_argument('--tag', '-t', default='test', type=str)
args = parser.parse_args()

cv = KFold(n_splits=args.n_splits, shuffle=False)
p = pathlib.Path(f"./res_{args.tag}")
p.mkdir(exist_ok=True)

def smape_expm1(A, F):
    return smape(np.expm1(A[:,None]), np.expm1(F[:,None]))
scorer = make_scorer(smape_expm1, greater_is_better=False)

scores = []
for n in range(1, 61):
    X_train, y_train, X_test, test_idx = common.prep_full(args.dataroot, n)
    y_train = np.log1p(y_train)

    gts, preds = [], []
    for train_ix, test_ix in cv.split(X_train):
        reg = LinearRegression().fit(X_train.iloc[train_ix], y_train.iloc[train_ix])
        score = reg.score(X_train, y_train)
        gt = y_train.iloc[test_ix].values.squeeze()
        pred = reg.predict(X_train.iloc[test_ix].values.squeeze())
        gts.append(gt)
        preds.append(pred)
    gt = np.concatenate(gts)
    pred = np.concatenate(preds)
    scores.append(smape_expm1(gt, pred))
    print(n, score, "smape=", smape_expm1(gt, pred))
print("mean smape=", np.array(scores).mean())

