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

parser = argparse.ArgumentParser()
parser.add_argument('--num', '-n', type=int, default=4, help='building number')
parser.add_argument('--dataroot', type=str, default="./data")
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--n_splits', type=int, default=5)
parser.add_argument('--tag', '-t', default='test', type=str)
parser.add_argument('--random_state', '-r', default=42, type=int)
args = parser.parse_args()

X_train, y_train, X_test, test_idx = common.prep_full(args.dataroot, args.num)
y_train = np.log1p(y_train)

cv = TimeSeriesSplit(n_splits=5, test_size=24*7)
#cv = TimeSeriesSplit(n_splits=5, test_size=24*7)

p = pathlib.Path(f"./res_{args.tag}")
p.mkdir(exist_ok=True)

def smape_expm1(A, F):
    return smape(np.expm1(A[:,None]), np.expm1(F[:,None]))
scorer = make_scorer(smape_expm1, greater_is_better=False)

best_params, trials = optimize_xgb(X_train, y_train, cv=cv, scorer=scorer, max_evals=50, seed=args.seed)
res = train_xgb(best_params, X_train, y_train, cv, scorer=scorer)
o = {'score': res['loss'], 'num': args.num, 'param': best_params}
print(res['scores'], res['loss'])

with open(p/f"{args.num:02d}.json", "w") as outf:
    json.dump(o, outf)
