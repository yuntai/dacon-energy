import argparse
import pathlib
import numpy as np
import pandas as pd

import common
from common import read_df, create_lag_features, smape, TargetTransformer
from train import optimize_xgb, train_xgb, fit_xgb
from hyperopt import STATUS_OK
from sklearn.metrics import make_scorer
import pickle
from sklearn.model_selection import TimeSeriesSplit, KFold

parser = argparse.ArgumentParser()
parser.add_argument('--num', '-n', type=int, default=4, help='building number')
parser.add_argument('--dataroot', type=str, default="./data")
parser.add_argument('--nweek', type=int, default=2)
parser.add_argument('--pacfth', type=float, default=0.2)
parser.add_argument('--logtr', type=bool, default=False)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--pacf_threshold', '-t', type=float, default=0.2)
parser.add_argument('--test_size', type=str, default='2W')
parser.add_argument('--simple', type=bool, action=argparse.BooleanOptionalAction)
parser.add_argument('--n_splits', type=int, default=5)
args = parser.parse_args()

def run_recursive_parallel(models, X_test, y_test, feature_lags):
    for lag in feature_lags:
        idx = X_test.iloc[lag:].index
        X_test.loc[idx, __get_lag_col(lag)] = np.nan

    __loc = lambda c: X_test.columns.get_loc(c)
    def __pred(ix):
        preds = []
        for m in models:
            p = m.predict(X_test.iloc[ix].values[None,:]).squeeze().item()
            preds.append(p)
        return  np.array(preds).mean()

    fpreds = [__pred(0)]

    for row_ix in range(1, X_test.shape[0]):
        X_test.iloc[row_ix, __loc('lag_01')] = fpreds[-1]

        for lag in feature_lags[1:]:
            col = __get_lag_col(lag)
            if np.isnan(X_test.iloc[row_ix][col]):
                X_test.iloc[row_ix, __loc(col)] = X_test.iloc[row_ix-lag+1]['lag_01']
        fpreds.append(__pred(row_ix))

    return np.array(fpreds)

def __get_lag_col(lags):
    return f'lag_{lags:02d}'

print(args)

max_lags = args.nweek * 24 * 7
print(f"{max_lags=}")

#X_train, X_test, y_train, y_test, feature_lags = common.prep(args.dataroot, args.num, max_lags, test_size='2W', threshold=args.pacf_threshold)
X_train, X_test, y_train, y_test, feature_lags = common.prep(args.dataroot, args.num, max_lags, test_size=-1, threshold=args.pacf_threshold)
lag_cols = [__get_lag_col(l) for l in feature_lags]
print("lags: ", feature_lags, len(feature_lags))

def smape_scale(A, F):
    return smape(np.expm1(A[:,None]), np.expm1(F[:,None]))

scorer = make_scorer(smape_scale, greater_is_better=False)

cv = TimeSeriesSplit(n_splits=args.n_splits)

best, trials = optimize_xgb(X_train, y_train, max_evals=50, cv=cv, scorer=scorer, seed=args.seed)
res = fit_xgb(best, X_train, y_train, seed=args.seed)

models = res['models'][-1:]
preds = run_recursive_parallel(models, X_test.copy(), y_test.copy(), feature_lags)
pd.DataFrame({'pred': np.expm1(preds)}).to_csv(f'./res_re/{args.num:02d}.csv', index=None, header=False)
