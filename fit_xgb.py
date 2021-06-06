import argparse
import pathlib
import numpy as np
import pandas as pd

import common
from common import read_df, create_lag_features, smape, TargetTransformer
from train import optimize_xgb, train_xgb
from hyperopt import STATUS_OK
from sklearn.metrics import make_scorer
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--num', '-n', type=int, default=4, help='building number')
parser.add_argument('--dataroot', type=str, default="./data")
parser.add_argument('--nweek', type=int, default=2)
parser.add_argument('--pacfth', type=float, default=0.2)
parser.add_argument('--logtr', type=bool, default=False)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--pacf_threshold', '-t', type=float, default=0.2)
parser.add_argument('--test_size', type=str, default='2W')
args = parser.parse_args()

print(args)

def __get_lag_col(lags):
    return f'lag_{lags:02d}'

max_lags = args.nweek * 24 * 7
print(f"{max_lags=}")

X_train, X_test, y_train, y_test, target_scaler, feature_lags = common.prep(args.dataroot, args.num, max_lags, test_size='2W', threshold=args.pacf_threshold)
print("lags: ", feature_lags, len(feature_lags))

def smape_scale(A, F):
    #return smape(target_scaler.inverse_transform(np.expm1(A[:,None])), target_scaler.inverse_transform(np.expm1(F[:,None])))
    #return smape(np.expm1(target_scaler.inverse_transform(A[:,None])), np.expm1(target_scaler.inverse_transform(F[:,None])))
    return smape(np.expm1(A[:,None]), np.expm1(F[:,None]))

if args.logtr:
    assert 0
    #scorer = make_scorer(smapem1, greater_is_better=False)
else:
    #scorer = make_scorer(smape, greater_is_better=False)
    scorer = make_scorer(smape_scale, greater_is_better=False)

best, trials = optimize_xgb(X_train, y_train, max_evals=50, scorer=scorer, seed=args.seed)

# evaluate the best model on the test set
res = train_xgb(best, X_test, y_test, scorer, seed=args.seed)
print(res)
model = res["model"]
fn = common.get_model_fn(args)
print(f"saving to {fn}")
model.save_model(f"models/{fn}")
preds = model.predict(X_test)
cv_score = min([f["loss"] for f in trials.results if f["status"] == STATUS_OK])
score = smape_scale(y_test.values, preds)

# forecast
for lag in feature_lags:
    idx = X_test.iloc[lag:].index
    X_test.loc[idx, __get_lag_col(lag)] = np.nan

__loc = lambda c: X_test.columns.get_loc(c)
__pred = lambda ix: model.predict(X_test.iloc[ix].values[None,:]).squeeze().item()
fpreds = [__pred(0)]

for row_ix in range(1, X_test.shape[0]):
    X_test.iloc[row_ix, __loc('lag_01')] = fpreds[-1]

    for lag in feature_lags[1:]:
        col = __get_lag_col(lag)
        if np.isnan(X_test.iloc[row_ix][col]):
            X_test.iloc[row_ix, __loc(col)] = X_test.iloc[row_ix-lag+1]['lag_01']
    fpreds.append(__pred(row_ix))

fscore = smape_scale(y_test.values, np.array(fpreds))
print(f"{args.num=},{cv_score=:.4f},{score=:.4f},{fscore=:.4f}")

y = pd.concat([y_train, y_test])

res = y.to_frame()
res.columns = ['gt']
res.loc[y_test.index, 'pred'] = preds
res.loc[y_test.index, 'fpred'] = fpreds
res[res.columns] = np.expm1(res)
res['num'] = args.num
res = res[['num', 'gt', 'pred', 'fpred']]
res.to_csv(f'vals/{args.num:02d}.csv', index=False, header=False)
