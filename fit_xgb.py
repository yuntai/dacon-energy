import argparse
import pathlib
import numpy as np
import pandas as pd

import common
from common import read_df, create_lag_features, smape, TargetTransformer
from train import optimize_xgb, train_xgb
from hyperopt import STATUS_OK
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--num', type=int, default=3, help='building number')
parser.add_argument('--dataroot', type=str, default="./data")
parser.add_argument('--nweek', type=int, default=2)
parser.add_argument('--pacfth', type=float, default=0.2)
parser.add_argument('--logtr', type=bool, default=False)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

print(args)

max_lags = args.nweek * 24 * 7
X_train, X_test, y_train, y_test, target_scaler, lag_cols = common.prep(args.dataroot, args.num, max_lags)

#y_train_trf = TargetTransformer(log=args.logtr, detrend=False)
#y_train = y_train_trf.transform(y_train.index, y_train.values)

#y_test_trf = TargetTransformer(log=args.logtr, detrend=False)
#y_test = y_test_trf.transform(y_test.index, y_test.values)

def smape_scale(A, F):
    return smape(target_scaler.inverse_transform(A), target_scaler.inverse_transform(F))

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
xgb_model = res["model"]
fn = common.get_model_fn(args)
print(f"saving to {fn}")
xgb_model.save_model(f"models/{fn}")
preds = xgb_model.predict(X_test)
cv_score = min([f["loss"] for f in trials.results if f["status"] == STATUS_OK])
score = smape_scale(y_test.values, preds)

#print(f"num({args.num}) RMSE CV/test: {cv_score:.4f}/{score:.4f}")

# forecast
feature_lags = [int(f.split("_")[1]) for f in X_train.columns if "lag" in f]
assert 1 in feature_lags

for lag in feature_lags:
    idx = X_test.iloc[lag:].index
    X_test.loc[idx, f"lag_{lag}"] = np.nan

__pred = lambda ix: xgb_model.predict(X_test.iloc[ix].values[None,:])[0]
pred = __pred(0)
fpreds = [pred]
__loc = lambda c: X_test.columns.get_loc(c)
for row_ix in range(1, X_test.shape[0]):
    X_test.iloc[row_ix, __loc('lag_1')] = pred

    for col, lag in zip(lag_cols[1:], feature_lags[1:]):
        if np.isnan(X_test.iloc[row_ix][col]):
            X_test.iloc[row_ix, __loc(col)] = X_test.iloc[row_ix-lag+1]['lag_1']
    pred = __pred(row_ix)
    fpreds.append(pred)

fpreds = np.array(fpreds)
fscore = smape(target_scaler.inverse_transform(fpreds), target_scaler.inverse_transform(y_test.values.squeeze()))
print(f"{args.num=},{cv_score=:.4f},{score=:.4f},{fscore=:.4f}")

res = y_test.to_frame()
res.columns = ['gt']
res['pred'] = preds
res['fpred'] = fpreds
res[res.columns] = target_scaler.inverse_transform(res)
res.to_csv(f'vals/{args.num:02d}.csv', index=False)
