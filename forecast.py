import xgboost as xgb
import common
import argparse
import numpy as np
from common import forecast_multi_recursive
from common import smape

parser = argparse.ArgumentParser()
parser.add_argument('--model_fn', type=str, default='models/model_3_42_3.json')
parser.add_argument('--dataroot', type=str, default='./data')
args = parser.parse_args()

num, seed, nweek = map(int, common.get_model_param(args.model_fn))
print(f"{num=}, {seed=}, {nweek=}")
X_train, X_test, y_train, y_test, target_scaler = common.prep(args.dataroot, nweek, num)

lag_columns = [f for f in X_train.columns if "lag" in f]
feature_lags = [int(f.split("_")[1]) for f in X_train.columns if "lag" in f]
assert 1 in feature_lags
lag_col_ixs = [list(X_train.columns).index(col) for col in lag_columns]

assert max(feature_lags) > X_test.shape[1]

for lag in feature_lags:
    idx = X_test.iloc[lag:].index
    X_test.loc[idx, f"lag_{lag}"] = np.nan

model = xgb.XGBRegressor()
model.load_model(args.model_fn)


__pred = lambda ix: model.predict(X_test.iloc[ix].values[None,:])[0]
pred = __pred(0)
preds = [pred]
__loc = lambda c: X_test.columns.get_loc(c)
for row_ix in range(1, X_test.shape[0]):
    X_test.iloc[row_ix, __loc('lag_1')] = pred

    for col, lag in zip(lag_columns[1:], feature_lags[1:]):
        if np.isnan(X_test.iloc[row_ix][col]):
            X_test.iloc[row_ix, __loc(col)] = X_test.iloc[row_ix-lag+1]['lag_1']
    pred = __pred(row_ix)
    preds.append(pred)

preds = target_scaler.inverse_transform(np.array(preds))
print(smape(preds, target_scaler.inverse_transform(y_test.values.squeeze())))
