import xgboost as xgb
import common
import argparse
import numpy as np
import pandas as pd
import json
from common import smape, prep_submission

model_sel = {}
with open('model_sel', 'r') as inf:
    for line in inf:
        a, b = line.strip().split()
        model_sel[a] = b

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', type=str, default='./models/model_02_42_2.json')
parser.add_argument('--dataroot', type=str, default='./data')
args = parser.parse_args()

num, seed, nweek = map(int, common.get_model_param(args.model))
print(f"{num=}, {seed=}, {nweek=}")

max_lags = 24*7*nweek
threshold = 0.2

lags = common.get_lags_from_model(args.model)

X_test, target_scaler = common.prep_submission(args.dataroot, num, lags, threshold)
lag_cols = [c for c in X_test.columns if 'lag' in c]
print("lag_cols=", lag_cols)

model = xgb.XGBRegressor()
model.load_model(args.model)

__loc = lambda c: X_test.columns.get_loc(c)
__pred = lambda ix: model.predict(X_test.iloc[ix].values[None,:])[0]
preds = [__pred(0)]

for row_ix in range(1, X_test.shape[0]):
    X_test.iloc[row_ix, __loc('lag_01')] = preds[-1]

    for col, lag in zip(lag_cols[1:], lags[1:]):
        if np.isnan(X_test.iloc[row_ix][col]):
            X_test.iloc[row_ix, __loc(col)] = X_test.iloc[row_ix-lag+1]['lag_01']
    pred = __pred(row_ix)
    preds.append(pred)

preds = np.expm1(np.array(preds))

sub_df = pd.DataFrame(preds, index=X_test.index, columns=['answer']).reset_index()
sub_df['num_date_time'] = sub_df['datetime'].apply(lambda t: t.strftime(f"{num} %Y-%m-%d %H"))
sub_df.drop('datetime', axis=1)
sub_df = sub_df[['num_date_time', 'answer']]
sub_df.to_csv(f'subs/submission_{num:02d}.csv', index=False)

