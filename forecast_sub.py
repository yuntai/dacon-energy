import xgboost as xgb
import common
import argparse
import numpy as np
import pandas as pd
from common import smape, prep_submission

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', type=str, default='./models/model_2_42_3.json')
parser.add_argument('--dataroot', type=str, default='./data')
args = parser.parse_args()

num, seed, nweek = map(int, common.get_model_param(args.model))
max_lags = 24*7*nweek
threshold = 0.2

X_test, target_scaler, lag_cols = common.prep_submission(args.dataroot, num, max_lags, threshold)

feature_lags = [int(f.split("_")[1]) for f in lag_cols if "lag" in f]

model = xgb.XGBRegressor()
model.load_model(args.model)

__loc = lambda c: X_test.columns.get_loc(c)
__pred = lambda ix: model.predict(X_test.iloc[ix].values[None,:])[0]
preds = [__pred(0)]

for row_ix in range(1, X_test.shape[0]):
    X_test.iloc[row_ix, __loc('lag_1')] = preds[-1]

    for col, lag in zip(lag_cols[1:], feature_lags[1:]):
        if np.isnan(X_test.iloc[row_ix][col]):
            X_test.iloc[row_ix, __loc(col)] = X_test.iloc[row_ix-lag+1]['lag_1']
    pred = __pred(row_ix)
    preds.append(pred)

preds = target_scaler.inverse_transform(np.array(preds))

sub_df = pd.DataFrame(preds, index=X_test.index, columns=['answer']).reset_index()
sub_df['num_date_time'] = sub_df['datetime'].apply(lambda t: t.strftime(f"{num} %Y-%m-%d %H"))
sub_df.drop('datetime', axis=1)
sub_df = sub_df[['num_date_time', 'answer']]
sub_df.to_csv(f'subs/submission_{num}.csv', index=False)

