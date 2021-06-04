import xgboost as xgb
import common
import argparse
import numpy as np
import pandas as pd
from common import forecast_multi_recursive
from common import smape

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', type=str, default='./models/model_3_42_3.json')
parser.add_argument('--dataroot', type=str, default='./data')
args = parser.parse_args()

num, seed, nweek = map(int, common.get_model_param(args.model))
X_train, test_df, target_scaler = common.prep_submission(args.dataroot, nweek, num)

lag_columns = [f for f in X_train.columns if "lag" in f]
feature_lags = [int(f.split("_")[1]) for f in X_train.columns if "lag" in f]
assert 1 in feature_lags
lag_col_ixs = [list(X_train.columns).index(col) for col in lag_columns]

for lag in feature_lags:
    idx = test_df.iloc[lag:].index
    test_df.loc[idx, f"lag_{lag}"] = np.nan

model = xgb.XGBRegressor()
model.load_model(args.model)

__pred = lambda ix: model.predict(test_df.iloc[ix].values[None,:])[0]
pred = __pred(0)
preds = [pred]
__loc = lambda c: test_df.columns.get_loc(c)
for row_ix in range(1, test_df.shape[0]):
    test_df.iloc[row_ix, __loc('lag_1')] = pred

    for col, lag in zip(lag_columns[1:], feature_lags[1:]):
        if np.isnan(test_df.iloc[row_ix][col]):
            test_df.iloc[row_ix, __loc(col)] = test_df.iloc[row_ix-lag+1]['lag_1']
    pred = __pred(row_ix)
    preds.append(pred)

preds = target_scaler.inverse_transform(np.array(preds))
sub_df = pd.DataFrame(preds, index=test_df.index, columns=['answer']).reset_index()
sub_df['num_date_time'] = sub_df['datetime'].apply(lambda t: t.strftime(f"{num} %Y-%m-%d %H"))
sub_df.drop('datetime', axis=1)
sub_df = sub_df[['num_date_time', 'answer']]
sub_df.to_csv(f'subs/submission_{num}.csv', index=False)

