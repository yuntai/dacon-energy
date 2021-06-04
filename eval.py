import xgboost as xgb
import common
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', type=str, default='models/model_2_42_3.json')
parser.add_argument('--dataroot', type=str, default='./data')
args = parser.parse_args()

num, seed, nweek = map(int, common.get_model_param(args.model))
#print(f"{num=}, {seed=}, {nweek=}")
X_train, X_test, y_train, y_test, target_scaler, num_lag_features = common.prep(args.dataroot, nweek, num)

model = xgb.XGBRegressor()
model.load_model(args.model)
preds = model.predict(X_test)
score = common.smape(target_scaler.inverse_transform(y_test.values), target_scaler.inverse_transform(preds))
#print(f"num({num}) smape({score:.4f})")
print(f"{num},{nweek},{seed},{score}")
res = y_test.to_frame()
res.columns = ['gt']
res['pred'] = preds
res[res.columns] = target_scaler.inverse_transform(res)

#import seaborn as sns
#print(y_test.to_dataframe())
