import xgboost as xgb
import common
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
from common import smape, prep_submission
from train import train_xgb
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer

parser = argparse.ArgumentParser()
parser.add_argument('--num', '-n', type=int, default=4, help='building number')
parser.add_argument('--tag', '-t', default='test', type=str)
args = parser.parse_args()

def smape_expm1(A, F):
    return smape(np.expm1(A[:,None]), np.expm1(F[:,None]))
scorer = make_scorer(smape_expm1, greater_is_better=False)

NUM2CL = {1: 1, 2: 3, 3: 3, 4: 2, 5: 1, 6: 3, 7: 3, 8: 3, 9: 1, 10: 2, 11: 2, 12: 2, 13: 3, 14: 3, 15: 3, 16: 3, 17: 3, 18: 3, 19: 0, 20: 0, 21: 0, 22: 3, 23: 3, 24: 3, 25: 3, 26: 3, 27: 3, 28: 2, 29: 2, 30: 2, 31: 3, 32: 3, 33: 3, 34: 1, 35: 3, 36: 2, 37: 3, 38: 3, 39: 3, 40: 2, 41: 2, 42: 2, 43: 3, 44: 3, 45: 3, 46: 3, 47: 3, 48: 3, 49: 0, 50: 0, 51: 0, 52: 3, 53: 3, 54: 3, 55: 3, 56: 3, 57: 3, 58: 3, 59: 2, 60: 2}


dataroot = "./data"

df = None
scores = []

X_train, y_train, X_test, test_idx = common.prep_full(dataroot, args.num)
y_train = np.log1p(y_train)

Y = y_train.to_frame()
Y['pred'] = 0

with open(f"./res_{args.tag}/{args.num:02d}.json", "r") as inf:
    o = json.load(inf)

cv = KFold(n_splits=5, shuffle=True, random_state=42)

res = train_xgb(o['param'], X_train, y_train, cv, scorer=scorer)
print(f"{args.num} ({NUM2CL[args.num]})", res['loss'])

scores = []
for m, (train_ix, test_ix) in zip(res['models'], cv.split(X_train)):
    pred = m.predict(X_train.loc[test_ix])
    score = smape_expm1(pred, y_train.iloc[test_ix].values)

    ix = Y.iloc[test_ix].index 
    Y.loc[ix, 'smape'] = 100*2*np.abs(pred - y_train.iloc[test_ix].values)/(np.abs(pred) + np.abs(y_train.iloc[test_ix].values))
    Y.loc[ix, 'pred'] = np.expm1(pred)
    scores.append(score)

Y['target'] = np.expm1(Y.target)
print(np.array(scores).mean())
