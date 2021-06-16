import torch
import argparse
import pathlib
import numpy as np
import pandas as pd
import json
import torch.nn as nn
from torch.optim import Adam

import common
from common import read_df, create_lag_features, smape, TargetTransformer
from train import optimize_xgb, train_xgb
from hyperopt import STATUS_OK
from sklearn.metrics import make_scorer
import pickle
from sklearn.model_selection import KFold, cross_val_score
from torch.utils.data import TensorDataset, DataLoader

def smape_expm1(A, F):
    return smape(np.expm1(A[:,None]), np.expm1(F[:,None]))

parser = argparse.ArgumentParser()
parser.add_argument('--num', '-n', type=int, default=4, help='building number')
parser.add_argument('--dataroot', type=str, default="./data")
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--n_splits', type=int, default=5)
parser.add_argument('--tag', '-t', default='test', type=str)
parser.add_argument('--random_state', '-r', default=42, type=int)
args = parser.parse_args()

X_train, y_train, X_test, test_idx = common.prep_full(args.dataroot, args.num)
cols_to_drop = ['min_temperature', 'max_temperature', 'THI', 'mean_THI', 'CDH', 'mean_CDH','THI_CAT_THI_1', 'THI_CAT_THI_2', 'THI_CAT_THI_3', 'THI_CAT_THI_4']
X_train.drop(cols_to_drop, axis=1, inplace=True)
X_test.drop(cols_to_drop, axis=1, inplace=True)
y_train = np.log1p(y_train)

n_feats = X_train.shape[1]

# model simple feed forward
from torch.nn import Dropout, Linear, ReLU
def build_model(n_sizes, n_out, p=0.2):
    layers = []
    for i in range(len(n_sizes)-1):
        layers.append(Linear(n_sizes[i], n_sizes[i+1]))
        layers.append(Dropout(p))
        layers.append(ReLU(p))
    layers.pop(); layers.pop() # skip last dropout & relu
    layers.append(nn.Linear(n_sizes[i+1], n_out))
    return nn.Sequential(*layers)

n_sizes = (n_feats, 512, 1024, 512, 256)

m = build_model(n_sizes, 1)
m.cuda()

kf = KFold(n_splits=5, shuffle=True, random_state=args.random_state)
train_ix, test_ix = next(kf.split(X_train))

x_tr = X_train.iloc[train_ix].values
x_te = X_train.iloc[test_ix].values
y_tr = y_train.iloc[train_ix].values
y_te = y_train.iloc[test_ix].values

ds = TensorDataset(torch.tensor(x_tr).float(), torch.tensor(y_tr).float())
tr_dl = DataLoader(ds, batch_size=64, shuffle=True)

ds = TensorDataset(torch.tensor(x_te).float(), torch.tensor(y_te).float())
te_dl = DataLoader(ds, batch_size=64, shuffle=True)

opt = Adam(m.parameters())
print(opt)

n_epochs = 3600
loss_fn = torch.nn.MSELoss()

print("*** train loss")
train_losses, test_losses = [], []
for epoch in range(1, n_epochs+1):
    print(epoch, end=",")
    m.train()
    losses = []
    for x, y in tr_dl:
        x = x.to('cuda')
        y = y.to('cuda')
        opt.zero_grad()
        with torch.enable_grad():
            pred = m(x)
        loss = loss_fn(pred.squeeze(), y)
        loss.backward()
        opt.step()
        losses.append(loss.cpu().detach().numpy())
    train_loss = np.array(losses).mean()
    print(train_loss, end=",")
    train_losses.append(train_loss)

    losses = []
    preds = []
    gts = []
    with torch.no_grad():
        for x, y in te_dl:
            x = x.to('cuda')
            y = y.to('cuda')
            pred = m(x)
            gts.append(y)
            preds.append(pred)
            loss = loss_fn(pred.squeeze(), y)
            losses.append(loss.cpu().detach().numpy())
    preds = torch.cat(preds).cpu().numpy().squeeze()
    gts = torch.cat(gts).cpu().numpy().squeeze()

    test_loss = np.array(losses).mean()
    test_losses.append(test_loss)
    print(test_loss, end=",")
    print(smape_expm1(gts, preds))
