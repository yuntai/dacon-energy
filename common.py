import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from statsmodels.tsa.stattools import pacf
from sklearn.model_selection import train_test_split
import pathlib
import os
import json
import datetime

CATE_COLS = ['num', "mgrp", 'holiday', 'dow', 'cluster', 'hot', 'nelec_cool_flag', 'solar_flag']

cluster = {
    0: [19, 20, 21, 49, 50, 51],
    1: [1, 5, 9, 34],
    2: [4, 10, 11, 12, 28, 29, 30, 36, 40, 41, 42, 59, 60],
    3: [2, 3, 6, 7, 8, 13, 14, 15, 16, 17, 18, 22, 23, 24, 25, 26, 27, 31, 32, 33, 35, 37, 38, 39, 43, 44, 45, 46, 47, 48, 52, 53, 54, 55, 56, 57, 58],
}

def date_prep(df):

    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['datetime'].dt.hour
    df['dow'] = df['datetime'].dt.weekday
    df['date'] = df['datetime'].dt.date.astype('str')
    df['day'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month

    special_days = ['2020-06-06', '2020-08-15', '2020-08-17']
    df['holiday'] = df['dow'].isin([5,6]).astype(int)
    df.loc[df.date.isin(special_days), 'holiday'] = 1

    hot = df.groupby('date').first()['holiday'].shift(-1).fillna(0).astype(int)
    hot = hot.to_frame().reset_index().rename({'holiday': "hot"}, axis=1)
    df = df.merge(hot, on='date', how='left')

    h = (df.groupby('date').first()['holiday'] != 0).iloc[::-1]
    df1 = h.cumsum() - h.cumsum().where(~h).ffill().fillna(0).astype(int).iloc[::-1]
    df1 = df1.to_frame().reset_index().rename({'holiday': "cumhol"}, axis=1)
    df = df.merge(df1, on='date', how='left')

    #df['cumhol'] = df.cumhol.astype(str).astype('category')

    return df

def read_df(dataroot):
    train_columns = ['num','datetime','target','temperature','windspeed','humidity','precipitation','insolation','nelec_cool_flag','solar_flag']
    test_columns = [c for c in train_columns if c != 'target']

    dataroot = pathlib.Path(dataroot) if type(dataroot) == str else dataroot

    train_df = pd.read_csv(dataroot/'train.csv', skiprows=[0], names=train_columns)
    test_df = pd.read_csv(dataroot/'test.csv', skiprows=[0], names=test_columns)

    __sz = train_df.shape[0]

    df = pd.concat([train_df, test_df])

    for k, nums in cluster.items():
        df.loc[df.num.isin(nums), 'cluster'] = k

    df = date_prep(df)

    return df.iloc[:__sz].copy(), df.iloc[__sz:].copy()

def interpolate(test_df):
    methods = {
        'temperature': 'quadratic',
        'windspeed':'linear',
        'humidity':'quadratic',
        'precipitation':'linear',
        'insolation': 'pad'
    }

    for col, method in methods.items():
        test_df[col] = test_df[col].interpolate(method=method)
        if method == 'quadratic':
            test_df[col] = test_df[col].interpolate(method='linear')

def prep_tst(dataroot):
    train_df, test_df = read_df(dataroot)
    test_df = test_df.drop(['nelec_cool_flag','solar_flag'], axis=1)

    # interpolate na in test_df
    test_df = test_df.merge(train_df.groupby("num").first()[['nelec_cool_flag','solar_flag']].reset_index(), on="num", how="left")
    interpolate(test_df)

    s = train_df[train_df.datetime=='2020-06-01 00:00:00'].groupby(['temperature', 'windspeed']).ngroup()
    s.name = 'mgrp'
    mgrps = train_df[['num']].join(s, how='inner')

    sz = train_df.shape[0]

    df = pd.concat([train_df, test_df])
    df = df.merge(mgrps, on='num', how='left')

    df = add_feats(df)

    df["log_target"] = np.log(df.target + 1e-8)

    for col in CATE_COLS:
        df[col] = df[col].astype(str).astype('category')

    __ix = df.columns.get_loc('datetime')
    df['time_idx'] = (df.loc[:, 'datetime'] - df.iloc[0, __ix]).astype('timedelta64[h]').astype('int')

    train_df = df.iloc[:sz].copy()
    test_df = df.iloc[sz:].copy()

    return train_df, test_df

def add_feats(df):
    df.reset_index(drop=True, inplace=True)

    #df['THI'] = 9/5*df['temperature'] - 0.55*(1-df['humidity']/100)*(9/5*df['temperature']-26) + 32

    #for num, g in df.groupby('num'):
    #    cdh = (g['temperature']-26).rolling(window=12, min_periods=1).sum()
    #    df.loc[cdh.index, 'CDH'] = cdh

    #cols = ['temperature', 'windspeed', 'humidity', 'precipitation', 'insolation']
    cols = ['target']
    #cols = ['target']
    stats = ['mean']

    # target null in test set to null for other columns care must be taken
    g = df.groupby(['date', 'cluster'])
    for s in stats:
        col_mapper = {c:f"{s}_{c}_cluster" for c in cols}
        tr = g[cols].transform(s).rename(col_mapper, axis=1)
        df = pd.concat([df, tr], axis=1)

    g = df.groupby(['date', 'num'])
    for s in stats:
        col_mapper = {c:f"{s}_{c}_num" for c in cols}
        tr = g[cols].transform(s).rename(col_mapper, axis=1)
        df = pd.concat([df, tr], axis=1)

    g = df.groupby(['date', 'mgrp'])
    for s in stats:
        col_mapper = {c:f"{s}_{c}_mgrp" for c in cols}
        tr = g[cols].transform(s).rename(col_mapper, axis=1)
        df = pd.concat([df, tr], axis=1)

    g = df.groupby(['date'])
    for s in stats:
        col_mapper = {c:f"{s}_{c}" for c in cols}
        tr = g[cols].transform(s).rename(col_mapper, axis=1)
        df = pd.concat([df, tr], axis=1)

    #df['THI_CAT'] = pd.cut(df.THI, [0, 68, 75, 80, 1000], right=False, labels=['THI_1', 'THI_2', 'THI_3', 'THI_4'])

    return df
