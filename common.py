import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from statsmodels.tsa.stattools import pacf
from sklearn.model_selection import train_test_split
import pathlib
import os
import json
import datetime

def read_df(dataroot, nums=[]):
    train_columns = ['num','datetime','target','temperature','windspeed','humidity','precipitation','insolation','nelec_cool_flag','solar_flag']
    test_columns = ['num','datetime','temperature','windspeed','humidity','precipitation','insolation','nelec_cool_flag','solar_flag']

    dataroot = pathlib.Path(dataroot) if type(dataroot) == str else dataroot

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

        h = df.groupby('date').first()['holiday'] != 0
        h = h.shift(-1).fillna(False).iloc[::-1]
        df1 = h.cumsum()-h.cumsum().where(~h).ffill().fillna(0).astype(int).iloc[::-1]
        df1 = df1.to_frame().reset_index().rename({'holiday': "cumhol"}, axis=1)
        df = df.merge(df1, on='date', how='left')

        #if num != -1:
        #    df = df.set_index('datetime').asfreq('1H', 'bfill')
        return df


    train_df = pd.read_csv(dataroot/'train.csv', skiprows=[0], names=train_columns)
    test_df = pd.read_csv(dataroot/'test.csv', skiprows=[0], names=test_columns)

    if len(nums) > 0:
        train_df = train_df[train_df.num.isin(nums)]
        test_df = test_df[test_df.num.isin(nums)]

    train_df = date_prep(train_df)
    test_df = date_prep(test_df)

    return train_df, test_df

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

def prep_tst(dataroot, nums=[]):
    train_df, test_df = read_df(dataroot, nums)

    interpolate(test_df)

    s = train_df[train_df.datetime=='2020-06-01 00:00:00'].groupby(['temperature', 'windspeed']).ngroup()
    s.name = 'mgrp'
    mgrps = train_df[['num']].join(s, how='inner')

    sz = train_df.shape[0]

    combined_df = pd.concat([train_df, test_df])
    combined_df = combined_df.merge(mgrps, on='num', how='left')

    combined_df = add_feats(combined_df)

    train_df = combined_df.iloc[:sz].copy()
    test_df = combined_df.iloc[sz:].copy()

    return train_df, test_df

def add_feats(df):
    df.reset_index(drop=True, inplace=True)

    #df['THI'] = 9/5*df['temperature'] - 0.55*(1-df['humidity']/100)*(9/5*df['temperature']-26) + 32

    #for num, g in df.groupby('num'):
    #    cdh = (g['temperature']-26).rolling(window=12, min_periods=1).sum()
    #    df.loc[cdh.index, 'CDH'] = cdh

    #cols = ['temperature', 'windspeed', 'humidity', 'precipitation', 'insolation']
    cols = ['target', 'precipitation', 'insolation']
    #cols = ['target']
    stats = ['mean']

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
