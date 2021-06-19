import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from statsmodels.tsa.stattools import pacf
from sklearn.model_selection import train_test_split
import pathlib
import os
import json
import datetime

def add_feats(df):
    df.reset_index(drop=True, inplace=True)

    df['THI'] = 9/5*df['temperature'] - 0.55*(1-df['humidity']/100)*(9/5*df['temperature']-26)+32

    #for num, g in df.groupby('num'):
    #    cdh = (g['temperature']-26).rolling(window=12, min_periods=1).sum()
    #    df.loc[cdh.index, 'CDH'] = cdh

    #cols = ['temperature', 'windspeed', 'humidity', 'precipitation', 'insolation']
    cols = ['target']
    stats = ['mean']

    g = df.groupby(['date', 'num'])
    for c in cols:
        col_mapper = {s:f"{s}_{c}_num" for s in stats}
        val = g[c].agg(stats).reset_index().rename(col_mapper, axis=1)
        df = df.merge(val, on=['date', 'num'], how='left')

    g = df.groupby(['date', 'mgrp'])
    for c in cols:
        col_mapper = {s:f"{s}_{c}_grp" for s in stats}
        val = g[c].agg(stats).reset_index().rename(col_mapper, axis=1)
        df = df.merge(val, on=['date', 'mgrp'], how='left')

    g = df.groupby(['date'])
    for c in cols:
        col_mapper = {s:f"{s}_{c}_date" for s in stats}
        val = g[c].agg(stats).reset_index().rename(col_mapper, axis=1)
        df = df.merge(val, on=['date'], how='left')

    #df['date_num'] = df['month'] + df['day']/31.
    #df['THI_CAT'] = pd.cut(df.THI, [0, 68, 75, 80, 1000], right=False, labels=['THI_1', 'THI_2', 'THI_3', 'THI_4'])

    return df

def now():
    return datetime.datetime.now().strftime("%Y%m%d%H%M%S")

def ensemble(models, X_test):
    preds = []
    for i in range(len(models)):
        pred = np.expm1(models[i].predict(X_test))
        preds.append(pred)

    preds = np.vstack(preds).mean(axis=0)
    return preds

def get_lags_from_model(m):
    with open(m, 'r') as inf:
        o = json.load(inf)
    fns = o['learner']['feature_names']
    return [int(n.split('_')[-1]) for n in fns if 'lag' in n]

def get_scaler(y):
    return MinMaxScaler().fit(y)

def get_model_param(fn):
    fn = os.path.basename(fn)
    num, seed, nweek = fn.split('.')[0].split('_')[1:]
    return num, seed, nweek

def prep_common(train_df, test_df, lags):
    combined_df = pd.concat([train_df, test_df])

    lag_cols = []
    if len(lags) > 0:
        lag_df = create_lag_features(combined_df, combined_df.target, lags)
        lag_cols = list(lag_df.columns)

        combined_df = combined_df.join(lag_df, how='outer')

    #dummies = ['hour','weekday','weekend']
    #for col in dummies:
    #    dummy_df = pd.get_dummies(combined_df[col], prefix=col)
    #    combined_df = combined_df.merge(dummy_df, left_index=True, right_index=True).drop(col, axis=1)

    train_df = combined_df.loc[:train_df.iloc[-1].name].dropna().copy()
    test_df = combined_df.loc[test_df.iloc[0].name:].copy()

    cols = lag_cols + ['target']
    #train_df.loc[:, cols] = train_df[cols].apply(lambda x: target_scaler.transform(np.log1p(x.values[:,None])).squeeze())
    #test_df.loc[:, cols] = test_df[cols].apply(lambda x: target_scaler.transform(np.log1p(x.values[:,None])).squeeze())
    train_df.loc[:, cols] = train_df[cols].apply(lambda x: np.log1p(x.values))
    test_df.loc[:, cols] = test_df[cols].apply(lambda x: np.log1p(x.values))

    assert train_df.isna().sum().sum()==0

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

def prep_submission(dataroot, num, lags):
    train_df, test_df = read_df(dataroot)
    train_df = filter_df(train_df, num)
    test_df = filter_df(test_df, num)

    target_scaler = get_scaler(train_df.target.values[:, None])

    _, test_df, target_scaler = prep_common(train_df, test_df, lags, target_scaler)

    X_test = test_df.drop('target', axis=1)

    return X_test, target_scaler

def prep_tst(dataroot, nums=[]):
    train_df, test_df = read_df(dataroot, nums)

    interpolate(test_df)

    s = train_df[train_df.datetime=='2020-06-01 00:00:00'].groupby(['temperature', 'windspeed']).ngroup()
    s.name = 'mgrp'
    mgrps = train_df[['num']].join(s, how='inner')

    train_df = train_df.merge(mgrps, on='num', how='left')
    test_df = test_df.merge(mgrps, on='num', how='left')

    sz = train_df.shape[0]

    combined_df = pd.concat([train_df, test_df])

    combined_df = add_feats(combined_df)

    train_df = combined_df.iloc[:sz].copy()
    test_df = combined_df.iloc[sz:].copy()

    return train_df, test_df

def prep_full(dataroot, num):
    train_df, test_df = read_df(dataroot, num)

    interpolate(test_df)
    test_idx = test_df.index

    sz = train_df.shape[0]

    y_train = train_df.pop('target')

    combined_df = pd.concat([train_df, test_df])

    combined_df.reset_index(inplace=True)
    combined_df = add_feats(combined_df)

    dummies = ['hour', 'weekday', 'weekend', 'THI_CAT']
    for col in dummies:
        combined_df = combined_df.join(pd.get_dummies(combined_df[col], prefix=col)).drop(col, axis=1)

    combined_df.drop(['num', 'date','solar_flag','nelec_cool_flag', 'datetime'], axis=1, inplace=True)

    train_df = combined_df.iloc[:sz].copy()
    test_df = combined_df.iloc[sz:].copy()

    numerical_features = ['temperature', 'windspeed', 'humidity', 'precipitation', 'insolation', 'min_temperature', 'max_temperature', 'THI', 'mean_THI', 'CDH', 'mean_CDH', 'date_num']

    for feat in numerical_features:
        scaler = StandardScaler()
        train_df[feat] = scaler.fit_transform(train_df[[feat]])
        test_df[feat] = scaler.transform(test_df[[feat]])

    return train_df, y_train, test_df, test_idx

# lag features, scale -> log1p tr
def prep(dataroot, num, max_lags, test_size=0.3, threshold=0.2):
    train_df, test_df = read_df(dataroot, num)

    if test_size != -1:
        if test_size == '2W':
            ix = 24*7*2
            ix = train_df.iloc[-ix].name
        elif type(test_size) == float:
            assert type(test_size) == float
            ix = int(train_df.shape[0] * 0.3)
            ix = train_df.iloc[-ix].name

        test_df = train_df.loc[ix + pd.Timedelta('1H'):]
        train_df = train_df.loc[:ix]
    sz = train_df.shape[0]

    combined_df = pd.concat([train_df, test_df])

    combined_df.reset_index(inplace=True)
    #combined_df = add_feats(combined_df)

    dummies = ['hour', 'weekday', 'weekend']
    for col in dummies:
        combined_df = combined_df.join(pd.get_dummies(combined_df[col], prefix=col)).drop(col, axis=1)

    combined_df.drop(['num', 'month', 'day', 'date','solar_flag','nelec_cool_flag', 'datetime'], axis=1, inplace=True)

    train_df = combined_df.iloc[:sz].copy()
    test_df = combined_df.iloc[sz:].copy()

    numerical_features = ['temperature', 'windspeed', 'humidity', 'precipitation', 'insolation']

    for feat in numerical_features:
        scaler = StandardScaler()
        train_df[feat] = scaler.fit_transform(train_df[[feat]])
        test_df[feat] = scaler.transform(test_df[[feat]])


    #assert sz == test_df.shape[0] + train_df.shape[0]

    #lags = [1, 2, 6, 12, 24, 7*24]
    lags = get_lags(train_df.target, max_lags, threshold)

    train_df, test_df = prep_common(train_df, test_df, lags)

    y_train, X_train = train_df.target, train_df.drop('target', axis=1)
    y_test, X_test = test_df.target, test_df.drop('target', axis=1)

    return X_train, X_test, y_train, y_test, lags

# MAPE computation
def mape(y, yhat, perc=True):
    n = len(yhat.index) if type(yhat) == pd.Series else len(yhat)    
    mape = []
    for a, f in zip(y, yhat):
        # avoid division by 0
        if f > 1e-9:
            mape.append(np.abs((a - f)/a))
    mape = np.mean(np.array(mape))
    return mape * 100. if perc else mape

def SMAPE(true, pred):
    return np.mean((np.abs(true-pred))/(np.abs(true) + np.abs(pred)))*100

# SMAPE computation
def smape(A, F):
    return 100./len(A) * np.sum(2*np.abs(F-A)/(np.abs(A)+np.abs(F)))

def get_lags(y, max_lags, threshold):
    partial = pd.Series(data=pacf(y, nlags=max_lags))
    lags = list(partial[np.abs(partial) >= threshold].index)
    # avoid to insert the time series itself
    lags.remove(0)
    assert 1 in lags
    return lags

def create_lag_features(train_df, y, lags, cols=[]):
    df = pd.DataFrame()
    for l in lags:
        df[f"lag_{l:02d}"] = y.shift(l)
        for c in cols:
            df[f"{c}_lag_{l:02d}"] = train_df[c].shift(l)
    df.index = y.index
    return df

def create_lag_features_with_scale(y, nlags, threshold):

    scaler = StandardScaler()
    features = pd.DataFrame()

    partial = pd.Series(data=pacf(y, nlags=nlags))
    lags = list(partial[np.abs(partial) >= threshold].index)

    df = pd.DataFrame()

    # avoid to insert the time series itself
    lags.remove(0)

    for l in lags:
        df[f"lag_{l}"] = y.shift(l)

    features = pd.DataFrame(scaler.fit_transform(df[df.columns]),
                            columns=df.columns)
    features.index = y.index

    return features

#전력사용량(kWh) 기온(°C) 풍속(m/s) 습도(%) 강수량(mm) 일조(hr) 비전기냉방설비운영 태양광보유
train_columns = ['num','datetime','target','temperature','windspeed','humidity','precipitation','insolation','nelec_cool_flag','solar_flag']
test_columns = ['num','datetime','temperature','windspeed','humidity','precipitation','insolation','nelec_cool_flag','solar_flag']


def read_df(dataroot, nums=[]):
    dataroot = pathlib.Path(dataroot) if type(dataroot) == str else dataroot

    def date_prep(df):
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['hour'] = df['datetime'].dt.hour
        df['weekday'] = df['datetime'].dt.weekday
        df['date'] = df['datetime'].dt.date.astype('str')
        df['day'] = df['datetime'].dt.day
        df['month'] = df['datetime'].dt.month
        df['weekend'] = df['weekday'].isin([5,6]).astype(int)

        special_days = ['2020-06-06', '2020-08-15', '2020-08-17']
        special_days = [datetime.date(*map(int,s.split("-"))) for s in special_days]

        df['special_days'] = '-'
        df.loc[df.date.isin(special_days), 'special_days'] = '1'

        #df['holiday'] = ((df.weekend==1) | (df.special_days == '1')).astype(int)
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

class TargetTransformer:
    """
    Perform some transformation on the time series
    data in order to make the model more performant and
    avoid non-stationary effects.
    """

    def __init__(self, log=False, detrend=False, diff=False):

        self.trf_log = log
        self.trf_detrend = detrend
        self.trend = pd.Series(dtype=np.float64)

    def transform(self, index, values):
        """
        Perform log transformation to the target time series

        :param index: the index for the resulting series
        :param values: the values of the initial series

        Return:
            transformed pd.Series
        """
        res = pd.Series(index=index, data=values)

        if self.trf_detrend:
            self.trend = TargetTransformer.get_trend(res) - np.mean(res.values)
            res = res.subtract(self.trend)

        if self.trf_log:
            res = pd.Series(index=index, data=np.log1p(res.values))

        return res

    def inverse(self, index, values):
        """
        Go back to the original time series values

        :param index: the index for the resulting series
        :param values: the values of series to be transformed back

        Return:
            inverse transformed pd.Series
        """
        res = pd.Series(index=index, data=values)

        if self.trf_log:
            res = pd.Series(index=index, data=np.exp(values))
        try:
            if self.trf_detrend:
                assert len(res.index) == len(self.trend.index)
                res = res + self.trend

        except AssertionError:
            print("Use a different transformer for each target to transform")

        return res

    @staticmethod
    def get_trend(data):
        """
        Get the linear trend on the data which makes the time
        series not stationary
        """
        n = len(data.index)
        X = np.reshape(np.arange(0, n), (n, 1))
        y = np.array(data)
        model = LinearRegression()
        model.fit(X, y)
        trend = model.predict(X)
        return pd.Series(index=data.index, data=trend)

def forecast_multi_recursive(y, model, lags, n_steps, step="1H"):
    
    """Multi-step recursive forecasting using the input time 
    series data and a pre-trained machine learning model
    
    Parameters
    ----------
    y: pd.Series holding the input time-series to forecast
    model: an already trained machine learning model implementing the scikit-learn interface
    lags: list of lags used for training the model
    n_steps: number of time periods in the forecasting horizon
    step: forecasting time period given as Pandas time series frequencies
    
    Returns
    -------
    fcast_values: pd.Series with forecasted values indexed by forecast horizon dates 
    """
    
    # get the dates to forecast
    last_date = y.index[-1] + pd.Timedelta(hours=1)
    fcast_range = pd.date_range(last_date, periods=n_steps, freq=step)

    fcasted_values = []
    target = y.copy()

    for date in fcast_range:

        new_point = fcasted_values[-1] if len(fcasted_values) > 0 else 0.0   
        target = target.append(pd.Series(index=[date], data=new_point))

        # forecast
        ts_features = create_ts_features(target)
        if len(lags) > 0:
            lags_features = create_lag_features(target, lags=lags)
            features = ts_features.join(lags_features, how="outer").dropna()
        else:
            features = ts_features

        predictions = model.predict(features)
        fcasted_values.append(predictions[-1])

    return pd.Series(index=fcast_range, data=fcasted_values)

#rec_fcast = forecast_multi_recursive(y, xgb_model, feature_lags)
