import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import pacf
from sklearn.model_selection import train_test_split
import pathlib
import os

def get_model_fn(args):
    return f"model_{args.num}_{args.seed}_{args.nweek}.json"

def get_model_param(fn):
    fn = os.path.basename(fn)
    num, seed, nweek = fn.split('.')[0].split('_')[1:]
    return num, seed, nweek

def prep_submission(dataroot, nweek, num, test_size=0.3):
    _, test_df = read_df(pathlib.Path(dataroot))
    test_df = test_df[test_df.num==num].set_index('datetime').asfreq('1H', 'bfill')
    test_df = test_df.drop(['date','num','nelec_cool_flag','solar_flag'], axis=1)

    X_train, *_, target_scaler, _ = prep(dataroot, nweek, num)
    return X_train, test_df, target_scaler

# lag features, scale -> log1p tr
def prep(dataroot, nweek, num, test_size=0.3):
    train_df, _ = read_df(pathlib.Path(dataroot))

    df = train_df[train_df.num==num].set_index('datetime').asfreq('1H', 'bfill')
    df = df.drop(['date','num','nelec_cool_flag','solar_flag'], axis=1)

    nlags = 24*7*nweek

    lags = create_lag_features2(df.target, nlags, 0.2) # w/o scale
    num_lag_features = lags.shape[1]
    features = df.join(lags, how="outer").dropna()
    target = features.target
    features = features.drop('target', axis=1)

    X_train_, X_test_, y_train_, y_test_ = train_test_split(features,
                                                        target,
                                                        test_size=test_size,
                                                        shuffle=False)

    lag_cols = lags.columns

    target_scaler = StandardScaler()
    target_scaler.fit(y_train_.values[:,None])

    X_train = X_train_.copy()
    X_test = X_test_.copy()
    X_train.loc[:, lag_cols] = X_train_[lag_cols].apply(lambda x: target_scaler.transform(x.values[:,None]).squeeze())
    X_test.loc[:, lag_cols] = X_test_[lag_cols].apply(lambda x: target_scaler.transform(x.values[:,None]).squeeze())

    y_train = pd.Series(target_scaler.transform(y_train_.values[:,None]).squeeze())
    y_test = pd.Series(target_scaler.transform(y_test_.values[:,None]).squeeze())

    y_train.index = y_train_.index
    y_test.index = y_test_.index

    return X_train, X_test, y_train, y_test, target_scaler, num_lag_features

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

# SMAPE computation
def smape(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

def create_lag_features2(y, nlags, threshold):

    features = pd.DataFrame()

    partial = pd.Series(data=pacf(y, nlags=nlags))
    lags = list(partial[np.abs(partial) >= threshold].index)

    df = pd.DataFrame()

    # avoid to insert the time series itself
    lags.remove(0)

    for l in lags:
        df[f"lag_{l}"] = y.shift(l)

    df.index = y.index

    return df

def create_lag_features(y, nlags, threshold):

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


def read_df(dataroot):
    train_df = pd.read_csv(dataroot/'train.csv', skiprows=[0], names=train_columns)
    
    def date_prep(df):
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['hour'] = df['datetime'].dt.hour
        df['weekday'] = df['datetime'].dt.weekday
        df['date'] = df['datetime'].dt.date
        df['day'] = df['datetime'].dt.day
        df['month'] = df['datetime'].dt.month
        df['weekend'] = df['weekday'].isin([5,6]).astype(int)

    test_df = pd.read_csv(dataroot/'test.csv', skiprows=[0], names=test_columns)

    date_prep(train_df)
    date_prep(test_df)

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
