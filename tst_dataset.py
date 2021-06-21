import pytorch_lightning as pl
import torch
import sys
from pytorch_forecasting.data import (
    TimeSeriesDataSet,
    GroupNormalizer
)
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor
)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_forecasting.metrics import QuantileLoss, SMAPE, RMSE, MAE, CompositeMetric
from pytorch_forecasting.models import TemporalFusionTransformer, Baseline
from pytorch_forecasting.metrics import SMAPE
import argparse
import common
import pandas as pd
import numpy as np

def load_dataset(dataroot="./data", nums=[]):
    train_df, test_df = common.prep_tst(dataroot)

    col_ix = train_df.columns.get_loc("datetime")
    train_df['time_idx'] = (train_df.loc[:, 'datetime'] - train_df.iloc[0, col_ix]).astype('timedelta64[h]').astype('int')
    data = train_df
    #data = train_df[(train_df.time_idx < 1872)].copy()

    max_encoder_length = 24*7*5 # use up to 6 weeks of history
    max_prediction_length = 24*7*2  # forecast 1 week
    training_cutoff = data["time_idx"].max() - max_prediction_length

    data["log_target"] = np.log(data.target + 1e-8)

    cate_cols = ['num', "mgrp", 'holiday', 'dow', 'cluster', 'hot', 'nelec_cool_flag', 'solar_flag']
    for col in cate_cols:
        data[col] = data[col].astype(str).astype('category')

    tr_ds = TimeSeriesDataSet(
        data[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="target",
        group_ids=["num"],
        min_encoder_length=1,
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length//2,
        max_prediction_length=max_prediction_length,
        time_varying_known_categoricals=cate_cols,
        static_categoricals=["num", "mgrp", "cluster"],
        # group of categorical variables can be treated as one variable
        #variable_groups={"special_days": special_days},
        time_varying_known_reals=[
            "time_idx",
            'hour',
            "temperature",
            "windspeed",
            "humidity",
            "precipitation",
            "insolation",
            'cumhol'
        ],
        target_normalizer=GroupNormalizer(groups=["num"], transformation="softplus"),
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=[
            "target",
            "log_target",
            "mean_target",
            "mean_target_num",
            "mean_target_mgrp",
            "mean_target_cluster"
        ],
        add_relative_time_idx=True,  # add as feature
        add_target_scales=True,  # add as feature
        add_encoder_length=True,  # add as feature
    )

    # create validation set (predict=True) which means to predict the
    # last max_prediction_length points in time for each series
    va_ds = TimeSeriesDataSet.from_dataset(
        tr_ds, data, predict=True, stop_randomization=True
    )

    return tr_ds, va_ds
