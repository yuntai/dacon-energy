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

def load_data(dataroot="./data", nums=[]):
    train_df, test_df = common.prep_tst(dataroot)

    col_ix = train_df.columns.get_loc("datetime")
    train_df['time_idx'] = (train_df.loc[:, 'datetime'] - train_df.iloc[0, col_ix]).astype('timedelta64[h]').astype('int')
    data = train_df

    max_prediction_length = 24*7*2  # forecast 2 weeks
    max_encoder_length = 24*7*8 # use 2 months of history
    training_cutoff = data["time_idx"].max() - max_prediction_length

    data["log_target"] = np.log(data.target + 1e-8)

    #cat_cols = ['num', 'weekday', 'weekend', 'hour', 'THI_CAT', "mgrp", "special_days", "holiday"]
    cat_cols = ['num', "mgrp", 'holiday']
    for col in cat_cols:
        data[col] = data[col].astype(str).astype('category')

    training = TimeSeriesDataSet(
        data[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="target",
        group_ids=["num", "mgrp"],
        min_encoder_length=1,
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length//2,
        max_prediction_length=max_prediction_length,
        time_varying_known_categoricals=cat_cols,
        static_categoricals=["num", "mgrp"],
        # group of categorical variables can be treated as one variable
        #variable_groups={"special_days": special_days},
        time_varying_known_reals=[
            "time_idx",
            "temperature",
            "windspeed",
            "humidity",
            "precipitation",
            "insolation",
            "hour",
            "cumhol"
        ],
        target_normalizer=GroupNormalizer(
            groups=["num", "mgrp"],
            transformation="softplus"
        ),
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=[
            "target",
            "log_target",
            "mean_target_num",
            "mean_target_mgrp",
            "mean_target",
        ],
        add_relative_time_idx=True,  # add as feature
        add_target_scales=True,  # add as feature
        add_encoder_length=True,  # add as feature
    )

    # create validation set (predict=True) which means to predict the
    # last max_prediction_length points in time for each series
    validation = TimeSeriesDataSet.from_dataset(
        training, data, predict=True, stop_randomization=True
    )
    # create dataloaders for model
    batch_size = 128
    train_dataloader = training.to_dataloader(
        train=True, batch_size=batch_size, num_workers=12
    )
    val_dataloader = validation.to_dataloader(
        train=False, batch_size=batch_size * 10, num_workers=12
    )

    return train_dataloader, val_dataloader, training
