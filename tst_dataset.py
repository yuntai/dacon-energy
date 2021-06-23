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

def load_dataset(dataroot="./data", encoder_length_in_weeks=5):
    train_df, test_df = common.prep_tst(dataroot)
    data = train_df

    max_encoder_length = 24*7*encoder_length_in_weeks
    max_prediction_length = 24*7
    training_cutoff = data["time_idx"].max() - max_prediction_length

    tr_ds = TimeSeriesDataSet(
        data[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="target",
        group_ids=["num"],
        min_encoder_length=1,
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length//2,
        max_prediction_length=max_prediction_length,
        time_varying_known_categoricals=common.CATE_COLS,
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
