# autor: Yuntai KYONG (yuntai.kyong@gmail.com)

#GPU memory usage:
#23546MiB / 24263MiB

#requirements:
#pytorch-forecasting==0.9.0

import sys
import os
import argparse
import shutil
import random
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl

from pytorch_forecasting.data import (
    TimeSeriesDataSet,
    GroupNormalizer
)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor
)
from pytorch_forecasting.metrics import SMAPE
from pytorch_forecasting.models import TemporalFusionTransformer


# category columns
CATE_COLS = ['num', "mgrp", 'holiday', 'dow', 'cluster', 'hot', 'nelec_cool_flag', 'solar_flag']

# building cluster based on kmeans
CLUSTER = {
    0: [19, 20, 21, 49, 50, 51],
    1: [1, 5, 9, 34],
    2: [4, 10, 11, 12, 28, 29, 30, 36, 40, 41, 42, 59, 60],
    3: [2, 3, 6, 7, 8, 13, 14, 15, 16, 17, 18, 22, 23, 24, 25, 26, 27, 31, 32, 33, 35, 37, 38, 39, 43, 44, 45, 46, 47, 48, 52, 53, 54, 55, 56, 57, 58],
}

# length of training data for prediction (5 weeks)
ENCODER_LENGTH_IN_WEEKS = 5

# learning rate determined by a cv run with train data less 1 trailing week as validation 
LRS = [0.05099279397234306, 0.05099279397234306, 0.05099279397234306, 0.05099279397234306,
       0.05099279397234306, 0.05099279397234306, 0.05099279397234306, 0.05099279397234306,
       0.05099279397234306, 0.05099279397234306, 0.05099279397234306, 0.05099279397234306,
       0.05099279397234306, 0.05099279397234306, 0.05099279397234306, 0.05099279397234306,
       0.05099279397234306, 0.05099279397234306, 0.05099279397234306, 0.05099279397234306,
       0.05099279397234306, 0.05099279397234306, 0.05099279397234306, 0.05099279397234306,
       0.05099279397234306, 0.05099279397234306, 0.05099279397234306, 0.05099279397234306,
       0.05099279397234306, 0.05099279397234306, 0.05099279397234306, 0.05099279397234306,
       0.05099279397234306, 0.05099279397234306, 0.05099279397234306, 0.05099279397234306,
       0.05099279397234306, 0.05099279397234306, 0.05099279397234306, 0.05099279397234306,
       0.05099279397234306, 0.05099279397234306, 0.05099279397234306, 0.05099279397234306,
       0.05099279397234306, 0.05099279397234306, 0.05099279397234306, 0.05099279397234306,
       0.05099279397234306 , 0.05099279397234306, 0.05099279397234306, 0.05099279397234306,
       0.005099279397234306, 0.005099279397234306, 0.005099279397234306, 0.005099279397234306,
       0.005099279397234306, 0.005099279397234306, 0.005099279397234306, 0.005099279397234306,
       0.005099279397234306, 0.0005099279397234307, 0.0005099279397234307, 0.0005099279397234307,
       0.0005099279397234307, 0.0005099279397234307, 0.0005099279397234307]

# number of epochs found in cv run
NUM_EPOCHS = 66

# number of seeds to use
NUM_SEEDS = 10

BATCH_SIZE = 128

# hyper parameters determined by cv runs with train data less 1 trailing week as validation 
PARAMS = {
    'gradient_clip_val': 0.9658579636307634,
    'hidden_size': 180,
    'dropout': 0.19610151695402608,
    'hidden_continuous_size': 90,
    'attention_head_size': 4,
    'learning_rate': 0.08
}

parser = argparse.ArgumentParser()
parser.add_argument('--seed','-s', nargs='+', type=int, default=list(range(42, 42+NUM_SEEDS)))
#parser.add_argument('--val', default=False, action='store_true')
#parser.add_argument('--nepochs', '-e', type=int, default=NUM_EPOCHS)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--fit', default=False, action='store_true')
parser.add_argument('--forecast', default=False, action='store_true')
parser.add_argument('--dataroot', '-d', type=str, default="/data")
args = parser.parse_args()
args.val = False
args.nepochs = NUM_EPOCHS
print(args)

DATAROOT = Path(args.dataroot) # 코드에 ‘/data’ 데이터 입/출력 경로 포함
CKPTROOT = DATAROOT/"ckpts" # directory for model checkpoints
CSVROOT = DATAROOT/"csvs" # directory for prediction outputs
SUBFN = DATAROOT/"sub.csv" # final submission file path
LOGDIR = DATAROOT/"logs" # pytorch_forecasting requirs logger

def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# prepare data features
def __date_prep(df):

    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['datetime'].dt.hour
    df['dow'] = df['datetime'].dt.weekday
    df['date'] = df['datetime'].dt.date.astype('str')
    df['day'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month

    # FEATURE: saturday, sunday and speical holidays flagged as `holiday` flag
    special_days = ['2020-06-06', '2020-08-15', '2020-08-17']
    df['holiday'] = df['dow'].isin([5,6]).astype(int)
    df.loc[df.date.isin(special_days), 'holiday'] = 1

    # FEATURE: `hot` flag when the next day is holiday
    hot = df.groupby('date').first()['holiday'].shift(-1).fillna(0).astype(int)
    hot = hot.to_frame().reset_index().rename({'holiday': "hot"}, axis=1)
    df = df.merge(hot, on='date', how='left')

    # FEATURE: `cumhol` - how many days left in 연휴
    h = (df.groupby('date').first()['holiday'] != 0).iloc[::-1]
    df1 = h.cumsum() - h.cumsum().where(~h).ffill().fillna(0).astype(int).iloc[::-1]
    df1 = df1.to_frame().reset_index().rename({'holiday': "cumhol"}, axis=1)
    df = df.merge(df1, on='date', how='left')

    return df

# read data, process date and assign cluster number
def __read_df():
    train_columns = ['num','datetime','target','temperature','windspeed','humidity','precipitation','insolation','nelec_cool_flag','solar_flag']
    test_columns = [c for c in train_columns if c != 'target']

    train_df = pd.read_csv(DATAROOT/'train.csv', skiprows=[0], names=train_columns)
    test_df = pd.read_csv(DATAROOT/'test.csv', skiprows=[0], names=test_columns)

    __sz = train_df.shape[0]

    df = pd.concat([train_df, test_df])

    # assing cluster number to building
    for k, nums in CLUSTER.items():
        df.loc[df.num.isin(nums), 'cluster'] = k

    df = __date_prep(df)

    return df.iloc[:__sz].copy(), df.iloc[__sz:].copy()

# add aggregate(mean) target feature for 'cluster', 'building', 'mgrp' per date
def add_feats(df):
    df.reset_index(drop=True, inplace=True)

    cols = ['target']
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

    return df

# interpolate NA values in test dataset
def interpolate_(test_df):
    # https://dacon.io/competitions/official/235736/codeshare/2844?page=1&dtype=recent
    # 에서 제안된 방법으로
    __methods = {
        'temperature': 'quadratic',
        'windspeed':'linear',
        'humidity':'quadratic',
        'precipitation':'linear',
        'insolation': 'pad'
    }

    for col, method in __methods.items():
        test_df[col] = test_df[col].interpolate(method=method)
        if method == 'quadratic':
            test_df[col] = test_df[col].interpolate(method='linear')

# prepare train and test data
def prep():

    train_df, test_df = __read_df()

    # get nelec_cool_flag and solar_flag from training data
    test_df = test_df.drop(['nelec_cool_flag','solar_flag'], axis=1)
    test_df = test_df.merge(train_df.groupby("num").first()[['nelec_cool_flag','solar_flag']].reset_index(), on="num", how="left")

    # interpolate na in test_df for temperature, windspeed, humidity, precipitation & insolation
    interpolate_(test_df)

    # FEATURE(mgrp): group buildings having same temperature and windspeed measurements
    s = train_df[train_df.datetime=='2020-06-01 00:00:00'].groupby(['temperature', 'windspeed']).ngroup()
    s.name = 'mgrp'
    mgrps = train_df[['num']].join(s, how='inner')

    sz = train_df.shape[0]

    df = pd.concat([train_df, test_df])
    df = df.merge(mgrps, on='num', how='left')

    # add aggregate target features
    df = add_feats(df)

    # add log target
    df["log_target"] = np.log(df.target + 1e-8)

    for col in CATE_COLS:
        df[col] = df[col].astype(str).astype('category')

    # add time index feature
    __ix = df.columns.get_loc('datetime')
    df['time_idx'] = (df.loc[:, 'datetime'] - df.iloc[0, __ix]).astype('timedelta64[h]').astype('int')

    train_df = df.iloc[:sz].copy()
    test_df = df.iloc[sz:].copy()

    return train_df, test_df

# build traind datset
def load_dataset(train_df, validate=False):

    max_encoder_length = 24*7*ENCODER_LENGTH_IN_WEEKS # use 5 past weeks
    max_prediction_length = 24*7 # to predict 1 week of future
    training_cutoff = train_df["time_idx"].max() - max_prediction_length

    # build training dataset
    tr_ds = TimeSeriesDataSet(
        # with validate=False use all data
        train_df[lambda x: x.time_idx <= training_cutoff] if validate else train_df,
        time_idx="time_idx",
        target="target",
        group_ids=["num"],
        min_encoder_length=1,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        time_varying_known_categoricals=CATE_COLS,
        static_categoricals=["num", "mgrp", "cluster"],
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

    va_ds = None
    if validate:
        # validation dataset not used for submission
        va_ds = TimeSeriesDataSet.from_dataset(
            tr_ds, train_df, predict=True, stop_randomization=True
        )

    return tr_ds, va_ds

# training
def fit(seed, tr_ds, va_loader=None):
    seed_all(seed) # doesn't really work as training is non-deterministic

    # create dataloaders for model
    tr_loader = tr_ds.to_dataloader(
        train=True, batch_size=BATCH_SIZE, num_workers=12
    )

    if va_loader is not None:
        # stop training, when loss metric does not improve on validation set
        early_stopping_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=1e-4,
            patience=20,
            verbose=True,
            mode="min"
        )
        lr_logger = LearningRateMonitor(logging_interval="epoch")  # log the learning rate
        callbacks = [lr_logger, early_stopping_callback]
    else:
        # gather 10 checkpoints with best traing loss
        checkpoint_callback = ModelCheckpoint(
            monitor='train_loss',
            dirpath=CKPTROOT,
            filename=f'seed={seed}'+'-{epoch:03d}-{train_loss:.2f}',
            save_top_k=10
        )
        callbacks = [checkpoint_callback]

    # create trainer
    trainer = pl.Trainer(
        max_epochs=args.nepochs,
        gpus=[args.gpu],
        gradient_clip_val=PARAMS['gradient_clip_val'],
        limit_train_batches=30,
        callbacks=callbacks,
        logger=TensorBoardLogger(LOGDIR)
    )

    # use pre-deterined leraning rate schedule for final submission
    learning_rate = LRS if va_loader is None else PARAMS['learning_rate']

    # initialise model with pre-determined hyperparameters
    tft = TemporalFusionTransformer.from_dataset(
        tr_ds,
        learning_rate=learning_rate,
        hidden_size=PARAMS['hidden_size'],
        attention_head_size=PARAMS['attention_head_size'],
        dropout=PARAMS['dropout'],
        hidden_continuous_size=PARAMS['hidden_continuous_size'],
        output_size=1,
        loss=SMAPE(), # SMAPE loss
        log_interval=10,  # log example every 10 batches
        logging_metrics=[SMAPE()],
        reduce_on_plateau_patience=4,  # reduce learning automatically
    )
    print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

    kwargs = {'train_dataloader': tr_loader}
    if va_loader:
        kwargs['val_dataloaders'] = va_loader

    # fit network
    trainer.fit(
        tft,
        **kwargs
    )

    best_model_path = trainer.checkpoint_callback.best_model_path
    print(f"{best_model_path=}")
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

    return best_tft

# predict 1 week
def forecast(ckpt, train_df, test_df):
    # load model
    best_tft = TemporalFusionTransformer.load_from_checkpoint(ckpt)
    max_encoder_length = best_tft.dataset_parameters['max_encoder_length']
    max_prediction_length = best_tft.dataset_parameters['max_prediction_length']

    assert max_encoder_length == 5*24*7 and max_prediction_length == 1*24*7

    # use 5 weeks of training data at the end
    encoder_data = train_df[lambda x: x.time_idx > x.time_idx.max() - max_encoder_length]

    # get last entry from training data
    last_data = train_df.iloc[[-1]]

    # fill NA target value in test data with last values from the train dataset
    target_cols = [c for c in test_df.columns if 'target' in c]
    for c in target_cols:
        test_df.loc[:, c] = last_data[c].item()

    decoder_data = test_df

    # combine encoder and decoder data. decoder data is to be predicted
    new_prediction_data = pd.concat([encoder_data, decoder_data], ignore_index=True)
    new_raw_predictions, new_x = best_tft.predict(new_prediction_data, mode="raw", return_x=True)

    # num_labels: mapping from 'num' categorical feature to index in new_raw_predictions['prediction']
    #             {'5': 4, '6': 6, ...}
    # new_raw_predictions['prediction'].shape = (60, 168, 1)
    num_labels = best_tft.dataset_parameters['categorical_encoders']['num'].classes_

    preds = new_raw_predictions['prediction'].squeeze()

    sub_df = pd.read_csv(DATAROOT/"sample_submission.csv")

    # get prediction for each building (num)
    for n, ix in num_labels.items():
        sub_df.loc[sub_df.num_date_time.str.startswith(f"{n} "), 'answer'] = preds[ix].numpy()

    # save predction to a csv file
    outfn = CSVROOT/(Path(ckpt).stem + '.csv')
    print(outfn)
    sub_df.to_csv(outfn, index=False)

def ensemble(outfn):
    # get all prediction csv files
    fns = list(CSVROOT.glob("*.csv"))
    df0 = pd.read_csv(fns[0])
    df = pd.concat([df0] + [pd.read_csv(fn).loc[:,'answer'] for fn in fns[1:]], axis=1)
    # get median of all predcitions
    df['median'] = df.iloc[:,1:].median(axis=1)
    df = df[['num_date_time', 'median']]
    df = df.rename({'median': 'answer'}, axis=1)
    # save to submission file
    df.to_csv(outfn, index=False)

# not used for final submission
def validate(seed, tr_ds, va_ds):
    va_loader = va_ds.to_dataloader(
        train=False, batch_size=BATCH_SIZE*10, num_workers=12
    )
    best_tft = fit(seed, tr_ds, va_loader)
    actuals = torch.cat([y[0] for x, y in iter(va_loader)])
    predictions = best_tft.predict(va_loader)
    smape_per_num = SMAPE(reduction="none")(predictions, actuals).mean(1)
    print(smape_per_num)
    print(smape_per_num.mean())

if __name__ == "__main__":
    [p.mkdir(exist_ok=True) for p in (CKPTROOT, CSVROOT, LOGDIR)]

    train_df, test_df = prep()
    tr_ds, va_ds = load_dataset(train_df, args.val)

    if args.val:
        validate(args.seed[0], tr_ds, va_ds)
    else:
        if args.fit:
            print("### FIT ###")
            for s in args.seed:
                fit(s, tr_ds)

        if args.forecast:
            print("### FORECAST ###")
            for p in CKPTROOT.glob("*.ckpt"):
                forecast(p, train_df, test_df)

            print("### ENSEMBLING ###")
            ensemble(SUBFN)
