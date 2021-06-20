import sys
import argparse

import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl

import common

from pytorch_forecasting.data import (
    TimeSeriesDataSet,
    GroupNormalizer
)
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor
)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_forecasting.metrics import QuantileLoss, SMAPE, RMSE, MAE, CompositeMetric
from pytorch_forecasting.models import TemporalFusionTransformer, Baseline
from pytorch_forecasting.metrics import SMAPE

parser = argparse.ArgumentParser();
parser.add_argument('--num', '-n', type=int, default=-1)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--logdir', type=str, default="./logs")
args = parser.parse_args()
print(args)

from tst_dataset import load_dataset

tr_ds, va_ds = load_dataset("./data")

# create dataloaders for model
batch_size = 128
tr_loader = tr_ds.to_dataloader(
    train=True, batch_size=batch_size, num_workers=12
)
va_loader = va_ds.to_dataloader(
    train=False, batch_size=batch_size * 10, num_workers=12
)

# TRAINING
PARAMS = {
    'gradient_clip_val': 0.9658579636307634,
    'hidden_size': 80,
    'dropout': 0.19610151695402608,
    'hidden_continuous_size': 40,
    'attention_head_size': 4,
    'learning_rate': 0.085
}

# stop training, when loss metric does not improve on validation set
early_stop_callback = EarlyStopping(
    monitor="val_loss",
    min_delta=1e-3,
    patience=10,
    verbose=True,
    mode="min"
)

lr_logger = LearningRateMonitor()  # log the learning rate
logger = TensorBoardLogger(args.logdir)  # log to tensorboard
#logger = WandbLogger()

# create trainer
trainer = pl.Trainer(
    max_epochs=60,
    gpus=[0],  # train on CPU, use gpus = [0] to run on GPU
    gradient_clip_val=PARAMS['gradient_clip_val'],
    limit_train_batches=30,  # running validation every 30 batches
    # fast_dev_run=True,  # comment in to quickly check for bugs
    callbacks=[lr_logger, early_stop_callback],
    logger=logger,
)

loss_fn = SMAPE()

# initialise model
tft = TemporalFusionTransformer.from_dataset(
    tr_ds,
    learning_rate=PARAMS['learning_rate'],
    hidden_size=PARAMS['hidden_size'],
    attention_head_size=PARAMS['attention_head_size'],
    dropout=PARAMS['dropout'],
    hidden_continuous_size=PARAMS['hidden_continuous_size'],
    output_size=1,  # QuantileLoss has 7 quantiles by default
    loss=loss_fn,
    log_interval=10,  # log example every 10 batches
    reduce_on_plateau_patience=4,  # reduce learning automatically
)
print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

# fit network
trainer.fit(
    tft,
    train_dataloader=tr_loader,
    val_dataloaders=va_loader
)

# evaluate

# load the best model according to the validation loss (given that
# we use early stopping, this is not necessarily the last epoch)
best_model_path = trainer.checkpoint_callback.best_model_path
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
# calculate mean absolute error on validation set
actuals = torch.cat([y[0] for x, y in iter(va_loader)])
predictions = best_tft.predict(va_loader)
smape_per_num = SMAPE(reduction="none")(predictions, actuals).mean(1)
print(f"{best_model_path=}")
print(smape_per_num)
print(smape_per_num.mean())
