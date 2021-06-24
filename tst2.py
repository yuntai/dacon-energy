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
    LearningRateMonitor,
    ModelCheckpoint
)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_forecasting.metrics import QuantileLoss, SMAPE, RMSE, MAE, CompositeMetric
from pytorch_forecasting.models import TemporalFusionTransformer, Baseline

parser = argparse.ArgumentParser();
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--logdir', type=str, default="./logs")
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--encoder_length', '-e', type=int, default=5)
parser.add_argument('--num_epochs', type=int, default=66)
args = parser.parse_args()
print(args)

from tst_dataset import load_dataset

tr_ds, _ = load_dataset(
    "./data",
    encoder_length_in_weeks=args.encoder_length,
    no_va=True
)

# create dataloaders for model
batch_size = 128
tr_loader = tr_ds.to_dataloader(
    train=True, batch_size=batch_size, num_workers=12
)

#PARAMS = {
#    'gradient_clip_val': 0.26051895622603816,
#    'hidden_size': 42,
#    'dropout': 0.26385904502541296,
#    'hidden_continuous_size': 41,
#    'attention_head_size': 4,
#    'learning_rate': 0.05099279397234306
#}

PARAMS = {
    'gradient_clip_val': 0.9658579636307634,
    'hidden_size': 180,
    'dropout': 0.19610151695402608,
    'hidden_continuous_size': 90,
    'attention_head_size': 4,
    'learning_rate': 0.08
}

lr_logger = LearningRateMonitor()  # log the learning rate
logger = TensorBoardLogger(args.logdir)  # log to tensorboard
#logger = WandbLogger()

checkpoint_callback = ModelCheckpoint(
    monitor='train_loss',
    dirpath='./models',
    filename='{epoch:03d}-{train_loss:.2f}',
    save_top_k=10
)
lrs = [0.05099279397234306, 0.05099279397234306, 0.05099279397234306, 0.05099279397234306,
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

# create trainer
trainer = pl.Trainer(
    max_epochs=args.num_epochs,
    gpus=[args.gpu],
    #gpus=[0,1],  # train on CPU, use gpus = [0] to run on GPU
    #accelerator='dp',
    gradient_clip_val=PARAMS['gradient_clip_val'],
    limit_train_batches=30,  # running validation every 30 batches
    # fast_dev_run=True,  # comment in to quickly check for bugs
    callbacks=[checkpoint_callback],
    logger=logger,
)

loss_fn = SMAPE()

# initialise model
tft = TemporalFusionTransformer.from_dataset(
    tr_ds,
    #learning_rate=PARAMS['learning_rate'],
    learning_rate=lrs,
    hidden_size=PARAMS['hidden_size'],
    attention_head_size=PARAMS['attention_head_size'],
    dropout=PARAMS['dropout'],
    hidden_continuous_size=PARAMS['hidden_continuous_size'],
    output_size=1,  # QuantileLoss has 7 quantiles by default
    loss=loss_fn,
    log_interval=10,  # log example every 10 batches
    #reduce_on_plateau_patience=4,  # reduce learning automatically
    logging_metrics=[SMAPE()]
)
print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

# fit network
trainer.fit(
    tft,
    train_dataloader=tr_loader
)

# load the best model according to the validation loss (given that
# we use early stopping, this is not necessarily the last epoch)
#best_model_path = trainer.checkpoint_callback.best_model_path
#best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
## calculate mean absolute error on validation set
#actuals = torch.cat([y[0] for x, y in iter(va_loader)])
#predictions = best_tft.predict(va_loader)
#smape_per_num = SMAPE(reduction="none")(predictions, actuals).mean(1)
#print(f"{best_model_path=}")
#print(smape_per_num)
#print(smape_per_num.mean())