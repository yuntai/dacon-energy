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

parser = argparse.ArgumentParser();
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

from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
print("start studying...")
# create study
study = optimize_hyperparameters(
    tr_loader,
    va_loader,
    model_path="optuna_test",
    n_trials=200,
    max_epochs=50,
    gradient_clip_val_range=(0.01, 1.0),
    hidden_size_range=(8, 128),
    hidden_continuous_size_range=(8, 128),
    attention_head_size_range=(1, 4),
    learning_rate_range=(0.001, 0.1),
    dropout_range=(0.1, 0.3),
    trainer_kwargs=dict(limit_train_batches=30),
    reduce_on_plateau_patience=4,
    use_learning_rate_finder=False,  # use Optuna to find ideal learning rate or use in-built learning rate finder
    verbose=2
)

# save study results - also we can resume tuning at a later point in time
with open("test_study.pkl", "wb") as fout:
    pickle.dump(study, fout)
    # show best hyperparameters
print(study.best_trial.params)

if False:
    # LR finder
    pl.seed_everything(args.seed)
    trainer = pl.Trainer(
        gpus=0,
        # clipping gradients is a hyperparameter and important to prevent divergance
        # of the gradient for recurrent neural networks
        gradient_clip_val=0.1,
    )


    tft = TemporalFusionTransformer.from_dataset(
        training,
        # not meaningful for finding the learning rate but otherwise very important
        learning_rate=0.03,
        hidden_size=16,  # most important hyperparameter apart from learning rate
        # number of attention heads. Set to up to 4 for large datasets
        attention_head_size=1,
        dropout=0.1,  # between 0.1 and 0.3 are good values
        hidden_continuous_size=8,  # set to <= hidden_size
        output_size=7,  # 7 quantiles by default
        loss=SMAPE(),
        # reduce learning rate if no improvement in validation loss after x epochs
        reduce_on_plateau_patience=4,
    )
    print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

    # find optimal learning rate
    res = trainer.tuner.lr_find(
        tft,
        train_dataloader=train_dataloader,
        val_dataloaders=val_dataloader,
        max_lr=10.0,
        min_lr=1e-6,
    )

    print(f"suggested learning rate: {res.suggestion()}")
    fig = res.plot(show=True, suggest=True)
    fig.show()
