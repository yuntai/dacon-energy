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
parser.add_argument('--num', '-n', type=int, default=-1)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--tune', default=False, action='store_true')
#parser.add_argument("--nums", nargs="+", default=["a", "b"])
args = parser.parse_args()
print(args)

from tst_load_data import load_data

train_dataloader, val_dataloader, training = load_data("./data")

# calculate baseline mean absolute error, i.e. predict next value as the last available value from the history
actuals = torch.cat([y for x, (y, weight) in iter(val_dataloader)])
baseline_predictions = Baseline().predict(val_dataloader)
print("baseline smape=", SMAPE(reduction='mean')(actuals, baseline_predictions).mean().item())

if args.tune:
    from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
    print("start studying...")
    # create study
    study = optimize_hyperparameters(
        train_dataloader,
        val_dataloader,
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
    sys.exit(0)

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

    sys.exit(0)

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
logger = TensorBoardLogger("lightning_logs")  # log to tensorboard

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
    training,
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
    train_dataloader=train_dataloader,
    val_dataloaders=val_dataloader
)

# evaluate

# load the best model according to the validation loss (given that
# we use early stopping, this is not necessarily the last epoch)
best_model_path = trainer.checkpoint_callback.best_model_path
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
# calculate mean absolute error on validation set
actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
predictions = best_tft.predict(val_dataloader)
smape_per_num = SMAPE(reduction="none")(predictions, actuals).mean(1)
print(f"{best_model_path=}")
print(smape_per_num)
print(smape_per_num.mean())

#
#tensor([0.0069, 0.0735, 0.0497, 0.0495, 0.0448, 0.0355, 0.0910, 0.0306, 0.0657,
#        0.1214, 0.1062, 0.0385, 0.0892, 0.0812, 0.0638, 0.1145, 0.0426, 0.1075,
#        0.0434, 0.1144, 0.0448, 0.0858, 0.0223, 0.0770, 0.0137, 0.0069, 0.0147,
#        0.1224, 0.0994, 0.0605, 0.0457, 0.0452, 0.0716, 0.1777, 0.0997, 0.0781,
#        0.1085, 0.0664, 0.0379, 0.0584, 0.0622, 0.0473, 0.1357, 0.1095, 0.1084,
#        0.1211, 0.0623, 0.0427, 0.0513, 0.0581, 0.0644, 0.0776, 0.0952, 0.0631,
#        0.1442, 0.0678, 0.0438, 0.1025, 0.0849, 0.0352])

