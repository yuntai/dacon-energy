from tst_dataset import load_dataset
import matplotlib.pyplot as plt
import argparse
import copy
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

parser = argparse.ArgumentParser()
parser.add_argument("--path", "-p", type=str, default="./logs/default/version_0/checkpoints/epoch=22-step=689.ckpt")
args = parser.parse_args()

best_tft = TemporalFusionTransformer.load_from_checkpoint(args.path)

tr_ds, va_ds = load_dataset("./data")

# create dataloaders for model
batch_size = 128
tr_loader = tr_ds.to_dataloader(
    train=True, batch_size=batch_size, num_workers=12
)
va_loader = va_ds.to_dataloader(
    train=False, batch_size=batch_size*10, num_workers=12
)

raw_predictions, x = best_tft.predict(va_loader, mode="raw", return_x=True)
interpretation = best_tft.interpret_output(raw_predictions, reduction="sum")
best_tft.plot_interpretation(interpretation)
plt.show()

# show wrost performer
actuals = torch.cat([y[0] for x, y in iter(va_loader)])
predictions = best_tft.predict(va_loader)

# calcualte metric by which to display
predictions = best_tft.predict(va_loader)
mean_losses = SMAPE(reduction="none")(predictions, actuals).mean(1)
indices = mean_losses.argsort(descending=True)  # sort losses
for idx in range(10):  # plot 10 examples
    best_tft.plot_prediction(
        x, raw_predictions, idx=indices[idx], add_loss_to_title=SMAPE(quantiles=best_tft.loss.quantiles)
    );

