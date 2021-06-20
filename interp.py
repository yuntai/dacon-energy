from tst_load_data import load_data
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
parser.add_argument("--path", "-p", type=str, default="lightning_logs/default/version_56/checkpoints/epoch=19-step=599.ckpt")
args = parser.parse_args()

best_model_path=args.path
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
tr_loader, va_loader, training = load_data("./data")

raw_predictions, x = best_tft.predict(va_loader, mode="raw", return_x=True)
interpretation = best_tft.interpret_output(raw_predictions, reduction="sum")
best_tft.plot_interpretation(interpretation)
plt.show()


