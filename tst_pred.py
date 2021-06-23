import argparse

import tst_dataset
import common
import pandas as pd

from pytorch_forecasting.models import TemporalFusionTransformer

parser = argparse.ArgumentParser();
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument("--path", "-p", type=str, default="./logs/default/version_12/checkpoints/epoch=21-step=659.ckpt")
args = parser.parse_args()

best_tft = TemporalFusionTransformer.load_from_checkpoint(args.path)
max_encoder_length = best_tft.dataset_parameters['max_encoder_length']
max_prediction_length = best_tft.dataset_parameters['max_prediction_length']
print(f"{max_encoder_length=}, {max_prediction_length=}")

train_df, test_df = common.prep_tst("./data")

encoder_data = train_df[lambda x: x.time_idx > x.time_idx.max() - max_encoder_length]
decoder_data = test_df[lambda x: x.time_idx - x.time_idx.min() < max_prediction_length]

# combine encoder and decoder data
new_prediction_data = pd.concat([encoder_data, decoder_data], ignore_index=True)

new_raw_predictions, new_x = best_tft.predict(new_prediction_data, mode="raw", return_x=True)

#for idx in range(10):  # plot 10 examples
#    best_tft.plot_prediction(new_x, new_raw_predictions, idx=idx, show_future_observed=False);
