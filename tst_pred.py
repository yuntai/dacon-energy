import argparse

import tst_dataset
import common
import pandas as pd

from pytorch_forecasting.models import TemporalFusionTransformer

parser = argparse.ArgumentParser();
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument("--path", "-p", type=str)
parser.add_argument("--outfn", "-o", type=str)
args = parser.parse_args()

best_tft = TemporalFusionTransformer.load_from_checkpoint(args.path)
max_encoder_length = best_tft.dataset_parameters['max_encoder_length']
max_prediction_length = best_tft.dataset_parameters['max_prediction_length']
print(f"{max_encoder_length=}, {max_prediction_length=}")

train_df, test_df = common.prep_tst("./data")

encoder_data = train_df[lambda x: x.time_idx > x.time_idx.max() - max_encoder_length]
last_data = train_df.iloc[[-1]]
target_cols = [c for c in test_df.columns if 'target' in c]
for c in target_cols:
    test_df.loc[:, c] = last_data[c].item()
decoder_data = test_df

# combine encoder and decoder data
new_prediction_data = pd.concat([encoder_data, decoder_data], ignore_index=True)

new_raw_predictions, new_x = best_tft.predict(new_prediction_data, mode="raw", return_x=True)

#for idx in range(10):  # plot 10 examples
#    best_tft.plot_prediction(new_x, new_raw_predictions, idx=idx, show_future_observed=False);

sub_df = pd.read_csv("./data/sample_submission.csv")
num_labels = best_tft.dataset_parameters['categorical_encoders']['num'].classes_
preds = new_raw_predictions['prediction'].squeeze()

for n, ix in num_labels.items():
    sub_df.loc[sub_df.num_date_time.str.startswith(f"{n} "), 'answer'] = preds[ix].numpy()

sub_df.to_csv(args.outfn, index=False)
