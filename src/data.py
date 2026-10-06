# DATA FROM https://github.com/JeffSackmann/tennis_atp
import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

#combine datasets
raw_path = str(ROOT / "data" / "raw")
files = glob.glob(raw_path + "/*.csv")
total_raw_data = []
for file in files:
    df = pd.read_csv(file)
    total_raw_data.append(df)

data = pd.concat(total_raw_data, ignore_index=True)
pd.set_option('display.max_columns', None)

data_filtered = data.dropna(subset=[
    "winner_id", "loser_id", "winner_ht", "loser_ht","winner_age", "loser_age",
    "w_ace","w_df","w_svpt","w_1stIn","w_1stWon","w_2ndWon","w_SvGms","w_bpSaved","w_bpFaced",
    "l_ace","l_df","l_svpt","l_1stIn","l_1stWon","l_2ndWon","l_SvGms","l_bpSaved","l_bpFaced",
    "surface","winner_rank_points","loser_rank_points", "winner_rank", "loser_rank"
])
data_filtered = data_filtered.reset_index(drop=True)
print(data_filtered)

SURFACES = {"hard": 0, "clay": 1, "grass": 2}

final_data = pd.DataFrame({
    "winner_id": data_filtered["winner_id"],
    "loser_id": data_filtered["loser_id"],
    "surface": data_filtered["surface"].str.lower().map(SURFACES),
    "best_of": data_filtered["best_of"],
    "draw_size": data_filtered["draw_size"],
    "rank_diff": data_filtered["winner_rank"] - data_filtered["loser_rank"],
    "rank_points_diff": data_filtered["winner_rank_points"] - data_filtered["loser_rank_points"],
    "age_diff": data_filtered["winner_age"] - data_filtered["loser_age"],
    "height_diff": data_filtered["winner_ht"] - data_filtered["loser_ht"],
})

X = final_data[["surface", "best_of", "draw_size", "rank_diff", "rank_points_diff", "age_diff", "height_diff"]]

X.to_csv(ROOT / "data" / "processed" / "tennis_features.csv", index=False)
