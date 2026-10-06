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

#remove empty data

data_filtered = data.dropna(subset=[
    "winner_id", "loser_id", "winner_ht", "loser_ht","winner_age", "loser_age",
    "w_ace","w_df","w_svpt","w_1stIn","w_1stWon","w_2ndWon","w_SvGms","w_bpSaved","w_bpFaced",
    "l_ace","l_df","l_svpt","l_1stIn","l_1stWon","l_2ndWon","l_SvGms","l_bpSaved","l_bpFaced",
    "surface","winner_rank_points","loser_rank_points", "winner_rank", "loser_rank"
])
data_filtered = data_filtered.reset_index
print(data_filtered)

final_data["WINNER_ID"] = final_data["WINNER_ID"]
final_data["LOSER_ID"] = final_data["loser_id"]
final_data["ATP_POINT_DIFF"] = final_data["winner_rank_points"] - final_data["loser_rank_points"]
final_data["AGE_DIFF"] = final_data["winner_age"] - final_data["loser_age"]
final_data["HEIGHT_DIFF"] = final_data["winner_ht"] - final_data["loser_ht"]
final_data["ATP_RANK_DIFF"] = final_data["winner_rank"] - final_data["loser_rank"]
final_data["BEST_OF"] = final_data["best_of"]
final_data["draw_size"] = final_data["DRAW_SIZE"]

X.to_csv(ROOT / "data" / "processed" / "tennis_features.csv", index=False)
Y.to_csv(ROOT / "data" / "processed" / "tennis_labels.csv", index=False)
