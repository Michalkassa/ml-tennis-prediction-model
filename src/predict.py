"""Predict upcoming tennis matches with the model trained in model.py.

Usage:
    python src/predict.py                          # uses data/upcoming_matches.csv
    python src/predict.py path/to/matches.csv
"""
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "tennis_model.joblib"
DEFAULT_MATCHES = ROOT / "data" / "upcoming_matches.csv"

FEATURES = ["surface", "best_of", "rank_diff", "rank_points_diff", "age_diff", "height_diff"]
SURFACES = {"hard": 0, "clay": 1, "grass": 2}


def build_features(matches):
    """Turn one row per match (both players' stats) into the model's features."""
    return pd.DataFrame({
        "surface": matches["surface"].str.lower().map(SURFACES),
        "best_of": matches["best_of"],
        "rank_diff": matches["player1_rank"] - matches["player2_rank"],
        "rank_points_diff": matches["player1_rank_points"] - matches["player2_rank_points"],
        "age_diff": matches["player1_age"] - matches["player2_age"],
        "height_diff": matches["player1_height"] - matches["player2_height"],
    })[FEATURES]


def predict(matches):
    """Add predicted winner and win probability columns to a matches DataFrame."""
    model = joblib.load(MODEL_PATH)
    features = build_features(matches)

    if features["surface"].isna().any():
        raise ValueError(f"surface must be one of: {', '.join(SURFACES)}")

    p1_win = model.predict_proba(features)[:, 1]
    result = matches[["player1", "player2"]].copy()
    result["winner"] = matches["player1"].where(p1_win >= 0.5, matches["player2"])
    result["player1_win_prob"] = p1_win.round(3)
    return result


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MATCHES
    print(predict(pd.read_csv(path)).to_string(index=False))
