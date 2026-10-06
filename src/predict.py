"""Predict upcoming tennis matches with the model trained in model.py.

To predict a match, add it to MATCHES at the bottom of this file, then run:
    uv run python src/predict.py
"""
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "runs" / "<model>" / "<tuner>" / "<timestamp>" / "model.joblib"

FEATURES = ["surface", "best_of", "rank_diff", "rank_points_diff", "age_diff", "height_diff"]
SURFACES = {"hard": 0, "clay": 1, "grass": 2}

# Load the model once, when the script starts.
model = joblib.load(MODEL_PATH)


def player(name, rank, rank_points, age, height):
    return {"name": name, "rank": rank, "rank_points": rank_points, "age": age, "height": height}


def predict_match(player1, player2, surface, best_of=3):
    if surface.lower() not in SURFACES:
        raise ValueError(f"surface must be one of: {', '.join(SURFACES)}")

    features = pd.DataFrame([{
        "surface": SURFACES[surface.lower()],
        "best_of": best_of,
        "rank_diff": player1["rank"] - player2["rank"],
        "rank_points_diff": player1["rank_points"] - player2["rank_points"],
        "age_diff": player1["age"] - player2["age"],
        "height_diff": player1["height"] - player2["height"],
    }])[FEATURES]

    return model.predict_proba(features)[0, 1]


def show(player1, player2, surface, best_of=3):
    p1_win = predict_match(player1, player2, surface, best_of)
    winner, prob = (player1, p1_win) if p1_win >= 0.5 else (player2, 1 - p1_win)
    print(f"{player1['name']} vs {player2['name']} ({surface}, best of {best_of})"
          f"  ->  {winner['name']} wins ({prob:.0%})")


# ---------------------------------------------------------------------------
# Upcoming matches: edit below
#
#   player(name, rank, rank_points, age, height_cm)
#   (player1, player2, surface, best_of)    surface: "hard", "clay" or "grass"
# ---------------------------------------------------------------------------

MATCHES = [
    (
        player("ADM", rank=8, rank_points=3430, age=27, height=183),
        player("Djokovic", rank=10, rank_points=3310, age=39, height=188),
        "hard", 3,
    ),
]

if __name__ == "__main__":
    for match in MATCHES:
        show(*match)
