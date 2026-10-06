from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = ROOT / "data" / "processed" / "tennis_features.csv"
FEATURES = ["surface", "best_of", "rank_diff", "rank_points_diff", "age_diff", "height_diff"]
DIFF_COLS = ["rank_diff", "rank_points_diff", "age_diff", "height_diff"]


def load_data(test_size=0.2, random_state=42):
    """Return X_train, X_test, y_train, y_test."""
    X = pd.read_csv(FEATURES_PATH)[FEATURES].dropna()

    rng = np.random.default_rng(random_state)
    swap = rng.random(len(X)) < 0.5
    X.loc[swap, DIFF_COLS] *= -1
    y = pd.Series(np.where(swap, 0, 1), index=X.index, name="player1_wins")

    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)