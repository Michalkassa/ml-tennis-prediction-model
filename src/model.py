"""Tennis match prediction model.

Trains a decision tree that predicts whether player 1 beats player 2, then saves
it to models/tennis_model.joblib for predict.py to use.

Usage:
    python src/model.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = ROOT / "data" / "processed" / "tennis_features.csv"
MODEL_PATH = ROOT / "models" / "tennis_model.joblib"

FEATURES = ["surface", "best_of", "rank_diff", "rank_points_diff", "age_diff", "height_diff"]

# Load data
# Each row is one match, with every difference computed as winner minus loser.
# Rows with no surface are dropped.
X = pd.read_csv(FEATURES_PATH)[FEATURES].dropna()
print(X.head())

# Build labels
# As loaded, player 1 is always the winner, so there is nothing to learn. For a
# random half of the matches we swap the players: the differences flip sign and
# the label becomes 0.
#   1 = player 1 wins
#   0 = player 2 wins
rng = np.random.default_rng(42)
swap = rng.random(len(X)) < 0.5

diff_cols = ["rank_diff", "rank_points_diff", "age_diff", "height_diff"]
X.loc[swap, diff_cols] *= -1
y = pd.Series(np.where(swap, 0, 1), index=X.index, name="player1_wins")

print(y.value_counts())

# Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=99)

model = DecisionTreeClassifier(
    max_depth=6,
    min_samples_split=100,
    min_samples_leaf=50,
    random_state=42,
)
model.fit(X_train, y_train)

# Evaluate
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Accuracy: {accuracy:.3f}")

print(pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False))

# Save
# The saved model is what src/predict.py loads.
MODEL_PATH.parent.mkdir(exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f"Saved model to {MODEL_PATH}")
