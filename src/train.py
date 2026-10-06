"""Usage: uv run python src/train.py --model decision_tree --tuner grid_search"""
import argparse
from pathlib import Path

from dataset import load_data
from estimators import ESTIMATORS
from tuning import TUNERS

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "models" / "runs"


def evaluate(model, X_test, y_test):
    """Return a dict of metrics, e.g. accuracy, f1, roc_auc."""
    raise NotImplementedError


def save_run(model, params, metrics, model_name, tuner_name):
    """Write models/runs/<model>/<tuner>/<timestamp>/model.joblib and results.json (params + metrics)."""
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=ESTIMATORS, required=True)
    parser.add_argument("--tuner", choices=TUNERS, required=True)
    args = parser.parse_args()

    # load_data -> TUNERS[args.tuner].tune(ESTIMATORS[args.model], ...) -> evaluate -> save_run
    raise NotImplementedError


if __name__ == "__main__":
    main()
