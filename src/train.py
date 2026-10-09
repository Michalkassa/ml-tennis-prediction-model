"""Usage: uv run python src/train.py --model decision_tree --tuner grid_search"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score

from dataset import load_data
from estimators import ESTIMATORS
from tuning import TUNERS

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "models" / "runs"


def evaluate(model, X_test, y_test):
    """Return a dict of metrics, e.g. accuracy, f1, roc_auc."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "log_loss": log_loss(y_test, y_proba),
    }


def save_run(model, params, metrics, model_name, tuner_name):
    """Write models/runs/<model>/<tuner>/<timestamp>/model.joblib and results.json (params + metrics)."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS_DIR / model_name / tuner_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, run_dir / "model.joblib")
    results = {
        "model": model_name,
        "tuner": tuner_name,
        "timestamp": timestamp,
        "params": params,
        "metrics": metrics,
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))
    return run_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=ESTIMATORS, required=True)
    parser.add_argument("--tuner", choices=TUNERS, required=True)
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = load_data()

    start = time.perf_counter()
    model, params = TUNERS[args.tuner].tune(ESTIMATORS[args.model], X_train, y_train)
    train_seconds = time.perf_counter() - start

    metrics = evaluate(model, X_test, y_test)
    metrics["train_seconds"] = train_seconds
    run_dir = save_run(model, params, metrics, args.model, args.tuner)

    print(f"params: {params}")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    print(f"saved to {run_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
