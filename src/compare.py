"""Usage: uv run python src/compare.py"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "models" / "runs"


def load_runs():
    """Read every results.json under RUNS_DIR into one DataFrame, one row per run."""
    rows = []
    for path in RUNS_DIR.glob("*/*/*/results.json"):
        results = json.loads(path.read_text())
        rows.append({
            "model": results["model"],
            "tuner": results["tuner"],
            "timestamp": results["timestamp"],
            **results["metrics"],
            "params": results["params"],
        })
    return pd.DataFrame(rows)


def main():
    """Print load_runs() sorted by your main metric."""
    runs = load_runs()
    if runs.empty:
        print(f"No runs found in {RUNS_DIR.relative_to(ROOT)}. Train one with src/train.py first.")
        return

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)
    print(runs.sort_values("roc_auc", ascending=False).to_string(index=False, float_format="{:.4f}".format))


if __name__ == "__main__":
    main()
