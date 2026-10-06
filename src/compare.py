"""Usage: uv run python src/compare.py"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "models" / "runs"


def load_runs():
    """Read every results.json under RUNS_DIR into one DataFrame, one row per run."""
    raise NotImplementedError


def main():
    """Print load_runs() sorted by your main metric."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
