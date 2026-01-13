
import pandas as pd
import numpy as np

# Surface encoding used in your pipeline
SURFACE_MAP = {"Hard": 0, "Clay": 1, "Grass": 2}

def prepare_single_match_row(
    *,
    p1_id,
    p2_id,
    p1_rank=None, p2_rank=None,
    p1_rank_points=None, p2_rank_points=None,
    p1_age=None, p2_age=None,
    p1_ht=None, p2_ht=None,
    p1_seed=0, p2_seed=0,
    surface="Hard",
    best_of=3,
    X_model_columns=None,
    id_columns_in_model=False
):
    """
    Build a one-row DataFrame aligned with your model's feature set.
    If id_columns_in_model=False (typical), Player IDs will be dropped from the row.

    Returns: DataFrame with columns == X_model_columns (same order).
    """
    # Map/validate surface
    surface_mapped = SURFACE_MAP.get(str(surface), -1)

    def to_num(x):
        return np.nan if x is None or x == "" else float(x)

    # Compute diffs (P1 - P2), mirroring your training pipeline
    rank_diff        = (to_num(p1_rank)         - to_num(p2_rank))         if (p1_rank is not None and p2_rank is not None) else np.nan
    rank_points_diff = (to_num(p1_rank_points)  - to_num(p2_rank_points))  if (p1_rank_points is not None and p2_rank_points is not None) else np.nan
    age_diff         = (to_num(p1_age)          - to_num(p2_age))          if (p1_age is not None and p2_age is not None) else np.nan
    height_diff      = (to_num(p1_ht)           - to_num(p2_ht))           if (p1_ht is not None and p2_ht is not None) else np.nan
    seed_diff        = (to_num(p1_seed)         - to_num(p2_seed))

    row = {
        "Player1_id": p1_id,
        "Player2_id": p2_id,
        "surface_mapped": int(surface_mapped),
        "best_of": int(best_of) if not pd.isna(best_of) else 3,
        "rank_diff": rank_diff,
        "rank_points_diff": rank_points_diff,
        "age_diff": age_diff,
        "height_diff": height_diff,
        "seed_diff": seed_diff,
    }

    df_row = pd.DataFrame([row])

    # If your model does NOT use IDs (recommended), drop them
    if not id_columns_in_model:
        df_row = df_row.drop(columns=[c for c in ["Player1_id", "Player2_id"] if c in df_row.columns], errors="ignore")

    # Align to model columns: add any missing cols with 0 and reorder
    if X_model_columns is not None:
        for col in X_model_columns:
            if col not in df_row.columns:
                df_row[col] = 0
        df_row = df_row[X_model_columns]

    # Simple numeric imputation for any NaNs
    df_row = df_row.fillna(df_row.median(numeric_only=True))

    return df_row
