from scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier

# Used by tuning/grid_search.py: every combination gets tried.
PARAM_GRID = {
    "criterion": ["gini", "entropy"],
    "max_depth": [3, 5, 8, 12, None],
    "min_samples_split": [2, 20, 100],
    "min_samples_leaf": [1, 10, 50],
}

# Used by tuning/random_search.py: lists or scipy.stats distributions to sample from.
PARAM_DISTRIBUTIONS = {
    "criterion": ["gini", "entropy", "log_loss"],
    "max_depth": randint(2, 20),
    "min_samples_split": randint(2, 200),
    "min_samples_leaf": randint(1, 100),
    "max_features": [None, "sqrt", "log2"],
}


def build(**params):
    """Return an unfitted DecisionTreeClassifier with these params."""
    return DecisionTreeClassifier(random_state=42, **params)
