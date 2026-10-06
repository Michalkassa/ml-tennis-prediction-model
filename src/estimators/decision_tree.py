from sklearn.tree import DecisionTreeClassifier

# Used by tuning/grid_search.py: every combination gets tried.
PARAM_GRID = {}

# Used by tuning/random_search.py: lists or scipy.stats distributions to sample from.
PARAM_DISTRIBUTIONS = {}


def build(**params):
    """Return an unfitted DecisionTreeClassifier with these params."""
    raise NotImplementedError
