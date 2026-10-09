from sklearn.model_selection import GridSearchCV


def tune(estimator, X_train, y_train):
    """GridSearchCV over estimator.PARAM_GRID. Return (best_estimator_, best_params_)."""
    search = GridSearchCV(estimator.build(), estimator.PARAM_GRID, scoring="roc_auc", cv=5, n_jobs=-1)
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_
