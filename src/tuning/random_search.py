from sklearn.model_selection import RandomizedSearchCV


def tune(estimator, X_train, y_train):
    """RandomizedSearchCV over estimator.PARAM_DISTRIBUTIONS. Return (best_estimator_, best_params_)."""
    search = RandomizedSearchCV(
        estimator.build(), estimator.PARAM_DISTRIBUTIONS,
        n_iter=50, scoring="roc_auc", cv=5, n_jobs=-1, random_state=42,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_
