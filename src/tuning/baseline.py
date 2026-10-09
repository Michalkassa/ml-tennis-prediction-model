def tune(estimator, X_train, y_train):
    """No search: fit estimator.build() with defaults. Return (fitted_model, params)."""
    model = estimator.build()
    model.fit(X_train, y_train)
    return model, {}
