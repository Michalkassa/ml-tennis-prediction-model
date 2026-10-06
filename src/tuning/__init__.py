from tuning import baseline, grid_search, random_search

TUNERS = {
    "baseline": baseline,
    "grid_search": grid_search,
    "random_search": random_search,
}
