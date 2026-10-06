# ml-tennis-prediction-model

A machine learning project that predicts the winner of a men's professional tennis match.

## What it is

- Predicts which of two players will win, given basic facts about the match and the players
- Trained on ATP Challenger and qualifying match results from 2000 to 2024
- Uses a decision tree classifier from scikit-learn
- Work in progress: the data pipeline and model are still being built

## Data

- Source: [Jeff Sackmann's tennis_atp dataset](https://github.com/JeffSackmann/tennis_atp)
- `data/raw/`: one CSV of match results per year
- `data/processed/`: cleaned features (`tennis_features.csv`) and labels (`tennis_labels.csv`)
- Matches with missing player or match stats are dropped

## Features

Each match is described as differences between the two players:

- Surface (Hard, Clay, Grass)
- Best of 3 or 5 sets
- Ranking difference
- Ranking points difference
- Age difference
- Height difference

## Project structure

- `src/data.py`: loads the raw CSVs, cleans them and builds the features
- `src/model.py`: trains the decision tree, reports test accuracy and saves it to `models/tennis_model.joblib`
- `src/predict.py`: loads the saved model and predicts the matches listed at the bottom of the file

## Getting started

- Install [uv](https://docs.astral.sh/uv/), then install the dependencies: `uv sync`
- Run `uv run python src/model.py` to train and save the model (about 63% test accuracy)
- Run `uv run python src/predict.py` to predict the matches listed in `src/predict.py`

## Predicting upcoming matches

- Open `src/predict.py` and add each match to the `MATCHES` list at the bottom:

  ```python
  (
      player("Player A", rank=45, rank_points=1200, age=24.5, height=188),
      player("Player B", rank=120, rank_points=520, age=29.1, height=180),
      "hard", 3,   # surface (hard/clay/grass), best_of
  ),
  ```

- Run `uv run python src/predict.py`
- Each line of output shows the predicted winner and their win probability
- For one match, call `predict_match(player1, player2, surface, best_of)`, which returns the probability that player 1 wins

## License

- MIT, see [LICENSE](LICENSE)
