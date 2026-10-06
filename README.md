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
- `src/predict.py`: loads the saved model and predicts upcoming matches
- `data/upcoming_matches.csv`: example input for `predict.py`

## Getting started

- Install [uv](https://docs.astral.sh/uv/), then install the dependencies: `uv sync`
- Run `uv run python src/model.py` to train and save the model (about 63% test accuracy)
- Run `uv run python src/predict.py` to predict the matches in `data/upcoming_matches.csv`

## Predicting upcoming matches

- Add one row per match to `data/upcoming_matches.csv` (or any CSV with the same columns)
- Columns: `player1`, `player2`, `surface` (Hard/Clay/Grass), `best_of`, and for each player their current ATP `rank`, `rank_points`, `age` and `height` in cm (e.g. `player1_rank`, `player2_rank`)
- Run `uv run python src/predict.py path/to/your.csv`
- The output shows the predicted winner and the probability that player 1 wins
- From Python: `from predict import predict`, then call `predict(df)` with a DataFrame that has the same columns

## License

- MIT, see [LICENSE](LICENSE)
