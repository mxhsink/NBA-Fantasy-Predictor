# NBA Fantasy Predictor

Predicting each NBA player's box-score stats for their **next game**, so the predictions can be turned into **ESPN fantasy points** for a points league.

> **Status: Version 1.** One model per stat, trained, compared and tested on the 2025-26 regular season. Combining the stat predictions into a fantasy-point forecast comes next (see [Roadmap](#roadmap)).

---

## Why I built this

In a fantasy points league, every stat is worth a fixed number of points (a steal is worth 4, a turnover costs 2, and so on). Deciding who to start comes down to one question: **how many points will this player score in his next game?**

The common shortcut is to use a player's recent average. This project tests whether machine learning can beat that shortcut, and by how much.

## Approach

### 1. Data
- **Source:** official NBA stats through the [`nba_api`](https://github.com/swar/nba_api) package
  - `LeagueGameLog`: one row per player per game (points, rebounds, assists and other box-score stats)
  - `PlayerGameLogs` (Advanced): usage rate (`USG_PCT`) per game
- **Scope:** 2025-26 regular season only (playoffs left out on purpose)
- **Size:** 26,651 player-games from 582 players (Oct 21, 2025 to Apr 12, 2026)

### 2. One model per stat
Instead of predicting fantasy points directly, I predict each **raw stat** separately and then apply the league's scoring rules:

| Stat | FGA | FGM | PTS | 3PM | REB | AST | STL | BLK | FTM | TOV |
|---|---|---|---|---|---|---|---|---|---|---|
| Fantasy points | −1 | +2 | +1 | +1 | +1 | +2 | +4 | +4 | +1 | −2 |

**Why:** the scoring lives in a single dictionary. If the league changes its rules, the predictions don't need retraining. It also shows which stats are predictable and which aren't.

### 3. Features (17 in total)
All features are known **before tip-off**.

| Feature | What it captures |
|---|---|
| `HOME` | Home or away game |
| `REST_DAYS`, `BACK_TO_BACK` | Days since the last game (capped at 4), and a flag for back-to-back games |
| `<STAT>_LAST5` (one per stat) | Average of each stat over the player's previous 5 games |
| `MIN_LAST5`, `FTA_LAST5` | Recent minutes and free-throw attempts |
| `USG_PCT_LAST5` | Recent usage rate, **weighted by minutes** |
| `FG_PCT_LAST5` | Recent shooting %, computed as total FGM ÷ total FGA over the window, not as an average of percentages |

### 4. Preventing data leakage
- Every rolling feature uses `shift(1)` before `rolling(5)`, so a game's own stats never appear in its features.
- Data is sorted by player and date before every `shift`, `rolling` and `diff`.
- The train, validation and test sets are split **by date**, not at random, so the model is never trained on games played after the games it predicts:

| Split | Dates | Rows |
|---|---|---|
| Train | Oct 21, 2025 to Jan 31, 2026 | 13,333 |
| Validation | Feb 1 to Mar 14, 2026 | 5,760 |
| Test (**used once, at the very end**) | Mar 15 to Apr 12, 2026 | 4,707 |

The **first 5 games for each player** are dropped, because they don't have a full rolling window yet.

### 5. Models and tuning
For each stat, four models are tuned with `GridSearchCV` using `TimeSeriesSplit(n_splits=4)` (cross-validation that respects time order) and scored on mean absolute error (MAE):

- **Linear regression** (with feature scaling)
- **Ridge regression** (alpha: 0.1, 1, 10, 100)
- **Random forest** (300 trees; min samples per leaf: 20/50/100; max features: 0.5/0.8)
- **HistGradientBoosting** (learning rate: 0.03/0.1; min samples per leaf: 20/50/100)

Each model is compared to a **baseline**: the stat's own 5-game average.

## Results

For each stat, the model with the lowest **validation** error was picked. That one model was then scored **once** on the locked test set (the last 4 weeks of the season, which no model had seen).

| Stat | Chosen model | Val MAE | Test MAE | Baseline test MAE | Improvement on test |
|---|---|---|---|---|---|
| TOV | Ridge | 0.895 | 0.922 | 0.955 | **+3.4%** |
| FGM | Linear | 1.789 | 1.840 | 1.903 | **+3.3%** |
| PTS | Linear | 4.697 | 4.828 | 4.989 | **+3.2%** |
| FGA | Linear | 2.859 | 2.949 | 3.001 | **+1.8%** |
| AST | Ridge | 1.392 | 1.449 | 1.474 | **+1.7%** |
| STL | Linear | 0.735 | 0.732 | 0.744 | **+1.7%** |
| FTM | Linear | 1.375 | 1.377 | 1.398 | **+1.5%** |
| REB | Linear | 1.957 | 1.977 | 2.006 | **+1.4%** |
| 3PM | Random forest | 0.931 | 0.933 | 0.939 | +0.6% |
| BLK | *Baseline* | 0.527 | 0.516 | 0.516 | 0% (every model lost to the baseline on validation) |

*MAE is in the stat's own units. For example, a PTS test MAE of 4.828 means predictions are off by about 4.8 points per game on average.*

**The results held up on the test set.** Every chosen model still beat the baseline on unseen games, so the validation results weren't a fluke of the tuning.

### Key findings
1. **Linear models beat tree models for 8 of the 10 stats.** They also had the smallest gap between train and validation error, so they overfit the least. Random forest and boosting fit the training data better but did worse on new games.
2. **Machine learning only beats the baseline by about 0.6 to 3.4% on the test set.** Since every model type ends up in the same place, the limit is **the information in the features, not the choice of model**. The next improvements need new kinds of features, such as opponent strength or teammate injuries, rather than more complex models.
3. **Blocks are the exception.** On validation, every model did 3.3 to 4.5% worse than the 5-game average, so the baseline is used for blocks. Blocks are rare and lopsided (most players get 0 or 1 per game). A likely cause is that the models are trained to minimize squared error but judged on absolute error, and the two disagree most on lopsided stats like this one. This is the first thing to investigate in Version 2.
4. **Turnovers, made shots and points improve the most** (over 3% on the test set). Three-pointers barely beat the 5-game average.

## Project structure

```
NBA-Fantasy-Predictor/
├── data/
│   ├── raw/            # API downloads, never edited
│   └── processed/      # derived tables
├── notebooks/
│   └── main.ipynb      # full pipeline: data → features → models → results
├── models/             # saved models (Version 2)
├── scripts/            # reusable code (Version 2)
└── README.md
```

## How to run

```bash
git clone https://github.com/mxhsink/NBA-Fantasy-Predictor.git
cd NBA-Fantasy-Predictor

python3 -m venv myenv
source myenv/bin/activate
pip install nba_api pandas numpy scikit-learn matplotlib jupyter

jupyter notebook notebooks/main.ipynb   # then Kernel → Restart & Run All
```

The first cell downloads the season's data from the NBA stats API and saves it to `data/raw/`. Running all 10 stats × 4 models takes about **8 minutes** on a laptop.

**Tech stack:** Python 3.13 · pandas · scikit-learn · nba_api · Jupyter

## Roadmap

**Version 2**
- [ ] Combine the best model for each stat into one **fantasy-point prediction**, and measure it against a baseline fantasy-point prediction
- [ ] New features: opponent defensive rating and pace, teammates missing through injury, season averages alongside the 5-game averages
- [ ] Fix the blocks result (MAE-based loss functions, or modeling block counts directly)
- [ ] Move the notebook code into reusable scripts in `scripts/`
- [ ] Add a `requirements.txt` and `.gitignore`

**Later**
- [ ] Daily predictions for upcoming games
- [ ] A simple start/sit dashboard
