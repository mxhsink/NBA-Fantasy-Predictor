# NBA Fantasy Predictor

Predicting each NBA player's box-score stats for their **next game**, so the predictions can be turned into **ESPN fantasy points** for a points league.

> **Status: Version 1 complete, Version 2 in progress.** Version 1 trained, compared and tested one model per stat on the 2025-26 regular season. Version 2 has added longer-term features (last 10 games and season average) and opponent features (what the other team gives up), and each step improved every stat on validation. Next up is combining the stat predictions into a fantasy-point forecast (see [Roadmap](#roadmap)).

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

### 3. Features (47 in total)
All features are known **before tip-off**.

| Feature | What it captures |
|---|---|
| `HOME` | Home or away game |
| `REST_DAYS`, `BACK_TO_BACK` | Days since the last game (capped at 4), and a flag for back-to-back games |
| `<STAT>_LAST5` (one per stat) | Average of each stat over the player's previous 5 games: **current form** |
| `<STAT>_LAST10` (one per stat) | Average over the previous 10 games: **recent form, less noisy** *(added in Version 2)* |
| `<STAT>_SEASONAVG` (one per stat) | Average over **every** previous game this season, using pandas `expanding()`: **normal level** *(added in Version 2)* |
| `MIN_LAST5`, `FTA_LAST5` | Recent minutes and free-throw attempts |
| `USG_PCT_LAST5` | Recent usage rate, **weighted by minutes** |
| `FG_PCT_LAST5` | Recent shooting %, computed as total FGM ÷ total FGA over the window, not as an average of percentages |
| `OPP_<STAT>_ALLOWED` (one per stat) | How much of each stat **tonight's opponent** has given up per game so far this season: **matchup difficulty** *(added in Version 2)* |

### 4. Preventing data leakage
- Every rolling and season-average feature uses `shift(1)` before `rolling()` or `expanding()`, so a game's own stats never appear in its features.
- Opponent features follow the same rule: each team's average uses only its **earlier** games (`shift(1)` + `expanding()`, grouped by team).
- Data is sorted by player (or team) and date before every `shift`, `rolling` and `diff`, and re-sorted after every merge.
- The train, validation and test sets are split **by date**, not at random, so the model is never trained on games played after the games it predicts:

| Split | Dates | Rows |
|---|---|---|
| Train | Oct 21, 2025 to Jan 31, 2026 | 11,057 |
| Validation | Feb 1 to Mar 14, 2026 | 5,619 |
| Test (held back for the final check) | Mar 15 to Apr 12, 2026 | 4,526 |

The **first 10 games for each player** are dropped, because they don't have a full 10-game window yet. (Version 1 only dropped the first 5 games, which gave 13,333 / 5,760 / 4,707 rows.)

### 5. Models and tuning
For each stat, four models are tuned with `GridSearchCV` using `TimeSeriesSplit(n_splits=4)` (cross-validation that respects time order) and scored on mean absolute error (MAE):

- **Linear regression** (with feature scaling)
- **Ridge regression** (alpha: 0.1, 1, 10, 100)
- **Random forest** (300 trees; min samples per leaf: 20/50/100; max features: 0.5/0.8)
- **HistGradientBoosting** (learning rate: 0.03/0.1; min samples per leaf: 20/50/100)

Each model is compared to a **baseline**: the stat's own 5-game average.

## Version 1 results

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

## Version 2 progress: longer-term features

Version 1's main finding was that the features, not the models, were the limit. So the first Version 2 change adds more information: each stat's **last 10 game average** and **season average**.

### Fair before/after check (ablation)
To see whether the new features really help, the same models were trained twice on **exactly the same rows**: once with the Version 1 features only, and once with all features. The baseline MAE was identical in both runs (4.841), which confirms the rows matched.

| PTS (validation) | Version 1 features | + last 10 and season average |
|---|---|---|
| Best model MAE | 4.722 (linear) | **4.644** (ridge) |
| Better than baseline by | 2.5% | **4.1%** |
| Train/val gap (linear) | 0.025 | 0.093 |

### All stats (validation, % better than the 5-game baseline)

| Stat | Version 1 | Version 2 | Change |
|---|---|---|---|
| TOV | 4.5% | **5.6%** | +1.0 |
| STL | 4.1% | **5.5%** | +1.4 |
| AST | 1.8% | **5.2%** | +3.4 |
| FGM | 3.2% | **4.8%** | +1.6 |
| REB | 1.5% | **4.2%** | +2.7 |
| FTM | 1.0% | **4.1%** | +3.1 |
| PTS | 2.3% | **4.1%** | +1.7 |
| FGA | 1.2% | **3.2%** | +2.0 |
| 3PM | 0.2% | **2.9%** | +2.7 |
| BLK | −3.3% | **−0.7%** | +2.6 |

*Version 2 drops each player's first 10 games instead of 5, so the rows differ slightly from Version 1. Percent-better-than-baseline is used here because it compares fairly across slightly different rows, and the PTS check above uses identical rows.*

### What changed
1. **Every stat improved**, with the biggest gains for assists, free throws, rebounds and threes. These stats seem to depend more on a player's **normal level** than on his last 5 games.
2. **Ridge regression now edges out linear on 7 of the 10 stats**, though only by a few thousandths. With 37 features, many of which overlap (a 5-game, 10-game and season average of the same stat), Ridge's penalty on large weights helps a little. Tree models still come last.
3. **Blocks are almost even with the baseline** (−0.7%, up from −3.3%), but still the only stat where the models don't win.
4. **The models overfit a little more** (larger train/val gaps). This is the trade-off with adding features.

### Opponent features
Each player row gets the opponent's season-to-date average of every stat **allowed**, so the model knows whether tonight's matchup is easy or hard. How it's built:
1. The opponent comes from the matchup text (`"LAL @ UTA"` → `UTA`).
2. All player stats in each game are added up by opponent, which gives what each team **gave up** in each game (2,460 team-games = 30 teams × 82).
3. Each team's average before each game is computed (`shift(1)` + `expanding()`), then merged back onto the player rows by opponent and game.

A same-rows check on rebounds first confirmed the effect (ridge val MAE 1.915 → 1.906). Then all 10 stats were rerun. The rows are identical to the previous run (every baseline MAE matches), so the comparison is fair.

| Stat | Before opponent features | **With opponent features** | Change |
|---|---|---|---|
| BLK | −0.7% (baseline won) | **+0.4%** | **+1.1** |
| FTM | 4.1% | **5.0%** | **+0.9** |
| TOV | 5.6% | **6.2%** | **+0.6** |
| STL | 5.5% | **5.9%** | +0.5 |
| FGM | 4.8% | **5.3%** | +0.5 |
| REB | 4.2% | **4.6%** | +0.4 |
| PTS | 4.1% | **4.5%** | +0.4 |
| FGA | 3.2% | **3.4%** | +0.2 |
| AST | 5.2% | **5.4%** | +0.2 |
| 3PM | 2.9% | **3.0%** | +0.1 |

*Validation set, best model per stat, % better than the 5-game baseline. PTS error went from 4.644 to 4.624 points per game.*

**What the opponent features showed:**
1. **For the first time, a model beats the baseline on all 10 stats.** Blocks, the weak spot since Version 1, now beat the baseline (linear, +0.4%).
2. **Free throws gained the most.** Some teams foul far more than others, so the opponent matters a lot for free-throw attempts.
3. **Three-pointers barely changed** (+0.1). I expected the opposite, but how many threes a player makes depends mostly on the player, not the defense.
4. **Ridge is the best model on 9 of 10 stats.** The random forest overfits badly with 47 features (its PTS train/val gap grew to 0.77, while Ridge's stayed at 0.08).

> **Note on the test set:** while building Version 2, the test-set cell was run once by accident with the new features. Those numbers have **not** been used for any decision. The final Version 2 test result will be reported with this note.

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

The first cell downloads the season's data from the NBA stats API and saves it to `data/raw/`. Running all 10 stats × 4 models takes several minutes on a laptop.

**Tech stack:** Python 3.13 · pandas · scikit-learn · nba_api · Jupyter

## Roadmap

**Version 2**
- [ ] Combine the best model for each stat into one **fantasy-point prediction**, and measure it against a baseline fantasy-point prediction
- [x] New features: last 10 game and season averages for every stat
- [x] Opponent features: stats allowed by tonight's opponent, for every stat
- [ ] More new features: opponent pace and defensive rating, teammates missing through injury
- [x] Beat the baseline on blocks (the opponent features did it)
- [ ] Move the notebook code into reusable scripts in `scripts/`
- [x] Add a `.gitignore`
- [ ] Add a `requirements.txt`

**Later**
- [ ] Daily predictions for upcoming games
- [ ] A simple start/sit dashboard
