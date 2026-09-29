# NFL Sack Probability Model

Estimates the probability that an NFL dropback ends in a sack, using only information knowable before the snap.

The question it answers: "If this play is a pass, how likely is a sack?" It is conditional on a dropback occurring and does not predict whether the next play will be a pass.

## Data

- **Source:** [nflfastR](https://www.nflfastr.com/) play-by-play via the [nflverse data releases](https://github.com/nflverse/nflverse-data/releases/tag/pbp)
- **Seasons:** 2021, 2022, 2023
- **Size:** 149,021 plays, filtered to 65,131 dropbacks (`qb_dropback == 1`)
- **Sacks:** 4,130, a league base rate of about 6.3%

The notebook downloads the data automatically on first run and caches it in `data/` (not tracked in git).

## Features

Every feature is knowable pre-snap. No play-outcome columns are used.

| Group | Features |
|---|---|
| Game situation | `down`, `ydstogo`, `yardline_100`, `qtr`, `game_seconds_remaining`, `score_differential` |
| Formation / tempo | `shotgun`, `no_huddle` |
| Pass expectation | `xpass` (nflfastR expected pass probability) |
| Point-in-time priors | `qb_prior_sack_rate`, `off_prior_sack_rate`, `def_prior_sack_rate` |

### Bayesian sack-rate priors

The QB, offense, and defense priors are the model's main signal. Each is an expanding sack rate within a season, built from games strictly before the current one (`cumsum().shift(1)`), so no feature can see the play it is predicting.

Small samples are shrunk toward the league rate with a Bayesian prior worth 100 pseudo-dropbacks:

```
prior_rate = (prior_sacks + 100 * league_rate) / (prior_dropbacks + 100)
```

Early in the season every rate starts at the league average and moves toward the observed rate as dropbacks accumulate. The notebook includes a leakage check confirming that each team's first game sits exactly at the league value.

## Models compared

Both models use a season-based split: train on 2021-2022 (43,106 dropbacks) and test on 2023 (21,740 dropbacks). The split is never random, so future games never inform the past. Because accuracy is meaningless at a 6% base rate, models are graded on log loss, Brier score, and calibration.

1. **Logistic regression** (RobustScaler + LogisticRegression), the interpretable baseline
2. **Gradient boosting** (HistGradientBoostingClassifier with isotonic calibration), to test whether nonlinearity and interactions help

Both are compared against a naive baseline that always predicts the training base rate.

## Results (2023 test set)

| Model | Log loss | Brier | ROC AUC | PR AUC |
|---|---|---|---|---|
| Base rate | 0.2463 | 0.0626 | 0.500 | 0.067 |
| Logistic regression | 0.2421 | 0.0621 | 0.603 | 0.097 |
| Gradient boosting | 0.2414 | 0.0619 | 0.606 | 0.103 |

Key findings:

- **Both models beat the base rate on every metric and are well calibrated.** A predicted 8% sack probability happens about 8% of the time.
- **Gradient boosting barely improves on logistic regression.** A flexible model finding almost no extra signal suggests the ceiling comes from missing pre-snap information, not model choice. What most determines a sack (protection breakdowns, coverage, the QB holding the ball) can't be observed before the snap.
- **The strongest signals are pass expectation (`xpass`), down, and the QB prior.** Shotgun and no-huddle lower sack probability because the ball comes out faster.
- **The quarterback matters more than any situational factor.** Career sack rates among QBs with 300+ dropbacks range from 3.2% (Brady) to 12.3% (Fields), nearly a 4x spread.
- **Sack rate rises with down:** about 5% on first down and 9.5% on third. It peaks at 6-8 yards to go and falls as the offense's lead grows.

The notebook ends with a `predict_sack()` helper that scores any pre-snap situation. For example, Brady on 3rd and 4 at midfield, up 11 in the 4th quarter, gets 5.8%. Fields in the same spot gets 10.9%. Unseen quarterbacks fall back to the league rate.

## How to run

Requires Python 3.9+.

```bash
git clone <repo-url>
cd nfl-sack-probability
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook nfl_sack_probability.ipynb
```

Run all cells. The first run downloads about 3 seasons of play-by-play from nflverse into `data/`, and later runs reuse the cached files.

## Next steps

- Add PFR pressure and blitz-rate features
- Model pass vs. run upstream so low-pass-probability situations are handled explicitly
- Compare predicted probabilities against historical betting markets
