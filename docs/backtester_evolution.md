# Backtester evolution — V1 → V6

Iterative log of betting-strategy experiments on `src/evaluation/backtester.py`.
Each section: hypothesis tested, what changed, headline result, what we kept /
discarded and why. Read top-to-bottom before proposing the next iteration —
the failed branches are at least as informative as the successful ones.

All runs share the same chronological split: last 20% of each league = test,
the rest = train. Flat 1u stake, 1.90 odds on both sides unless stated.

---

## V1 — Regression baseline

**Hypothesis.** If a CatBoostRegressor predicts `game_total` well, betting on
whichever side of `bookmaker_total_closing` the prediction falls will yield
positive ROI on volume.

**What.** Train CatBoostRegressor on all 48 features (incl. `bookmaker_total_
closing` and `market_vs_history_delta`); bet OVER if `pred > line` else UNDER.

**Result.** OVERALL 51.70% WR, **−1.77% ROI** on 1087 bets. EuroLeague spike
to 65% WR (later proved to be small-sample artefact).

**Kept.** Architecture template — load → features → chrono_split → train →
predict → simulate → aggregate. Stayed unchanged through all versions.

---

## V2 — Edge-threshold sweep on raw point delta

**Hypothesis.** Filtering to only confident picks (`|pred − line| ≥ edge`) will
lift winrate by trimming the dead zone around the line.

**What.** Loop simulation over `edges = [0.0, 1.5, 3.0, 4.5, 6.0]`.

**Result.** Mid-range edges (1.5–4.5) were **worse** than no filter. Only
`edge=6.0` produced +3.64% ROI / 54.55% WR — on 121 bets (too small).

**Discarded.** Point-delta edge as a decision rule. The regressor's point
deviation is not edge; it is mostly the regressor's own variance against a
sharp market. The model anchors on the line and treats nearby predictions as
"confident" when they are just noise.

---

## V3 — Reframe as classification (with line in features)

**Hypothesis.** Predicting `P(game_total > line)` directly (binary classifier)
will surface honest probability-edge instead of a derived sign-of-residual.

**What.** Swap to `CatBoostClassifier(Logloss / AUC)`. Bet OVER if
`prob ≥ T`, UNDER if `prob ≤ 1−T`. Threshold sweep `[0.50…0.60]`.

**Result.** Catastrophe. At T=0.50 OVERALL: **47.89% WR** (below 50%, below
the naive "always-OVER" baseline of 52%). NBA T=0.50: 47.93% WR. Higher
thresholds barely recovered to ~51%, never break-even.

**Discarded.** Trusting raw classifier probs when the line is a feature.

**Diagnosis.** Model fit noise around the line. With `bookmaker_total_closing`
as a feature, the classifier learned tiny patterns inside the rolling-stats
residual that the bookmaker already used to set the line. Those patterns
were spurious on test → systematically inverted picks.

---

## V4 — Independent consensus (line hidden from the model)

**Hypothesis.** Strip line-derived features so the classifier must form an
independent total estimate; compare it with the line only at bet time.

**What.** Exclude `BM_COLS = [bookmaker_total_closing, market_vs_history_delta]`
from `_get_x`. Keep `total_line` purely for target / filter / simulation.

**Result.** NBA T=0.50: **53.69% WR / +2.00% ROI** (+5.76 p.p. WR vs V3).
OVERALL break-even crossed at T ≥ 0.58. EuroLeague pattern looked noisy.

**Kept.** Hiding line-derived features from the model. This is now a
permanent invariant — `_EXCLUDED_FEATS` is a hard constraint, not an
experiment.

**Caveat.** Winrate curve was non-monotonic (drop at T=0.54–0.56). Pft on
high T came from few-but-confident picks. Bootstrap CI not yet computed.

---

## V5 — Isotonic calibration + Fractional Kelly

**Hypothesis.** Calibrating probabilities will smooth the WR curve and unlock
Kelly staking (which needs honest probabilities to be safe).

**What.** Manual 5-fold CV isotonic calibration on train (sklearn's
`CalibratedClassifierCV` fails to clone CatBoost with `cat_features=[...]`,
so calibration is hand-rolled with `IsotonicRegression`). Compounding
Fractional Kelly at 0.25 × full Kelly from a virtual 100u bankroll.

**Result.** Calibration collapsed the prob distribution: max **0.661**, mean
**0.524** (very close to the dataset OVER rate of 52.09% — a sign of honest
calibration). NBA T=0.50 WR fell to 52.30%; high-T cells emptied (T=0.60:
only 13 OVERALL bets). Kelly was strictly worse than Flat at every T (down to
−35% ROI) — predictable when true edge ≤ 0.

**Discarded for now.**
- Random k-fold calibration on chronologically-ordered data. The temporal
  dispersion between calibration folds and test era may explain the WR
  regression vs V4.
- Kelly until we actually have positive-edge probabilities. Variable sizing
  on a near-break-even classifier amplifies losses.

**Kept as future option.** Isotonic calibration *with a chronological holdout*
(last 20% of train by date as a single calibration set) — never tested.

**Diagnosis.** V4's high-T positive ROI was partly real edge + partly
overconfidence luck. Calibration correctly killed the luck but at the cost
of any tail signal. Honest verdict: the model has at most a small edge in
the 0.50–0.55 band.

---

## V6 — Significance test + per-league split (current ceiling)

**Hypothesis.** If there is real per-league edge (NBA, EuroLeague), a model
trained on a single league should surface it more cleanly than a global
model. Bootstrap CI on ROI tells us whether what we see is distinguishable
from zero.

**What.** Revert to V4 (raw probs, no calibration, Flat only). Add bootstrap
95% CI for ROI (5000 iterations per cell). Run three independent pipelines:
NBA-only, EuroLeague-only, OTHER-leagues control.

**Result.** **Not a single cell across NBA/EuroLeague/OTHER and any T has
CI95 lower bound above zero.**

| pipeline | T=0.50 WR | ROI | CI95 |
|---|---|---|---|
| NBA-only           | 50.69% | −3.69% | [−12.44%; +5.07%] |
| EuroLeague-only    | 46.25% | −12.13% | [−33.50%; +9.25%] |
| OTHER (control)    | 50.69% | −3.69% | [−11.57%; +4.20%] |

NBA-only **underperforms** V4 global on the same NBA test set (53.69% →
50.69% WR) — specialisation sacrificed cross-league data without finding
NBA-specific signal. EuroLeague at T ≥ 0.58 has CI95 *upper* bound below
zero ([−46.12%; −0.75%]): the model is statistically significant **anti-**
predictive there. The V1 "65% EuroLeague" was a small-sample mirage.

**Verdict.** The current 46-feature set has reached its information ceiling
on the closing line. We cannot distinguish our model's ROI from zero on
1000+ test bets. Further ML iteration on these features is shuffling noise.

---

## What we kept as permanent invariants

1. Hide all line-derived features from the model (`_EXCLUDED_FEATS = BM_COLS`).
2. Chronological per-league split (no shuffle, no leakage from future games).
3. Bootstrap CI on every ROI claim. No "+2% ROI" gets reported again without
   the interval.
4. Per-league test breakdown alongside OVERALL. Aggregates hide structure.

## Dead branches — do not revisit without new evidence

- Point-delta edge filtering on a regressor's output (V2). Mathematically
  not an edge.
- Including `bookmaker_total_closing` or any line-derived column in `X` (V3).
  Causes anchor-on-market collapse.
- Random k-fold isotonic calibration on chronologically-ordered data (V5).
  Try a chronological holdout calibrator instead, if revisited.
- Per-league CatBoost on NBA-only data (V6). 1700 rows is not enough; cross-
  league transfer learning beats specialisation on this dataset.

## Open paths (untried, ordered by expected ROI lift / cost)

1. **Late-scratch injury / lineup deltas.** Bookmakers lag by 30–60 min on
   late scratches; this is the largest known soft-market signal.
2. **Multi-book line consensus + reverse line movement.** Pinnacle/Circa
   moves first; off-shore books lag. RLM as a feature.
3. **Referee assignments.** NBA refs differ ±2–3 points on average total.
4. **Schedule fatigue features.** Back-to-back, time-zone delta, days until
   next game.

All four require new data pipelines, not new modelling tricks. Until at
least one of them is in the dataset, V6 is the documented ceiling.
