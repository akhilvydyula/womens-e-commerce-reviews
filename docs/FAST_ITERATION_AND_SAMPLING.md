# Fast iteration when data or training feels slow

Students often hit two different problems:

1. **Pandas feels slow** (big CSV in memory, many transforms).
2. **Model training feels slow** (TF-IDF + boosting, transformers, many CV folds).

This project supports a **teaching-friendly** path: learn the *crux* on a **stratified sample**, then run **full data** for real scores.

---

## 1) Stratified sampling (built into training)

After cleaning, you can shrink the dataframe **while keeping both classes** (when possible):

```bash
python -m src.train --model better --sample-frac 0.25 --cv-f1
python -m src.train --model better --max-rows 5000 --cv-f1
```

If you pass **both**, `--max-rows` wins.

The same flags work for **`python -m src.train_transformer`** (quick HF smoke runs).

**Caveat:** metrics on a sample are **directional only**. Your “real” benchmark is still **full data** (or a fixed holdout manifest) so you do not fool yourself.

---

## 2) Other standard techniques (no new code required)

| Technique | What it buys you |
|-----------|------------------|
| **Fewer CV folds** | `--cv-splits 2` instead of 5 |
| **Skip CV while debugging** | Omit `--cv-f1` until the pipeline works |
| **Smaller text budget** | In `src/features.py`, lower TF-IDF `max_features`, shorter `ngram_range` |
| **Simpler model first** | `baseline` → `better` before `advanced_xgb` |
| **Reuse trained models** | `make leaderboard-fast` ranks existing `*.joblib` without retraining |
| **ETL once** | `make etl` then `--data-path data/processed/clean_reviews.csv` for consistent inputs |
| **Chunked / out-of-core** | For truly huge data: Dask, Polars lazy scan, or incremental learning (advanced topic) |

---

## 3) Mental model for class

1. **Prototype** on a sample (fast loops, broken code is cheap).  
2. **Verify** on full data with the same `RANDOM_STATE` discipline.  
3. **Report** holdout + (optional) CV on the full training split for the final write-up.

For why probabilities in the API are not “accuracy,” see `docs/GENERALIZATION_AND_ACCURACY.md`.
