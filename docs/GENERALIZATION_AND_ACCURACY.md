# Generalization, “99% accuracy,” and how this project evaluates models

This note is for students who see a strong number (for example **0.94** in the API playground) and want to push toward **99%**, while still understanding **overfitting**, **underfitting**, and what **unseen data** means.

---

## 1) The playground “score” is usually not “accuracy”

In the browser playground (`/ui`), the API returns:

- **`prediction`**: the model’s class label (0 or 1 for “not recommended” / “recommended”).
- **`score`**: the model’s estimated **probability of the positive class** (recommended), between 0 and 1.

So **0.9387** means: “the model assigns about **93.9%** probability to the positive class for *this one review*,” **not** “the model is 93.87% accurate on a dataset.”

**Accuracy** is computed only when you have many rows **with known labels**: you compare predicted classes to true labels and count the fraction that match. A single prediction has no “accuracy” by itself.

---

## 2) How this repository defines train vs “unseen” holdout data

Training in `src/train.py` does the following:

1. Load and clean the CSV (`src/data.py`).
2. **Stratified split** (`train_test_split` with `stratify=y`): **80% train** / **20% holdout** by default (`TEST_SIZE`, `RANDOM_STATE` in `src/config.py`).
3. **Cross-validation** (`--cv-f1`) runs **only on the training 80%**. The holdout **20% is not used** to fit the model or to choose fold splits.
4. After fitting on the full training 80%, metrics are reported **once** on the holdout 20%.

That holdout slice is the project’s stand-in for **“data the model did not train on.”** It is **not** the same as truly future production traffic, but it is the right habit: **one locked subset for final offline scoring**.

To make the holdout explicit on disk, run:

```powershell
make train-debug
```

(equivalent to `python -m src.train --model better --cv-f1 --fit-gap --save-holdout-indices`; use `make train-debug MODEL=advanced_xgb` for another tier.)

That writes a JSON file under `data/processed/holdout_manifests/` listing **`valid_row_indices`** (and train indices). With the **same** cleaned dataframe row order, seed, and `TEST_SIZE`, you can reproduce the same split.

---

## 3) Overfitting vs underfitting (the “too much / too little” idea)

| Pattern | Typical signal | What it suggests |
|--------|----------------|------------------|
| **Overfitting** | **Train** metrics much **better** than **holdout** (large gap on F1 or accuracy). | Model memorized noise or quirks of the training slice. Often: too complex, too little regularization, or subtle **leakage** (see below). |
| **Underfitting** | **Train and holdout** both **low**. | Model too simple, features too weak, or problem is genuinely hard. |

To see the gap in this project:

```powershell
make train-debug
```

(`train-debug` includes `--fit-gap`. For gap only without saving indices: `python -m src.train --model better --cv-f1 --fit-gap`.)

You get a **train vs holdout** table. Small gaps are normal; **large** gaps are a warning.

**Important:** if you **tune hyperparameters while looking at the holdout** over and over, the holdout stops being “unseen” in practice—you **overfit the validation set**. Prefer tuning with **CV on the training split** (already supported via `--cv-f1`) or nested CV for serious work.

---

## 4) Why chasing “99%” can mislead you

On many real problems—including text and ratings—**99% accuracy** is either:

- **Unrealistic** without more information or better labels, or
- A **red flag** for **data leakage** (accidentally letting the model see information that would not be available in production), or
- **Misleading** because **accuracy** is inflated when one class dominates (a naive “always predict majority” baseline can look very accurate while being useless).

This dataset is already in a range where strong baselines score high on F1/ROC-AUC (see `docs/STUDENT_ASSIGNMENTS_AND_BASELINES.md`). Pushing a few more points is possible with careful tuning, but **the teaching goal** is to pair any headline number with:

- **Holdout** performance (not just training),
- **Calibration** (do predicted probabilities match reality?—see log-loss in training output),
- **Business metrics** (precision/recall on the class you care about),

not to fixate on a round **99%** without context.

---

## 5) Practical checklist before you trust a high score

1. **Define the metric**: F1, ROC-AUC, precision@k, etc.—not a single probability from one API call.
2. **Confirm the split**: train vs holdout; never tune on the holdout.
3. **Run `--fit-gap`**: train vs holdout should not diverge wildly.
4. **Check class balance**: report F1 and confusion matrix, not only accuracy.
5. **Ask about leakage**: would this feature exist at prediction time in production?
6. **Fresh data**: when possible, evaluate on a **new time period** or **new sample** collected after training.

---

## 6) Where to read more in this repo

- `docs/STUDENT_ASSIGNMENTS_AND_BASELINES.md` — benchmark table and integrity rules.
- `docs/KPI_DEFINITIONS.md` — F1, ROC-AUC, precision, recall.
- `docs/TRANSFORMER_AND_HIGH_ACCURACY.md` — optional GPU transformer fine-tune and XGBoost tuning (`make train-transformer`, `make train-xgb-tune`).
- `docs/FAST_ITERATION_AND_SAMPLING.md` — stratified `--sample-frac` / `--max-rows`, `make train-quick`, and other speed tactics.
- `src/train.py` — `--cv-f1`, `--fit-gap`, `--save-holdout-indices`.

If you align your workflow with the split + CV discipline above, you avoid both **overclaiming** (thinking one probability is “accuracy”) and **cheating yourself** (overfitting the holdout while chasing 99%).
