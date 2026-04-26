# LLMs vs classical ML for this project (MVP at scale)

Students often ask: *Why not use an LLM to get the best log-loss / best model?*

## What this codebase optimizes for

- **Cheap batch scoring** at high volume (microseconds to milliseconds per row on CPU/GPU boosters).
- **Reproducible training** without vendor API keys in the critical path.
- **Interpretable baselines** (TF-IDF + logistic regression) before exotic models.
- **Teaching path** from data quality → features → metrics → deployment.

## When classical ML (TF-IDF + linear / tree / boosting) wins

- Tabular + short text classification on a **fixed schema** (this dataset).
- Need **stable cost per prediction** and predictable latency.
- Need **offline training** on full data without per-row API spend.
- Need **simple compliance story** (data stays on your infra).

For many product MVPs, **XGBoost/LightGBM + TF-IDF** or **linear models** are the right first production tier.

## When LLMs help

- **Zero-shot or few-shot** when you have almost no labels.
- **Open-ended text** tasks (summarization, extraction, multi-intent).
- **Semantic search** or retrieval-augmented flows.

## Cost and “cheaper LLMs”

API pricing can be low per token, but **total cost = price × tokens × traffic**. At scale, a 1-line review still adds prompt overhead, caching complexity, and failure modes (rate limits, drift in model updates).

## Log-loss note

Log-loss (binary cross-entropy) is a fine metric, but **business metrics** (precision/recall/F1 at an operating threshold) usually matter more for recommendations. You can add log-loss to `src/train.py` if you want to align with probabilistic ranking.

## Suggested “scale MVP” path

1. Ship **batch + API** inference with the current sklearn/XGBoost stack (`src/inference.py`, `src/api.py`).
2. Add **embedding + linear head** or a **small fine-tuned classifier** if TF-IDF plateaus.
3. Add **LLM only** for auxiliary tasks (e.g., topic tags), not as the default scorer for every row.

This keeps the system teachable, fast, and cost-predictable while leaving room to grow.
