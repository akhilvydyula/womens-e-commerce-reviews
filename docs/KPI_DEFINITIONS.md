# KPI Definitions (Analytics + Model)

Use these KPIs to connect model work with business outcomes.

## Business-facing KPIs

- `recommendation_rate`: share of items predicted/reported as recommended
- `high_rating_rate`: share of reviews with rating >= 4
- `positive_feedback_p90`: 90th percentile of positive feedback count
- `segment_recommendation_rate`: recommendation rate by department/class

## Model KPIs

- `f1`: harmonic mean of precision and recall (primary)
- `roc_auc`: ranking quality for positive class
- `precision_positive`: trust in positive predictions
- `recall_positive`: coverage of positives

## Reliability KPIs

- `data_rows`: input row count per run
- `null_rate_top_columns`: null trend over time
- `inference_latency_batch_s`: batch run duration
- `quality_gate_pass_rate`: % runs that pass validation

## Suggested release gate

- `f1 >= previous_f1 - 0.01`
- `quality_gate_passed == true`
- no critical test failures
