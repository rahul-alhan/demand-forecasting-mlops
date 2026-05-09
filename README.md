# Demand Forecasting MLOps

End-to-end **quantile forecasting** pipeline with deep feature engineering, **drift detection**, **automated retraining triggers**, and **WAPE/bias backtesting** — designed for retail SKU-store granularity.

> Mirrors the production system at Limendo: 900+ SKUs × 51 stores, ~15% accuracy improvement, drift-triggered retraining via AWS EventBridge.

---

## Pipeline

```
   raw sales ─┐
              │
   weather  ──┤───▶  features/feature_engineering.py
              │            (lag, rolling, calendar, event)
   events   ──┘                        │
                                       ▼
                          ┌────────────────────────────┐
                          │  models/lightgbm_quantile  │
                          │  (q=[0.1, 0.5, 0.9])       │
                          └────────────┬───────────────┘
                                       │
              ┌────────────────────────┼─────────────────────────┐
              ▼                        ▼                         ▼
   mlops/backtesting.py     mlops/drift_detection.py    mlops/retraining_trigger.py
   (WAPE, MAE, bias)        (KS test on features)       (EventBridge mock)
```

---

## Quickstart

```bash
pip install -r requirements.txt

# 1. Generate synthetic SKU x store data
python -m scripts.generate_data --skus 50 --stores 5 --days 730

# 2. Build features
python -m features.feature_engineering --in data/raw.parquet --out data/features.parquet

# 3. Train quantile models with MLflow tracking
python -m models.lightgbm_quantile --features data/features.parquet

# 4. Backtest (WAPE / bias by SKU-store)
python -m mlops.backtesting --features data/features.parquet --report reports/

# 5. Check for drift
python -m mlops.drift_detection \
  --reference data/features.parquet \
  --current data/features_recent.parquet
```

---

## Feature Engineering

| Family | Features |
|---|---|
| **Temporal** | day_of_week, day_of_month, week_of_year, month, is_weekend, is_month_start/end |
| **Lag** | lag_1, lag_7, lag_14, lag_28 |
| **Rolling** | rolling_mean_7/14/28, rolling_std_7/28 |
| **Event** | is_promo, is_holiday, days_to_next_holiday |
| **Categorical** | sku_id, store_id, category, region (target-encoded) |
| **Cold-start fallback** | category-level mean for new SKUs (< 28 days history) |

---

## Quantile Forecasting

Three LightGBM models trained at quantiles **[0.1, 0.5, 0.9]** — gives a calibrated prediction interval for inventory planning, not just a point forecast.

```python
preds = {
    "p10": model_q10.predict(X),    # safety stock floor
    "p50": model_q50.predict(X),    # central forecast
    "p90": model_q90.predict(X),    # over-stock cap
}
```

A **residual bias correction** layer fits a per-SKU adjustment on validation residuals — corrects systematic over/under-prediction.

---

## Drift Detection

Per-feature **Kolmogorov-Smirnov** test against the training reference distribution.
A drift score above `0.15` (configurable) on any production feature emits a retraining event.

In production, `mlops/retraining_trigger.py` publishes the event to **AWS EventBridge** —
a SageMaker pipeline picks it up and retrains the affected SKU-store group.

---

## Backtesting Metrics

- **WAPE** (weighted absolute percentage error) — default headline metric
- **Bias** = mean(pred - actual) / mean(actual)
- **Pinball loss** at each quantile (calibration check)
- **Coverage** — % of actuals inside [p10, p90]

Reports written per (SKU, store) so issues are debuggable.

---

## Repository Layout

```
demand-forecasting-mlops/
├── README.md
├── requirements.txt
├── scripts/
│   └── generate_data.py             # synthetic data for demo
├── features/
│   ├── __init__.py
│   └── feature_engineering.py
├── models/
│   ├── __init__.py
│   └── lightgbm_quantile.py
├── mlops/
│   ├── __init__.py
│   ├── drift_detection.py
│   ├── retraining_trigger.py
│   └── backtesting.py
├── data/                            # gitignored
└── reports/                         # gitignored
```

---

## Production Notes

| Concern | Production solution |
|---|---|
| Storage | S3 (parquet, partitioned by date) |
| Compute | SageMaker Training Jobs (GPU not needed; xgboost/lightgbm CPU) |
| Tracking | MLflow on Databricks (model registry + experiment compare) |
| Retraining | EventBridge → SageMaker Pipelines, gated by WAPE delta |
| Serving | Batch nightly → DynamoDB lookup table for warehouse system |
| Monitoring | CloudWatch dashboards: WAPE / bias / coverage by region |

---

## License

MIT
