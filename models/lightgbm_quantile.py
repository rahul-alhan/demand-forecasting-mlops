"""Train LightGBM quantile models with MLflow tracking."""
from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd

QUANTILES = [0.1, 0.5, 0.9]
NUMERIC_COLS = [
    "dow", "dom", "month", "woy", "is_weekend",
    "is_month_start", "is_month_end", "is_promo",
    "lag_1", "lag_7", "lag_14", "lag_28",
    "rmean_7", "rmean_14", "rmean_28",
    "rstd_7", "rstd_28", "category_te", "is_cold_start",
]
CAT_COLS = ["sku_id", "store_id", "category"]


def _prepare(df: pd.DataFrame):
    df = df.dropna(subset=["lag_28", "rmean_28"]).copy()
    for c in CAT_COLS:
        df[c] = df[c].astype("category")
    X = df[NUMERIC_COLS + CAT_COLS]
    y = df["qty"]
    return X, y, df["date"]


def _split(X, y, dates, valid_frac=0.2):
    cutoff = dates.quantile(1 - valid_frac)
    train = dates < cutoff
    return X[train], y[train], X[~train], y[~train]


def train_one(q: float, Xt, yt, Xv, yv) -> lgb.Booster:
    params = {
        "objective": "quantile",
        "alpha": q,
        "metric": "quantile",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
        "verbose": -1,
    }
    dtrain = lgb.Dataset(Xt, label=yt, categorical_feature=CAT_COLS)
    dvalid = lgb.Dataset(Xv, label=yv, categorical_feature=CAT_COLS, reference=dtrain)
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=400,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)],
    )
    return booster


def pinball_loss(y_true, y_pred, q):
    diff = y_true - y_pred
    return np.mean(np.maximum(q * diff, (q - 1) * diff))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True)
    p.add_argument("--mlflow-uri", default="file:./mlruns")
    p.add_argument("--out-dir", default="models/artifacts")
    args = p.parse_args()

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment("demand_forecasting")

    df = pd.read_parquet(args.features)
    X, y, dates = _prepare(df)
    Xt, yt, Xv, yv = _split(X, y, dates)
    print(f"Train rows: {len(Xt):,}  Valid rows: {len(Xv):,}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run():
        mlflow.log_params({"n_train": len(Xt), "n_valid": len(Xv)})
        for q in QUANTILES:
            print(f"\n=== Training q={q} ===")
            booster = train_one(q, Xt, yt, Xv, yv)
            preds = booster.predict(Xv)
            loss = pinball_loss(yv.to_numpy(), preds, q)
            mlflow.log_metric(f"pinball_q{int(q*100)}", loss)
            booster.save_model(str(out / f"lgb_q{int(q*100)}.txt"))
            print(f"  Pinball loss: {loss:.4f}")

        print(f"\nArtifacts written to {out}")


if __name__ == "__main__":
    main()
