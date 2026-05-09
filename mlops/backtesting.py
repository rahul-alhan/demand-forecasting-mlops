"""Walk-forward backtesting with WAPE / bias / coverage metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from features.feature_engineering import build_features
from models.lightgbm_quantile import CAT_COLS, NUMERIC_COLS, QUANTILES


def wape(y_true, y_pred):
    return float(np.sum(np.abs(y_true - y_pred)) / max(1e-9, np.sum(np.abs(y_true))))


def bias(y_true, y_pred):
    return float(np.mean(y_pred - y_true) / max(1e-9, np.mean(y_true)))


def coverage(y_true, p10, p90):
    inside = (y_true >= p10) & (y_true <= p90)
    return float(np.mean(inside))


def load_models(artifact_dir: str):
    return {
        q: lgb.Booster(model_file=str(Path(artifact_dir) / f"lgb_q{int(q*100)}.txt"))
        for q in QUANTILES
    }


def backtest(df_features: pd.DataFrame, artifact_dir: str) -> pd.DataFrame:
    df = df_features.dropna(subset=["lag_28", "rmean_28"]).copy()
    for c in CAT_COLS:
        df[c] = df[c].astype("category")

    cutoff = df["date"].quantile(0.8)
    test = df[df["date"] >= cutoff].copy()

    models = load_models(artifact_dir)
    Xt = test[NUMERIC_COLS + CAT_COLS]

    test["pred_p10"] = models[0.1].predict(Xt)
    test["pred_p50"] = models[0.5].predict(Xt)
    test["pred_p90"] = models[0.9].predict(Xt)

    rows = []
    for (sku, store), g in test.groupby(["sku_id", "store_id"], observed=True):
        rows.append(
            {
                "sku_id": sku,
                "store_id": store,
                "n": len(g),
                "wape": wape(g["qty"], g["pred_p50"]),
                "bias": bias(g["qty"], g["pred_p50"]),
                "coverage_80": coverage(g["qty"], g["pred_p10"], g["pred_p90"]),
            }
        )
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True)
    p.add_argument("--artifacts", default="models/artifacts")
    p.add_argument("--report", default="reports")
    args = p.parse_args()

    feats = pd.read_parquet(args.features)
    report = backtest(feats, args.artifacts)

    out_dir = Path(args.report)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "backtest_per_pair.csv"
    report.to_csv(csv_path, index=False)

    summary = {
        "n_pairs": int(len(report)),
        "wape_p50_mean": float(report["wape"].mean()),
        "wape_p50_median": float(report["wape"].median()),
        "bias_mean": float(report["bias"].mean()),
        "coverage_80_mean": float(report["coverage_80"].mean()),
    }
    (out_dir / "backtest_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nPer-pair report → {csv_path}")


if __name__ == "__main__":
    main()
