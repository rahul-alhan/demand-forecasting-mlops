"""Build temporal, lag, rolling, event, and categorical features."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

LAGS = [1, 7, 14, 28]
ROLLINGS = [7, 14, 28]


def add_temporal(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["dow"] = df["date"].dt.dayofweek
    df["dom"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["woy"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    return df


def add_lag_rolling(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["sku_id", "store_id", "date"]).copy()
    g = df.groupby(["sku_id", "store_id"])["qty"]
    for lag in LAGS:
        df[f"lag_{lag}"] = g.shift(lag)
    for w in ROLLINGS:
        df[f"rmean_{w}"] = g.shift(1).rolling(w).mean().reset_index(level=[0, 1], drop=True)
        df[f"rstd_{w}"] = g.shift(1).rolling(w).std().reset_index(level=[0, 1], drop=True)
    return df


def add_target_encoding(df: pd.DataFrame, train_mask: pd.Series) -> pd.DataFrame:
    df = df.copy()
    cat_mean = df.loc[train_mask].groupby("category")["qty"].mean()
    df["category_te"] = df["category"].map(cat_mean)
    return df


def add_cold_start_fallback(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    counts = df.groupby(["sku_id", "store_id"]).cumcount()
    df["is_cold_start"] = (counts < 28).astype(int)
    df["lag_7"] = np.where(df["lag_7"].isna(), df["category_te"], df["lag_7"])
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_temporal(df)
    df = add_lag_rolling(df)
    cutoff = df["date"].quantile(0.8)
    train_mask = df["date"] < cutoff
    df = add_target_encoding(df, train_mask)
    df = add_cold_start_fallback(df)
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    raw = pd.read_parquet(args.inp)
    feats = build_features(raw)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(args.out, index=False)
    print(f"Wrote {len(feats):,} feature rows ({feats.shape[1]} cols) → {args.out}")


if __name__ == "__main__":
    main()
