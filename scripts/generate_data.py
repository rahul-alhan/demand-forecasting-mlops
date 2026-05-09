"""Generate synthetic SKU x store daily sales for demo."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate(n_skus: int, n_stores: int, n_days: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    end = pd.Timestamp("2026-05-01")
    dates = pd.date_range(end=end, periods=n_days, freq="D")

    rows = []
    for sku in range(n_skus):
        sku_base = rng.uniform(5, 80)
        sku_seasonality = rng.uniform(0.1, 0.4)
        category = f"cat_{sku % 6}"
        for store in range(n_stores):
            store_mult = rng.uniform(0.6, 1.6)
            for d in dates:
                dow = d.dayofweek
                weekly = 1 + 0.3 * np.sin(2 * np.pi * dow / 7)
                yearly = 1 + sku_seasonality * np.sin(2 * np.pi * d.dayofyear / 365)
                promo = 1.0 + 0.6 * (rng.random() < 0.05)
                noise = rng.normal(1.0, 0.15)
                qty = max(0, sku_base * store_mult * weekly * yearly * promo * noise)
                rows.append(
                    {
                        "date": d,
                        "sku_id": f"sku_{sku:04d}",
                        "store_id": f"store_{store:02d}",
                        "category": category,
                        "is_promo": int(promo > 1.0),
                        "qty": round(qty, 2),
                    }
                )

    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skus", type=int, default=20)
    p.add_argument("--stores", type=int, default=3)
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--out", default="data/raw.parquet")
    args = p.parse_args()

    df = generate(args.skus, args.stores, args.days)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"Wrote {len(df):,} rows to {args.out}")


if __name__ == "__main__":
    main()
