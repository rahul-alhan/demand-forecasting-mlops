"""Regression test: rolling features must not leak across (sku, store) groups."""
from __future__ import annotations

import numpy as np
import pandas as pd

from features.feature_engineering import add_lag_rolling, add_temporal


def _toy_frame() -> pd.DataFrame:
    rows = []
    for sku in ("A", "B"):
        for store in ("s1",):
            for i, q in enumerate([100, 200, 300, 400, 500, 600, 700, 800]):
                rows.append({
                    "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
                    "sku_id": sku,
                    "store_id": store,
                    "qty": q,
                })
    return pd.DataFrame(rows)


def test_add_lag_rolling_runs_without_error():
    df = add_temporal(_toy_frame())
    out = add_lag_rolling(df)
    assert "rmean_7" in out.columns
    assert "lag_7" in out.columns
    assert len(out) == len(df)


def test_rolling_does_not_leak_across_groups():
    """The first rolling value of group B must NOT incorporate group A's history."""
    df = add_temporal(_toy_frame())
    out = add_lag_rolling(df)
    # for sku=B, first 7 rmean_7 values should be NaN (no full window yet within B)
    b = out[out["sku_id"] == "B"].sort_values("date").reset_index(drop=True)
    assert b.loc[0, "rmean_7"] != b.loc[0, "rmean_7"] or np.isnan(b.loc[0, "rmean_7"])
    # specifically: position 7 of B should equal mean of B's first 7 qty values
    # (B values are also 100..800; the shift(1) means rmean at idx 7 uses positions 0..6)
    expected = np.mean([100, 200, 300, 400, 500, 600, 700])
    assert abs(float(b.loc[7, "rmean_7"]) - expected) < 1e-9


def test_lag_does_not_leak_across_groups():
    df = add_temporal(_toy_frame())
    out = add_lag_rolling(df)
    b = out[out["sku_id"] == "B"].sort_values("date").reset_index(drop=True)
    # lag_7 at position 7 of B must equal B's value at position 0, not A's
    assert float(b.loc[7, "lag_7"]) == 100.0
