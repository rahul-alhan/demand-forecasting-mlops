"""KS-test based feature drift detection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy.stats import ks_2samp

DEFAULT_FEATURES = [
    "qty", "lag_7", "rmean_7", "rmean_28", "category_te",
]
DRIFT_THRESHOLD = 0.15


def detect(reference: pd.DataFrame, current: pd.DataFrame, features=None) -> dict:
    features = features or DEFAULT_FEATURES
    results = {}
    for f in features:
        if f not in reference or f not in current:
            continue
        ref = reference[f].dropna()
        cur = current[f].dropna()
        if len(ref) < 50 or len(cur) < 50:
            continue
        stat, _ = ks_2samp(ref, cur)
        results[f] = {"ks_stat": float(stat), "drifted": bool(stat > DRIFT_THRESHOLD)}
    n_drifted = sum(1 for r in results.values() if r["drifted"])
    return {
        "threshold": DRIFT_THRESHOLD,
        "n_features_checked": len(results),
        "n_drifted": n_drifted,
        "features": results,
        "trigger_retrain": n_drifted > 0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", required=True)
    p.add_argument("--current", required=True)
    p.add_argument("--out", default="reports/drift.json")
    args = p.parse_args()

    ref = pd.read_parquet(args.reference)
    cur = pd.read_parquet(args.current)
    report = detect(ref, cur)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
