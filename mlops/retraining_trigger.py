"""Publish a retraining event (mocks AWS EventBridge)."""
from __future__ import annotations

import argparse
import json
import logging
import os

log = logging.getLogger("retraining_trigger")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def trigger_eventbridge(event: dict, bus: str = "ml-ops-bus") -> dict:
    """Publish to EventBridge if boto3+creds are available, else log only."""
    payload = {
        "Source": "demand_forecasting.drift",
        "DetailType": "RetrainingTriggered",
        "Detail": json.dumps(event),
        "EventBusName": bus,
    }
    if os.getenv("AWS_DEFAULT_REGION"):
        try:
            import boto3
            client = boto3.client("events")
            resp = client.put_events(Entries=[payload])
            log.info("Published EventBridge event: %s", resp)
            return resp
        except Exception as exc:
            log.warning("EventBridge publish failed (%s); logging only", exc)
    log.info("[mock] Would publish: %s", json.dumps(payload, indent=2))
    return {"mock": True, "payload": payload}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--drift-report", required=True)
    p.add_argument("--bus", default="ml-ops-bus")
    args = p.parse_args()

    report = json.loads(open(args.drift_report).read())
    if not report.get("trigger_retrain"):
        log.info("No drift detected; not triggering retrain.")
        return
    trigger_eventbridge(
        {"reason": "drift_detected", "drift_report": report},
        bus=args.bus,
    )


if __name__ == "__main__":
    main()
