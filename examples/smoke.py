"""Smoke test: ts_mean(fnd6_pnrsho, 10) on TOP1000, MARKET neut, decay=1, trunc=0.01."""
from __future__ import annotations

import json
import logging
import os
import sys

from dotenv import load_dotenv

from wq_pipeline.client import SimulationSettings, WQClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def main() -> int:
    load_dotenv()
    email = os.environ.get("WQ_EMAIL")
    password = os.environ.get("WQ_PASS")
    if not email or not password:
        print("ERROR: set WQ_EMAIL and WQ_PASS in .env (copy from .env.example)", file=sys.stderr)
        return 1

    client = WQClient(email, password)

    expression = "ts_mean(fnd6_pnrsho, 10)"
    settings = SimulationSettings(
        region="USA",
        universe="TOP1000",
        delay=1,
        decay=1,
        neutralization="MARKET",
        truncation=0.01,
        pasteurization="ON",
        unit_handling="VERIFY",
        nan_handling="OFF",
        language="FASTEXPR",
    )

    print(f"\n=== submit ===\nexpr: {expression}\nsettings: {settings}\n")
    result = client.run_simulation(expression, settings)

    print(f"location: {result['location']}")
    print(f"alpha_id: {result['alpha_id']}")
    print(f"sim status: {result['simulation'].get('status')}")

    alpha = result["alpha"]
    if not alpha:
        print("\nNo alpha returned. Full sim response:")
        print(json.dumps(result["simulation"], indent=2)[:2000])
        return 2

    is_block = alpha.get("is") or {}
    print("\n--- IS metrics ---")
    for k in (
        "sharpe",
        "fitness",
        "turnover",
        "returns",
        "drawdown",
        "margin",
        "longCount",
        "shortCount",
        "pnl",
        "bookSize",
    ):
        v = is_block.get(k)
        print(f"  {k:12s}: {v}")

    print("\n--- IS checks ---")
    for c in is_block.get("checks", []):
        name = c.get("name", "?")
        result_flag = c.get("result", "?")
        value = c.get("value")
        limit = c.get("limit")
        print(f"  {name:25s} {result_flag:6s}  value={value}  limit={limit}")

    print(f"\nfull alpha id (use to inspect on platform): {result['alpha_id']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
