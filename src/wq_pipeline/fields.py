"""Step 2: discover all data fields available to this account.

Pulls /data-fields paginated, dumps full JSON + a flattened CSV.
Defaults: USA, delay=1, TOP3000 (broadest universe = superset of TOP200/500/1000).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from typing import Any

from dotenv import load_dotenv

from .client import WQClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("discover")

CSV_COLUMNS = (
    "id",
    "description",
    "dataset",
    "category",
    "subcategory",
    "type",
    "coverage",
    "userCount",
    "alphaCount",
    "themes",
)


def _flatten_value(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, dict):
        for k in ("id", "name"):
            if k in v and v[k] is not None:
                return str(v[k])
        return json.dumps(v, separators=(",", ":"))
    if isinstance(v, list):
        return "|".join(_flatten_value(x) for x in v)
    return str(v)


def to_csv_row(field: dict[str, Any]) -> dict[str, str]:
    return {col: _flatten_value(field.get(col)) for col in CSV_COLUMNS}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default="USA")
    parser.add_argument("--delay", type=int, default=1)
    parser.add_argument("--universe", default="TOP3000")
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--search", default=None)
    parser.add_argument("--out-csv", default="fields.csv")
    parser.add_argument("--out-json", default="fields.json")
    args = parser.parse_args(argv)

    load_dotenv()
    email = os.environ.get("WQ_EMAIL")
    password = os.environ.get("WQ_PASS")
    if not email or not password:
        print("ERROR: set WQ_EMAIL and WQ_PASS in .env", file=sys.stderr)
        return 1

    client = WQClient(email, password)
    logger.info(
        "fetch fields region=%s delay=%d universe=%s dataset=%s search=%s",
        args.region, args.delay, args.universe, args.dataset_id, args.search,
    )
    fields = client.get_data_fields(
        region=args.region,
        delay=args.delay,
        universe=args.universe,
        dataset_id=args.dataset_id,
        search=args.search,
    )

    if not fields:
        print("empty response — check tier permissions / params", file=sys.stderr)
        return 2

    sample_keys = sorted(fields[0].keys())
    print(f"\nsample row keys: {sample_keys}")
    print(f"\nsample row:\n{json.dumps(fields[0], indent=2)[:800]}\n")

    with open(args.out_json, "w") as fh:
        json.dump(fields, fh, indent=2)
    logger.info("wrote %d rows -> %s", len(fields), args.out_json)

    with open(args.out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for f in fields:
            writer.writerow(to_csv_row(f))
    logger.info("wrote %d rows -> %s", len(fields), args.out_csv)

    by_prefix: dict[str, int] = {}
    for f in fields:
        fid = str(f.get("id", ""))
        prefix = fid.split("_", 1)[0] if "_" in fid else fid[:6]
        by_prefix[prefix] = by_prefix.get(prefix, 0) + 1
    top = sorted(by_prefix.items(), key=lambda x: -x[1])[:20]
    print("\n--- top prefixes ---")
    for prefix, n in top:
        print(f"  {prefix:20s} {n:5d}")

    type_counts: dict[str, int] = {}
    for f in fields:
        t = _flatten_value(f.get("type"))
        type_counts[t] = type_counts.get(t, 0) + 1
    print("\n--- types ---")
    for t, n in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t:20s} {n:5d}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
