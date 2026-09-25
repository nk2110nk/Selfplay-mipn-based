#!/usr/bin/env python3
"""Summarize Alpha-Nego TSV files while preserving the shared result contract."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


METRICS = ("my_util", "opp_util1", "opp_util2", "social", "nash", "step")


def summarize(root):
    values = defaultdict(lambda: defaultdict(list))
    files = list(Path(root).rglob("*.tsv"))
    for path in files:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                for metric in METRICS:
                    values[str(path.parent)][metric].append(float(row[metric]))
                values[str(path.parent)]["agreement_rate"].append(float(float(row["my_util"]) != 0))
    return files, values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("summary.csv"))
    args = parser.parse_args()
    files, groups = summarize(args.data_dir)
    if not files:
        parser.error(f"No TSV files under {args.data_dir}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ("group", *METRICS, "agreement_rate", "rows")
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for group, metrics in sorted(groups.items()):
            writer.writerow({"group": group, **{
                metric: sum(items) / len(items) for metric, items in metrics.items()
            }, "rows": len(metrics["my_util"])})
    print(args.output)


if __name__ == "__main__":
    main()
