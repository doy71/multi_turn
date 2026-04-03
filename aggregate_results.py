from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--summary_jsonl", type=str, required=True)
    p.add_argument("--output_csv", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    rows = []
    with Path(args.summary_jsonl).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    grouped = (
        df.groupby("setting", as_index=False)
        .agg(
            accuracy=("correct", "mean"),
            mean_target_fact_total_mass=("target_fact_total_mass_mean", "mean"),
            mean_topk_hit_rate=("topk_hit_rate", "mean"),
            n=("sample_id", "count"),
        )
        .sort_values("setting")
    )
    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(out, index=False)
    print(grouped)


if __name__ == "__main__":
    main()
