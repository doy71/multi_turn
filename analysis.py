#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd


SETTINGS_ORDER = [
    "packed_single_turn",
    "multi_turn_neutral",
    "multi_turn_dataset",
    "multi_turn_self_generated",
]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON parse error in {path} line {line_num}: {e}") from e
    return rows


def ensure_output_paths(run_dir: Path) -> Dict[str, Path]:
    summary_path = run_dir / "summary.jsonl"
    details_path = run_dir / "details.jsonl"
    analysis_dir = run_dir / "analysis"

    if not summary_path.exists():
        raise FileNotFoundError(f"summary.jsonl not found: {summary_path}")
    if not details_path.exists():
        print(f"[WARN] details.jsonl not found: {details_path}")
    analysis_dir.mkdir(parents=True, exist_ok=True)

    return {
        "summary": summary_path,
        "details": details_path,
        "analysis": analysis_dir,
    }


def analyze_summary(summary_rows: List[Dict[str, Any]], analysis_dir: Path) -> pd.DataFrame:
    df = pd.DataFrame(summary_rows)

    required_cols = [
        "sample_id",
        "setting",
        "prediction",
        "gold_answer",
        "correct",
        "target_fact_total_mass_mean",
        "topk_hit_rate",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"summary.jsonl missing required columns: {missing}")

    df["correct"] = df["correct"].astype(bool)

    # setting order
    df["setting"] = pd.Categorical(df["setting"], categories=SETTINGS_ORDER, ordered=True)
    df = df.sort_values(["sample_id", "setting"]).reset_index(drop=True)

    # 1) overall by setting
    by_setting = (
        df.groupby("setting", observed=True)
        .agg(
            n=("sample_id", "count"),
            accuracy=("correct", "mean"),
            fact_mass_mean=("target_fact_total_mass_mean", "mean"),
            fact_mass_std=("target_fact_total_mass_mean", "std"),
            topk_hit_rate_mean=("topk_hit_rate", "mean"),
            topk_hit_rate_std=("topk_hit_rate", "std"),
        )
        .reset_index()
    )

    # 2) correct vs incorrect within setting
    by_setting_correctness = (
        df.groupby(["setting", "correct"], observed=True)
        .agg(
            n=("sample_id", "count"),
            fact_mass_mean=("target_fact_total_mass_mean", "mean"),
            topk_hit_rate_mean=("topk_hit_rate", "mean"),
        )
        .reset_index()
    )

    # 3) pivot for sample-level comparison
    sample_pivot = df.pivot_table(
        index="sample_id",
        columns="setting",
        values=["correct", "target_fact_total_mass_mean", "topk_hit_rate"],
        aggfunc="first",
    )
    sample_pivot.columns = [
        f"{metric}__{setting}" for metric, setting in sample_pivot.columns.to_flat_index()
    ]
    sample_pivot = sample_pivot.reset_index()

    # 4) packed 대비 변화량
    packed_acc_col = "correct__packed_single_turn"
    packed_mass_col = "target_fact_total_mass_mean__packed_single_turn"
    packed_topk_col = "topk_hit_rate__packed_single_turn"

    for setting in SETTINGS_ORDER:
        if setting == "packed_single_turn":
            continue

        acc_col = f"correct__{setting}"
        mass_col = f"target_fact_total_mass_mean__{setting}"
        topk_col = f"topk_hit_rate__{setting}"

        if packed_acc_col in sample_pivot.columns and acc_col in sample_pivot.columns:
            sample_pivot[f"acc_drop_vs_packed__{setting}"] = (
                sample_pivot[packed_acc_col].astype(float) - sample_pivot[acc_col].astype(float)
            )

        if packed_mass_col in sample_pivot.columns and mass_col in sample_pivot.columns:
            sample_pivot[f"fact_mass_delta_vs_packed__{setting}"] = (
                sample_pivot[mass_col] - sample_pivot[packed_mass_col]
            )

        if packed_topk_col in sample_pivot.columns and topk_col in sample_pivot.columns:
            sample_pivot[f"topk_hit_delta_vs_packed__{setting}"] = (
                sample_pivot[topk_col] - sample_pivot[packed_topk_col]
            )

    # 5) packed correct였는데 multi-turn에서 틀린 샘플
    failure_rows = []
    for setting in SETTINGS_ORDER:
        if setting == "packed_single_turn":
            continue

        acc_col = f"correct__{setting}"
        if packed_acc_col in sample_pivot.columns and acc_col in sample_pivot.columns:
            failed = sample_pivot[
                (sample_pivot[packed_acc_col] == True) &
                (sample_pivot[acc_col] == False)
            ].copy()
            if not failed.empty:
                failed["degraded_setting"] = setting
                failure_rows.append(failed)

    degraded_cases = pd.concat(failure_rows, ignore_index=True) if failure_rows else pd.DataFrame()

    # save
    df.to_csv(analysis_dir / "summary_flat.csv", index=False)
    by_setting.to_csv(analysis_dir / "summary_by_setting.csv", index=False)
    by_setting_correctness.to_csv(analysis_dir / "summary_by_setting_correctness.csv", index=False)
    sample_pivot.to_csv(analysis_dir / "sample_level_comparison.csv", index=False)
    degraded_cases.to_csv(analysis_dir / "degraded_cases_from_packed.csv", index=False)

    print("\n=== Summary analysis ===")
    print(by_setting.to_string(index=False))

    return df


def extract_layer_head_fact_mass(details_rows: List[Dict[str, Any]]) -> pd.DataFrame:
    records = []

    for row in details_rows:
        sample_id = row.get("sample_id")
        setting = row.get("setting")
        correct = row.get("correct")
        metrics = row.get("metrics", {})
        layer_head_fact_mass = metrics.get("layer_head_fact_mass", [])

        for layer_idx, layer_heads in enumerate(layer_head_fact_mass):
            if not isinstance(layer_heads, list):
                continue

            for head_idx, head_dict in enumerate(layer_heads):
                if not isinstance(head_dict, dict):
                    continue

                # f1만 있는 구조라고 가정
                fact_keys = list(head_dict.keys())
                for fact_key in fact_keys:
                    value = head_dict[fact_key]
                    records.append(
                        {
                            "sample_id": sample_id,
                            "setting": setting,
                            "correct": bool(correct),
                            "layer": layer_idx,
                            "head": head_idx,
                            "fact_key": fact_key,
                            "fact_mass": float(value),
                        }
                    )

    return pd.DataFrame(records)


def analyze_details(details_rows: List[Dict[str, Any]], analysis_dir: Path) -> None:
    df_lh = extract_layer_head_fact_mass(details_rows)
    if df_lh.empty:
        print("[WARN] No layer_head_fact_mass records found in details.jsonl")
        return

    df_lh["setting"] = pd.Categorical(df_lh["setting"], categories=SETTINGS_ORDER, ordered=True)

    # 1) setting x layer x head 평균
    by_head = (
        df_lh.groupby(["setting", "layer", "head"], observed=True)
        .agg(
            n=("fact_mass", "count"),
            fact_mass_mean=("fact_mass", "mean"),
            fact_mass_std=("fact_mass", "std"),
        )
        .reset_index()
        .sort_values(["setting", "fact_mass_mean"], ascending=[True, False])
    )

    # 2) correct / incorrect 비교
    by_head_correctness = (
        df_lh.groupby(["setting", "correct", "layer", "head"], observed=True)
        .agg(
            n=("fact_mass", "count"),
            fact_mass_mean=("fact_mass", "mean"),
        )
        .reset_index()
    )

    # 3) layer 평균
    by_layer = (
        df_lh.groupby(["setting", "layer"], observed=True)
        .agg(
            n=("fact_mass", "count"),
            fact_mass_mean=("fact_mass", "mean"),
            fact_mass_std=("fact_mass", "std"),
        )
        .reset_index()
        .sort_values(["setting", "layer"])
    )

    # 4) 각 setting에서 상위 retrieval heads
    top_heads = (
        by_head.sort_values(["setting", "fact_mass_mean"], ascending=[True, False])
        .groupby("setting", observed=True)
        .head(20)
        .reset_index(drop=True)
    )

    # 5) packed 대비 각 head의 변화
    pivot = by_head.pivot_table(
        index=["layer", "head"],
        columns="setting",
        values="fact_mass_mean",
        aggfunc="first",
    ).reset_index()

    if "packed_single_turn" in pivot.columns:
        for setting in SETTINGS_ORDER:
            if setting == "packed_single_turn":
                continue
            if setting in pivot.columns:
                pivot[f"delta_vs_packed__{setting}"] = pivot[setting] - pivot["packed_single_turn"]

    # save
    df_lh.to_csv(analysis_dir / "layer_head_fact_mass_flat.csv", index=False)
    by_head.to_csv(analysis_dir / "layer_head_fact_mass_by_head.csv", index=False)
    by_head_correctness.to_csv(analysis_dir / "layer_head_fact_mass_by_head_correctness.csv", index=False)
    by_layer.to_csv(analysis_dir / "layer_head_fact_mass_by_layer.csv", index=False)
    top_heads.to_csv(analysis_dir / "top20_heads_per_setting.csv", index=False)
    pivot.to_csv(analysis_dir / "layer_head_fact_mass_pivot_vs_packed.csv", index=False)

    print("\n=== Top retrieval heads per setting ===")
    for setting in SETTINGS_ORDER:
        subset = top_heads[top_heads["setting"] == setting].head(10)
        if subset.empty:
            continue
        print(f"\n[{setting}]")
        print(subset[["layer", "head", "fact_mass_mean"]].to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True, help="Path containing summary.jsonl and details.jsonl")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    paths = ensure_output_paths(run_dir)

    summary_rows = load_jsonl(paths["summary"])
    analyze_summary(summary_rows, paths["analysis"])

    if paths["details"].exists():
        details_rows = load_jsonl(paths["details"])
        analyze_details(details_rows, paths["analysis"])

    print(f"\nSaved analysis files to: {paths['analysis']}")


if __name__ == "__main__":
    main()