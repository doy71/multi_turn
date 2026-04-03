from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.conversation_builders import SETTING_NAMES, build_messages
from src.dataset_io import dump_jsonl, load_samples
from src.metrics import recover_fact_token_spans
from src.modeling import HFChatModel
from src.runner import run_one_setting

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", type=str, default=None)
    p.add_argument("--input_jsonl", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument(
        "--settings",
        nargs="+",
        default=[
            "packed_single_turn",
            "multi_turn_neutral",
            "multi_turn_dataset",
            "multi_turn_self_generated",
        ],
    )
    p.add_argument("--assistant_temperature", type=float, default=0.0)
    p.add_argument("--answer_temperature", type=float, default=0.0)
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--capture_last_k_steps", type=int, default=1)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--torch_dtype", type=str, default="bfloat16")
    p.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    p.add_argument(
        "--self_test",
        action="store_true",
        help="Validate dataset + setting construction + fact span recovery without loading a model.",
    )
    p.add_argument(
        "--self_test_num_samples",
        type=int,
        default=5,
        help="How many samples to inspect in --self_test mode.",
    )
    return p.parse_args()


def _fake_render_and_recover(sample, messages, setting: str):
    """Tokenizer-free span recovery check used by --self_test."""
    cursor = 0
    full_text = ""
    message_char_spans = []
    for i, m in enumerate(messages):
        prefix = f"<msg{i}:{m['role']}>"
        full_text += prefix + m["content"]
        start = cursor + len(prefix)
        end = start + len(m["content"])
        message_char_spans.append(
            {
                "message_index": i,
                "role": m["role"],
                "content": m["content"],
                "char_start": start,
                "char_end": end,
            }
        )
        cursor = end
    offset_mapping = [(i, i + 1) for i in range(len(full_text))]
    return recover_fact_token_spans(
        sample=sample,
        messages=messages,
        message_char_spans=message_char_spans,
        offset_mapping=offset_mapping,
        setting=setting,
    )


def run_self_test(samples, settings, max_samples: int):
    checked = 0
    failures = []
    for sample in samples[:max_samples]:
        for setting in settings:
            messages = build_messages(sample, setting)
            if not messages:
                failures.append((sample.sample_id, setting, "no_messages"))
                continue
            if messages[-1]["role"] != "user":
                failures.append((sample.sample_id, setting, "last_message_not_user"))
                continue
            fact_spans = _fake_render_and_recover(sample, messages, setting)
            missing = [f.fact_id for f in sample.facts if f.fact_id not in fact_spans]
            if missing:
                failures.append((sample.sample_id, setting, f"missing_fact_spans:{missing}"))
            checked += 1

    logger.info("Self-test checked %d (sample, setting) pairs", checked)
    if failures:
        logger.error("Self-test found %d failure(s)", len(failures))
        for sample_id, setting, reason in failures[:20]:
            logger.error("sample=%s setting=%s reason=%s", sample_id, setting, reason)
        raise SystemExit(1)
    logger.info("Self-test passed")


def main():
    args = parse_args()

    unknown = set(args.settings) - SETTING_NAMES
    if unknown:
        raise ValueError(
            f"Unknown setting(s): {unknown}. Choose from {sorted(SETTING_NAMES)}"
        )

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(args.input_jsonl)
    if args.limit is not None:
        samples = samples[: args.limit]
    logger.info("Loaded %d samples (limit=%s)", len(samples), args.limit)

    if args.self_test:
        run_self_test(samples, args.settings, args.self_test_num_samples)
        return

    if not args.model_name_or_path:
        raise ValueError("--model_name_or_path is required unless --self_test is used")

    model = HFChatModel(
        args.model_name_or_path,
        torch_dtype=args.torch_dtype,
        attn_implementation=(None if args.attn_implementation.lower() == "none" else args.attn_implementation),
    )

    all_rows = []
    summary_rows = []
    for sample in samples:
        for setting in args.settings:
            row = run_one_setting(
                model=model,
                sample=sample,
                setting=setting,
                assistant_temperature=args.assistant_temperature,
                answer_temperature=args.answer_temperature,
                max_new_tokens=args.max_new_tokens,
                capture_last_k_steps=args.capture_last_k_steps,
            )
            all_rows.append(row)
            summary_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "setting": row["setting"],
                    "prediction": row["prediction"],
                    "gold_answer": row["gold_answer"],
                    "correct": row["correct"],
                    "target_fact_total_mass_mean": row["metrics"]["target_fact_total_mass_mean"],
                    "topk_hit_rate": row["metrics"]["topk_hit_rate"],
                }
            )
            print(json.dumps(summary_rows[-1], ensure_ascii=False))

    dump_jsonl(summary_rows, outdir / "summary.jsonl")
    dump_jsonl(all_rows, outdir / "details.jsonl")


if __name__ == "__main__":
    main()
