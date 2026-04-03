from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path
from tempfile import NamedTemporaryFile

from src.conversation_builders import SETTING_NAMES, build_messages
from src.dataset_io import dump_jsonl, load_samples
from src.metrics import recover_fact_token_spans
from src.modeling import HFChatModel
from src.runner import run_one_setting

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

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
    p.add_argument(
        "--save_every_seconds",
        type=int,
        default=3600,
        help="Periodically save partial results every N seconds. Set <=0 to disable time-based checkpointing.",
    )
    p.add_argument(
        "--save_every_steps",
        type=int,
        default=50,
        help="Also save partial results every N completed (sample, setting) steps. Set <=0 to disable step-based checkpointing.",
    )
    p.add_argument(
        "--no_progress_bar",
        action="store_true",
        help="Disable tqdm progress bar.",
    )
    return p.parse_args()


def _atomic_write_jsonl(rows, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(path.parent), suffix=".tmp") as tmp:
        tmp_path = Path(tmp.name)
        for row in rows:
            tmp.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def _atomic_write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(path.parent), suffix=".tmp") as tmp:
        tmp_path = Path(tmp.name)
        json.dump(obj, tmp, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def _format_seconds(seconds: float | None) -> str:
    if seconds is None or math.isinf(seconds) or math.isnan(seconds):
        return "unknown"
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:d}h {m:02d}m {s:02d}s"
    return f"{m:d}m {s:02d}s"


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


def _save_partial_results(
    outdir: Path,
    summary_rows: list[dict],
    all_rows: list[dict],
    completed_steps: int,
    total_steps: int,
    start_time: float,
    reason: str,
) -> None:
    elapsed = time.time() - start_time
    avg_sec = elapsed / completed_steps if completed_steps > 0 else None
    eta_sec = (total_steps - completed_steps) * avg_sec if avg_sec is not None else None

    _atomic_write_jsonl(summary_rows, outdir / "summary.partial.jsonl")
    _atomic_write_jsonl(all_rows, outdir / "details.partial.jsonl")
    _atomic_write_json(
        {
            "completed_steps": completed_steps,
            "total_steps": total_steps,
            "progress": (completed_steps / total_steps) if total_steps else 0.0,
            "elapsed_seconds": elapsed,
            "avg_seconds_per_step": avg_sec,
            "eta_seconds": eta_sec,
            "elapsed_human": _format_seconds(elapsed),
            "eta_human": _format_seconds(eta_sec),
            "last_save_reason": reason,
            "saved_at_unix": time.time(),
        },
        outdir / "progress.json",
    )
    logger.info(
        "Saved partial results (%s): %d/%d steps complete, elapsed=%s, ETA=%s",
        reason,
        completed_steps,
        total_steps,
        _format_seconds(elapsed),
        _format_seconds(eta_sec),
    )


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

    total_steps = len(samples) * len(args.settings)
    logger.info("Planned run size: %d samples x %d settings = %d total steps", len(samples), len(args.settings), total_steps)
    logger.info(
        "Checkpointing: every %s seconds, every %s steps",
        args.save_every_seconds if args.save_every_seconds > 0 else "disabled",
        args.save_every_steps if args.save_every_steps > 0 else "disabled",
    )

    model = HFChatModel(
        args.model_name_or_path,
        torch_dtype=args.torch_dtype,
        attn_implementation=(None if args.attn_implementation.lower() == "none" else args.attn_implementation),
    )

    all_rows = []
    summary_rows = []
    completed_steps = 0
    start_time = time.time()
    last_save_time = start_time

    use_pbar = (not args.no_progress_bar) and (tqdm is not None)
    pbar = tqdm(total=total_steps, desc="Running experiments", dynamic_ncols=True) if use_pbar else None

    try:
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

                completed_steps += 1
                elapsed = time.time() - start_time
                avg_sec = elapsed / completed_steps
                eta_sec = (total_steps - completed_steps) * avg_sec

                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix(
                        elapsed=_format_seconds(elapsed),
                        eta=_format_seconds(eta_sec),
                        avg_s=f"{avg_sec:.2f}",
                    )
                elif completed_steps == 1 or completed_steps % 10 == 0:
                    logger.info(
                        "Progress %d/%d | elapsed=%s | ETA=%s | avg=%.2fs/step",
                        completed_steps,
                        total_steps,
                        _format_seconds(elapsed),
                        _format_seconds(eta_sec),
                        avg_sec,
                    )

                now = time.time()
                due_to_time = args.save_every_seconds > 0 and (now - last_save_time) >= args.save_every_seconds
                due_to_steps = args.save_every_steps > 0 and completed_steps % args.save_every_steps == 0
                if due_to_time or due_to_steps:
                    reason = "timer" if due_to_time else "step_interval"
                    _save_partial_results(
                        outdir=outdir,
                        summary_rows=summary_rows,
                        all_rows=all_rows,
                        completed_steps=completed_steps,
                        total_steps=total_steps,
                        start_time=start_time,
                        reason=reason,
                    )
                    last_save_time = now
    finally:
        if pbar is not None:
            pbar.close()

    dump_jsonl(summary_rows, outdir / "summary.jsonl")
    dump_jsonl(all_rows, outdir / "details.jsonl")
    _save_partial_results(
        outdir=outdir,
        summary_rows=summary_rows,
        all_rows=all_rows,
        completed_steps=completed_steps,
        total_steps=total_steps,
        start_time=start_time,
        reason="final",
    )
    logger.info("Finished run. Final results written to %s", outdir)


if __name__ == "__main__":
    main()
