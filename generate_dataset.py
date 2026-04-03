from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from datasets import load_dataset

# Retrieval-head / needle-in-a-haystack style:
# - one short, fully controlled needle sentence
# - one unambiguous target value
# - one direct retrieval question
# The multi-turn adaptation here keeps the natural chat haystack,
# but inserts the same kind of synthetic needle as an extra user turn.

CITIES = [
    "Chicago", "Seoul", "Busan", "Tokyo", "Paris", "London", "Berlin", "Sydney",
    "Toronto", "Madrid", "Rome", "Lisbon", "Oslo", "Stockholm", "Helsinki", "Prague",
    "Vienna", "Dublin", "Warsaw", "Budapest", "Athens", "Zurich", "Geneva", "Munich",
]

NEEDLE_TYPES = [
    {
        "name": "magic_number",
        "fact": "The special magic {city} number is {value}.",
        "question": "What is the special magic {city} number?",
    },
    {
        "name": "secret_code",
        "fact": "The secret code for {city} is {value}.",
        "question": "What is the secret code for {city}?",
    },
    {
        "name": "reference_id",
        "fact": "The reference ID for {city} is {value}.",
        "question": "What is the reference ID for {city}?",
    },
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--chat_source", type=str, choices=["oasst1", "local_sharegpt"], required=True)
    p.add_argument("--oasst_split", type=str, default="train")
    p.add_argument("--sharegpt_path", type=str, default=None)
    p.add_argument("--output_jsonl", type=str, required=True)
    p.add_argument("--num_samples", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min_user_turns", type=int, default=4)
    p.add_argument("--min_post_fact_user_turns", type=int, default=2)
    p.add_argument("--max_total_turns", type=int, default=10)
    p.add_argument(
        "--needle_style",
        type=str,
        choices=[x["name"] for x in NEEDLE_TYPES] + ["mixed"],
        default="mixed",
    )
    return p.parse_args()


def iter_oasst_conversations(split: str, rng: random.Random | None = None):
    ds = load_dataset("OpenAssistant/oasst1", split=split)
    messages = [x for x in ds if x.get("text")]
    children = {}
    roots = []
    for m in messages:
        parent = m.get("parent_id")
        if parent is None:
            roots.append(m)
        else:
            children.setdefault(parent, []).append(m)

    # Shuffle roots so that different seeds produce different orderings
    if rng is not None:
        rng.shuffle(roots)

    for root in roots:
        chain = []
        cur = root
        while cur is not None:
            role = "assistant" if cur.get("role") == "assistant" else "user"
            chain.append({"role": role, "text": cur["text"]})
            nxt = children.get(cur["message_id"], [])
            if not nxt:
                cur = None
            elif rng is not None:
                cur = rng.choice(nxt)
            else:
                cur = nxt[0]
        if len(chain) >= 4:
            yield chain


def iter_sharegpt_conversations(path: str):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    for item in raw:
        convs = item.get("conversations", [])
        chain = []
        for x in convs:
            fr = str(x.get("from", "")).lower()
            role = "assistant" if fr in {"gpt", "assistant", "chatgpt"} else "user"
            value = x.get("value", "")
            if value:
                chain.append({"role": role, "text": value})
        if len(chain) >= 4:
            yield chain


def keep_chat_prefix(chain: List[Dict[str, str]], max_total_turns: int) -> List[Dict[str, str]]:
    return chain[:max_total_turns]


def generate_numeric_value(rng: random.Random) -> str:
    # retrieval-head style needle values are short and unambiguous
    return "".join(rng.choice("0123456789") for _ in range(5))


def choose_template(rng: random.Random, needle_style: str) -> Dict[str, str]:
    if needle_style == "mixed":
        return rng.choice(NEEDLE_TYPES)
    for t in NEEDLE_TYPES:
        if t["name"] == needle_style:
            return t
    raise ValueError(f"Unknown needle_style: {needle_style}")


def generate_needle(rng: random.Random, needle_style: str) -> Tuple[str, str, str, str]:
    template = choose_template(rng, needle_style)
    city = rng.choice(CITIES)
    value = generate_numeric_value(rng)
    fact_text = template["fact"].format(city=city, value=value)
    question = template["question"].format(city=city)
    return template["name"], fact_text, question, value


def build_sample(
    chain: List[Dict[str, str]],
    sample_idx: int,
    rng: random.Random,
    min_user_turns: int,
    min_post_fact_user_turns: int,
    needle_style: str,
) -> Optional[dict]:
    user_positions = [i for i, t in enumerate(chain) if t["role"] == "user"]
    if len(user_positions) < min_user_turns:
        return None

    possible_insert_after = []
    for pos_idx, chain_idx in enumerate(user_positions[:-1]):
        remaining_user_turns = len(user_positions) - (pos_idx + 1)
        if remaining_user_turns >= min_post_fact_user_turns:
            possible_insert_after.append(chain_idx)
    if not possible_insert_after:
        return None

    insert_after_idx = rng.choice(possible_insert_after)
    needle_kind, fact_text, final_question, gold_answer = generate_needle(rng, needle_style)

    out_turns = []
    turn_id = 0
    fact_inserted_turn_id = None

    for i, turn in enumerate(chain):
        out_turns.append(
            {
                "turn_id": turn_id,
                "role": turn["role"],
                "text": turn["text"],
                "contains_fact_ids": [],
                **({"source": "dataset"} if turn["role"] == "assistant" else {}),
            }
        )
        turn_id += 1

        if i == insert_after_idx:
            fact_inserted_turn_id = turn_id
            out_turns.append(
                {
                    "turn_id": turn_id,
                    "role": "user",
                    "text": fact_text,
                    "contains_fact_ids": ["f1"],
                }
            )
            turn_id += 1

    if fact_inserted_turn_id is None:
        return None

    return {
        "sample_id": f"sample_{sample_idx:06d}",
        "task_type": "memory_probe",
        "facts": [
            {
                "fact_id": "f1",
                "text": fact_text,
                "answer_target": gold_answer,
                "priority": 1,
                "revealed_at_turn": fact_inserted_turn_id,
            }
        ],
        "turns": out_turns,
        "final_question": {
            "role": "user",
            "text": final_question,
            "target_fact_ids": ["f1"],
        },
        "gold_answer": gold_answer,
        "metadata": {
            "language": "en",
            "domain": "casual_chat",
            "chat_source": "synthetic_from_chat",
            "needle_style": needle_kind,
            "needle_source": "procedural_retrieval_head_style",
        },
    }


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    if args.chat_source == "oasst1":
        iterator = iter_oasst_conversations(args.oasst_split, rng=rng)
    else:
        if not args.sharegpt_path:
            raise ValueError("--sharegpt_path is required for local_sharegpt")
        iterator = iter_sharegpt_conversations(args.sharegpt_path)

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with output_path.open("w", encoding="utf-8") as f:
        for chain in iterator:
            if written >= args.num_samples:
                break
            chain = keep_chat_prefix(chain, args.max_total_turns)
            sample = build_sample(
                chain=chain,
                sample_idx=written,
                rng=rng,
                min_user_turns=args.min_user_turns,
                min_post_fact_user_turns=args.min_post_fact_user_turns,
                needle_style=args.needle_style,
            )
            if sample is None:
                continue
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} samples to {output_path}")


if __name__ == "__main__":
    main()
