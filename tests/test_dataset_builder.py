from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path

from generate_dataset import build_sample, generate_needle, iter_sharegpt_conversations


def _toy_chain():
    return [
        {"role": "user", "text": "Hello."},
        {"role": "assistant", "text": "Hi."},
        {"role": "user", "text": "I went outside."},
        {"role": "assistant", "text": "Nice."},
        {"role": "user", "text": "I drank coffee."},
        {"role": "assistant", "text": "Sounds good."},
        {"role": "user", "text": "Now I am back home."},
    ]


def run():
    rng = random.Random(123)

    kind, fact_text, question, value = generate_needle(rng, "mixed")
    assert kind in {"magic_number", "secret_code", "reference_id"}
    assert value in fact_text
    assert question.endswith("?")
    print("  [PASS] generate_needle")

    sample = build_sample(
        chain=_toy_chain(),
        sample_idx=0,
        rng=random.Random(7),
        min_user_turns=4,
        min_post_fact_user_turns=2,
        needle_style="mixed",
    )
    assert sample is not None
    assert sample["facts"][0]["text"] in [t["text"] for t in sample["turns"]]
    assert sample["facts"][0]["answer_target"] == sample["gold_answer"]
    assert sample["final_question"]["target_fact_ids"] == ["f1"]
    assert sample["metadata"]["needle_source"] == "procedural_retrieval_head_style"
    print("  [PASS] build_sample")

    raw = [
        {
            "conversations": [
                {"from": "human", "value": "Hello"},
                {"from": "gpt", "value": "Hi"},
                {"from": "human", "value": "A"},
                {"from": "gpt", "value": "B"},
            ]
        }
    ]
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "toy_sharegpt.json"
        p.write_text(json.dumps(raw), encoding="utf-8")
        chains = list(iter_sharegpt_conversations(str(p)))
        assert len(chains) == 1
        assert chains[0][0]["role"] == "user"
        assert chains[0][1]["role"] == "assistant"
    print("  [PASS] iter_sharegpt_conversations")
