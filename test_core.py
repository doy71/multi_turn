"""Offline sanity checks — no GPU or model required."""
from __future__ import annotations

import json
import sys
import textwrap

sys.path.insert(0, ".")

from src.data_types import ConversationSample, Fact, FinalQuestion, Turn
from src.conversation_builders import (
    CORE_SETTINGS,
    EXPERIMENTAL_SETTINGS,
    SETTING_NAMES,
    build_messages,
)
from src.metrics import recover_fact_token_spans, exact_or_substring_match
from src.runner import _serializable_fact_spans


# ---- helpers -----------------------------------------------------------

def make_sample() -> ConversationSample:
    return ConversationSample(
        sample_id="test_001",
        task_type="memory_probe",
        facts=[
            Fact(
                fact_id="f1",
                text="The secret code for Seoul is 12345.",
                answer_target="12345",
                priority=1,
                revealed_at_turn=3,
            )
        ],
        turns=[
            Turn(turn_id=0, role="user", text="Hello there.", contains_fact_ids=[]),
            Turn(turn_id=1, role="assistant", text="Hi!", contains_fact_ids=[], source="dataset"),
            Turn(turn_id=2, role="user", text="I like coffee.", contains_fact_ids=[]),
            Turn(turn_id=3, role="user", text="The secret code for Seoul is 12345.", contains_fact_ids=["f1"]),
            Turn(turn_id=4, role="assistant", text="Got it.", contains_fact_ids=[], source="dataset"),
            Turn(turn_id=5, role="user", text="Also I went to school.", contains_fact_ids=[]),
        ],
        final_question=FinalQuestion(
            role="user",
            text="What is the secret code for Seoul?",
            target_fact_ids=["f1"],
        ),
        gold_answer="12345",
    )


# ---- test: settings ---------------------------------------------------

def test_setting_sets():
    assert "packed_single_turn" in CORE_SETTINGS
    assert "recap_final_turn" in EXPERIMENTAL_SETTINGS
    assert SETTING_NAMES == CORE_SETTINGS | EXPERIMENTAL_SETTINGS
    print("  [PASS] setting sets")


# ---- test: build_messages all settings ---------------------------------

def test_build_messages_all():
    sample = make_sample()
    for setting in SETTING_NAMES:
        msgs = build_messages(sample, setting)
        assert len(msgs) >= 1, f"{setting}: no messages"
        assert msgs[-1]["role"] == "user", f"{setting}: last msg not user"
        assert "Seoul" in msgs[-1]["content"], f"{setting}: final question missing"
    print("  [PASS] build_messages all settings")


# ---- test: packed_single_turn includes fact text -----------------------

def test_packed_contains_fact():
    sample = make_sample()
    msgs = build_messages(sample, "packed_single_turn")
    assert len(msgs) == 1
    assert "The secret code for Seoul is 12345." in msgs[0]["content"]
    print("  [PASS] packed_single_turn contains fact text")


# ---- test: fact span recovery for packed_single_turn -------------------

def test_packed_fact_recovery():
    """The critical bug fix: packed_single_turn must find fact spans."""
    sample = make_sample()
    msgs = build_messages(sample, "packed_single_turn")
    full_text = msgs[0]["content"]

    # Simulate char spans (single message starting at char 0)
    message_char_spans = [
        {
            "message_index": 0,
            "role": "user",
            "content": full_text,
            "char_start": 0,
            "char_end": len(full_text),
        }
    ]
    # Simulate a trivial 1-char-per-token offset mapping
    offset_mapping = [(i, i + 1) for i in range(len(full_text))]

    spans = recover_fact_token_spans(
        sample=sample,
        messages=msgs,
        message_char_spans=message_char_spans,
        offset_mapping=offset_mapping,
        setting="packed_single_turn",
    )
    assert "f1" in spans, "fact f1 not recovered in packed_single_turn!"
    s, e = spans["f1"][0]
    recovered = full_text[s:e]
    assert recovered == "The secret code for Seoul is 12345.", f"Wrong span: {recovered!r}"
    print("  [PASS] packed_single_turn fact span recovery")


# ---- test: multi_turn fact recovery ------------------------------------

def test_multiturn_fact_recovery():
    sample = make_sample()
    msgs = build_messages(sample, "multi_turn_neutral")

    # Find which message has the fact
    fact_msg_idx = None
    for i, m in enumerate(msgs):
        if "12345" in m["content"]:
            fact_msg_idx = i
            break
    assert fact_msg_idx is not None

    # Build fake char spans
    cursor = 0
    message_char_spans = []
    full_text = ""
    for i, m in enumerate(msgs):
        tag = f"<msg{i}>"
        full_text += tag + m["content"]
        start = cursor + len(tag)
        end = start + len(m["content"])
        message_char_spans.append({
            "message_index": i,
            "role": m["role"],
            "content": m["content"],
            "char_start": start,
            "char_end": end,
        })
        cursor = end

    offset_mapping = [(i, i + 1) for i in range(len(full_text))]

    spans = recover_fact_token_spans(
        sample=sample,
        messages=msgs,
        message_char_spans=message_char_spans,
        offset_mapping=offset_mapping,
        setting="multi_turn_neutral",
    )
    assert "f1" in spans, "fact f1 not recovered in multi_turn_neutral!"
    print("  [PASS] multi_turn_neutral fact span recovery")


# ---- test: silent fallback removed ------------------------------------

def test_no_silent_fallback():
    """If fact text is absent from message, span should NOT be added."""
    sample = make_sample()
    msgs = build_messages(sample, "multi_turn_neutral")

    # Corrupt the fact text in the message so find() fails
    for i, m in enumerate(msgs):
        if "12345" in m["content"]:
            msgs[i] = {**m, "content": "CORRUPTED TEXT"}
            break

    cursor = 0
    message_char_spans = []
    full = ""
    for i, m in enumerate(msgs):
        tag = f"<m{i}>"
        full += tag + m["content"]
        start = cursor + len(tag)
        end = start + len(m["content"])
        message_char_spans.append({
            "message_index": i, "role": m["role"],
            "content": m["content"],
            "char_start": start, "char_end": end,
        })
        cursor = end

    offset_mapping = [(i, i + 1) for i in range(len(full))]
    spans = recover_fact_token_spans(
        sample=sample, messages=msgs,
        message_char_spans=message_char_spans,
        offset_mapping=offset_mapping,
        setting="multi_turn_neutral",
    )
    # Old code would silently use whole-message span; new code skips
    assert "f1" not in spans, "f1 should NOT be recovered from corrupted message"
    print("  [PASS] no silent whole-message fallback")


# ---- test: serializable fact spans ------------------------------------

def test_serializable_spans():
    raw = {"f1": [(0, 10), (20, 30)]}
    result = _serializable_fact_spans(raw)
    assert result == {"f1": [[0, 10], [20, 30]]}
    # Verify JSON round-trip
    dumped = json.dumps(result)
    loaded = json.loads(dumped)
    assert loaded == result
    print("  [PASS] fact spans JSON-serializable")


# ---- test: exact_or_substring_match -----------------------------------

def test_matching():
    assert exact_or_substring_match("12345", "12345")
    assert exact_or_substring_match("The answer is 12345.", "12345")
    assert not exact_or_substring_match("12346", "12345")
    print("  [PASS] exact_or_substring_match")


# ---- run all ----------------------------------------------------------

if __name__ == "__main__":
    print("Running core tests...\n")
    test_setting_sets()
    test_build_messages_all()
    test_packed_contains_fact()
    test_packed_fact_recovery()
    test_multiturn_fact_recovery()
    test_no_silent_fallback()
    test_serializable_spans()
    test_matching()
    print("\nAll tests passed.")
