from __future__ import annotations

from typing import Dict, List

from .conversation_builders import build_messages
from .data_types import ConversationSample
from .metrics import (
    compute_attention_metrics,
    exact_or_substring_match,
    recover_fact_token_spans,
)


def materialize_messages(
    model,
    sample: ConversationSample,
    setting: str,
    assistant_temperature: float = 0.0,
) -> List[Dict[str, str]]:
    messages = build_messages(sample, setting)
    if setting != "multi_turn_self_generated":
        return messages

    realized: List[Dict[str, str]] = []
    for msg in messages:
        if msg["role"] == "assistant" and msg["content"] == "__SELF_GENERATE__":
            reply = model.generate_assistant_reply(
                realized, temperature=assistant_temperature
            )
            realized.append({"role": "assistant", "content": reply})
        else:
            realized.append(msg)
    return realized


def _serializable_fact_spans(
    fact_token_spans: Dict[str, List[tuple]],
) -> Dict[str, List[List[int]]]:
    """Convert tuple spans to JSON-friendly ``[start, end]`` lists."""
    return {
        fact_id: [[s, e] for s, e in spans]
        for fact_id, spans in fact_token_spans.items()
    }


def run_one_setting(
    model,
    sample: ConversationSample,
    setting: str,
    assistant_temperature: float = 0.0,
    answer_temperature: float = 0.0,
    max_new_tokens: int = 32,
    capture_last_k_steps: int = 1,
) -> Dict:
    messages = materialize_messages(
        model, sample, setting, assistant_temperature=assistant_temperature
    )
    step_output = model.capture_final_answer_step(
        messages,
        max_new_tokens=max_new_tokens,
        temperature=answer_temperature,
        capture_last_k_steps=capture_last_k_steps,
    )
    fact_token_spans = recover_fact_token_spans(
        sample=sample,
        messages=messages,
        message_char_spans=step_output.message_char_spans,
        offset_mapping=step_output.offset_mapping,
        setting=setting,
    )
    metrics = compute_attention_metrics(
        attentions=step_output.attentions,
        fact_token_spans=fact_token_spans,
        target_fact_ids=sample.final_question.target_fact_ids,
    )
    return {
        "sample_id": sample.sample_id,
        "setting": setting,
        "messages": messages,
        "prompt_text": step_output.prompt_text,
        "prediction": step_output.generated_text,
        "gold_answer": sample.gold_answer,
        "correct": exact_or_substring_match(
            step_output.generated_text, sample.gold_answer
        ),
        "fact_token_spans": _serializable_fact_spans(fact_token_spans),
        "metrics": metrics,
    }
