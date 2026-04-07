from __future__ import annotations

from typing import Dict, List

from .data_types import ConversationSample


NEUTRAL_ASSISTANT_REPLY = "I understand. Please continue."
DEFAULT_SYSTEM_INSTRUCTION = (
    "You are a helpful assistant. Read the conversation carefully and answer the final "
    "question using the information explicitly provided earlier in the conversation. "
    "If the answer was given earlier, repeat it exactly."
)

CORE_SETTINGS = {
    "packed_single_turn",
    "multi_turn_neutral",
    "multi_turn_dataset",
    "multi_turn_self_generated",
}

EXPERIMENTAL_SETTINGS = {
    "recap_final_turn",
    "snowball_user",
}

SETTING_NAMES = CORE_SETTINGS | EXPERIMENTAL_SETTINGS


def _system_instruction(sample: ConversationSample) -> str:
    return sample.system_instruction or DEFAULT_SYSTEM_INSTRUCTION


def build_messages(sample: ConversationSample, setting: str) -> List[Dict[str, str]]:
    if setting not in SETTING_NAMES:
        raise ValueError(
            f"Unknown setting: {setting!r}. "
            f"Choose from core={sorted(CORE_SETTINGS)} "
            f"or experimental={sorted(EXPERIMENTAL_SETTINGS)}"
        )

    if setting == "packed_single_turn":
        return _build_packed_single_turn(sample)
    if setting == "multi_turn_neutral":
        return _build_multi_turn(sample, assistant_mode="neutral", snowball=False, recap=False)
    if setting == "multi_turn_dataset":
        return _build_multi_turn(sample, assistant_mode="dataset", snowball=False, recap=False)
    if setting == "multi_turn_self_generated":
        return _build_multi_turn(sample, assistant_mode="self_generated", snowball=False, recap=False)
    if setting == "recap_final_turn":
        return _build_multi_turn(sample, assistant_mode="dataset", snowball=False, recap=True)
    if setting == "snowball_user":
        return _build_multi_turn(sample, assistant_mode="dataset", snowball=True, recap=False)

    raise AssertionError(f"unreachable: {setting}")


def _build_packed_single_turn(sample: ConversationSample) -> List[Dict[str, str]]:
    user_lines = []
    for t in sample.turns:
        if t.role == "user":
            user_lines.append(f"- {t.text}")
    user_lines.append(f"\nFinal question: {sample.final_question.text}")
    full_user = (
        "The following information was given by the user over several chat turns. "
        "Answer the final question using the earlier user information.\n\n"
        + "\n".join(user_lines)
    )
    return [
        {"role": "system", "content": _system_instruction(sample)},
        {"role": "user", "content": full_user},
    ]


def _build_multi_turn(
    sample: ConversationSample,
    assistant_mode: str,
    snowball: bool,
    recap: bool,
) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": _system_instruction(sample)}
    ]
    revealed_user_utts: List[str] = []

    for t in sample.turns:
        if t.role == "user":
            text = t.text
            if snowball and revealed_user_utts:
                bulleted = "\n".join(f"- {x}" for x in revealed_user_utts)
                text = f"Just to reiterate:\n{bulleted}\n\nAlso,\n{text}"
            messages.append({"role": "user", "content": text})
            revealed_user_utts.append(t.text)
        elif t.role == "assistant":
            if assistant_mode == "neutral":
                messages.append({"role": "assistant", "content": NEUTRAL_ASSISTANT_REPLY})
            elif assistant_mode == "dataset":
                messages.append({"role": "assistant", "content": t.text})
            elif assistant_mode == "self_generated":
                messages.append({"role": "assistant", "content": "__SELF_GENERATE__"})
            else:
                raise ValueError(f"Unknown assistant_mode: {assistant_mode}")

    if recap:
        bulleted = "\n".join(f"- {x}" for x in revealed_user_utts)
        recap_turn = (
            "Before the final question, here is a recap of the user-provided information so far:\n"
            f"{bulleted}"
        )
        messages.append({"role": "user", "content": recap_turn})

    messages.append({"role": sample.final_question.role, "content": sample.final_question.text})
    return messages
