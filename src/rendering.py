from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple


logger = logging.getLogger(__name__)


@dataclass
class RenderedConversation:
    text: str
    char_spans: List[Dict]


def apply_chat_template_with_spans(
    tokenizer, messages: List[Dict[str, str]]
) -> RenderedConversation:
    rendered = tokenizer.apply_chat_template(
        [{"role": m["role"], "content": m["content"]} for m in messages],
        tokenize=False,
        add_generation_prompt=True,
    )

    spans: List[Dict] = []
    cursor = 0
    for i, m in enumerate(messages):
        content = m["content"]
        start = rendered.find(content, cursor)
        if start < 0:
            # Previous fallback searched from position 0, which could match
            # an *earlier* occurrence and break span ordering.  Instead we
            # try a whitespace-stripped search in the forward region only.
            stripped = content.strip()
            start = rendered.find(stripped, cursor) if stripped != content else -1
            if start >= 0:
                logger.debug(
                    "Span for message %d found via stripped match at %d", i, start
                )
                content = stripped  # update length for end calculation
            else:
                raise ValueError(
                    f"Could not recover span for message {i} "
                    f"(cursor={cursor}): {m['content'][:80]!r}"
                )
        end = start + len(content)
        spans.append(
            {
                "message_index": i,
                "role": m["role"],
                "content": m["content"],
                "char_start": start,
                "char_end": end,
            }
        )
        cursor = end

    return RenderedConversation(text=rendered, char_spans=spans)


def char_span_to_token_span(
    offset_mapping: List[Tuple[int, int]], char_start: int, char_end: int
) -> Tuple[int, int]:
    token_ids = []
    for i, (s, e) in enumerate(offset_mapping):
        if e <= char_start:
            continue
        if s >= char_end:
            break
        if e > s:
            token_ids.append(i)
    if not token_ids:
        return -1, -1
    return token_ids[0], token_ids[-1] + 1
