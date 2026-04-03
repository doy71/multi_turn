from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .data_types import ConversationSample
from .rendering import char_span_to_token_span

logger = logging.getLogger(__name__)


def recover_fact_token_spans(
    sample: ConversationSample,
    messages: List[Dict[str, str]],
    message_char_spans: List[Dict],
    offset_mapping: List[Tuple[int, int]],
    setting: str = "",
) -> Dict[str, List[Tuple[int, int]]]:
    """Recover token-level spans for each fact in the rendered prompt.

    For ``packed_single_turn`` (all user text merged into one message) the
    turn-cursor approach does not work because there is no 1:1 mapping
    between messages and turns.  We therefore fall back to a direct
    substring search of the fact text inside every message char span.
    """
    fact_spans: Dict[str, List[Tuple[int, int]]] = defaultdict(list)

    if setting == "packed_single_turn":
        # --- direct text-search path for packed settings ----------------
        for fact in sample.facts:
            for span in message_char_spans:
                local_start = span["content"].find(fact.text)
                if local_start < 0:
                    continue
                local_end = local_start + len(fact.text)
                char_start = span["char_start"] + local_start
                char_end = span["char_start"] + local_end
                tok_span = char_span_to_token_span(
                    offset_mapping, char_start, char_end
                )
                if tok_span != (-1, -1):
                    fact_spans[fact.fact_id].append(tok_span)
        # warn if any fact was not found at all
        for fact in sample.facts:
            if fact.fact_id not in fact_spans:
                logger.warning(
                    "[%s] fact %s text not found in packed message for sample %s",
                    setting,
                    fact.fact_id,
                    sample.sample_id,
                )
        return dict(fact_spans)

    # --- standard multi-turn path: map messages -> turns ----------------
    msg_index_to_turn = {}
    turn_cursor = 0
    for msg_idx, msg in enumerate(messages):
        if msg["role"] == "user" and turn_cursor < len(sample.turns):
            while (
                turn_cursor < len(sample.turns)
                and sample.turns[turn_cursor].role != "user"
            ):
                turn_cursor += 1
            if turn_cursor < len(sample.turns):
                msg_index_to_turn[msg_idx] = sample.turns[turn_cursor]
                turn_cursor += 1

    for span in message_char_spans:
        msg_idx = span["message_index"]
        if msg_idx not in msg_index_to_turn:
            continue
        turn = msg_index_to_turn[msg_idx]
        for fact_id in turn.contains_fact_ids:
            fact = sample.fact_map[fact_id]
            local_start = span["content"].find(fact.text)
            if local_start < 0:
                logger.warning(
                    "[%s] fact %s text not found in message %d for sample %s; "
                    "skipping this span (no silent whole-message fallback)",
                    setting,
                    fact_id,
                    msg_idx,
                    sample.sample_id,
                )
                continue
            local_end = local_start + len(fact.text)
            char_start = span["char_start"] + local_start
            char_end = span["char_start"] + local_end
            tok_span = char_span_to_token_span(
                offset_mapping, char_start, char_end
            )
            if tok_span != (-1, -1):
                fact_spans[fact_id].append(tok_span)
    return dict(fact_spans)


def compute_attention_metrics(
    attentions: Sequence,
    fact_token_spans: Dict[str, List[Tuple[int, int]]],
    target_fact_ids: List[str],
    topk: int = 10,
) -> Dict:
    if not attentions:
        return {
            "layer_head_fact_mass": [],
            "layer_head_target_mass": [],
            "target_fact_total_mass_mean": 0.0,
            "topk_hit_rate": 0.0,
            "top_sources": [],
        }

    layer_head_fact_mass = []
    layer_head_target_mass = []
    topk_hits = []
    top_sources = []

    target_spans = []
    for fact_id in target_fact_ids:
        target_spans.extend(fact_token_spans.get(fact_id, []))

    for layer_idx, layer_attn in enumerate(attentions):
        # layer_attn shape: [batch, heads, query_steps, key_len]
        # After .numpy()[0] -> [heads, query_steps, key_len]  (3-D expected)
        arr = layer_attn.numpy()[0]
        if arr.ndim == 3:
            arr = arr[:, -1, :]  # [heads, key_len]
        elif arr.ndim == 2:
            # Already [heads, key_len] -- single query step was squeezed
            pass
        else:
            raise ValueError(
                f"Unexpected attention ndim={arr.ndim} at layer {layer_idx}; "
                f"expected 2 or 3 after removing batch dim"
            )

        layer_fact = []
        layer_target = []
        for head_idx in range(arr.shape[0]):
            weights = arr[head_idx]
            fact_masses = {}
            for fact_id, spans in fact_token_spans.items():
                mass = 0.0
                for s, e in spans:
                    mass += float(weights[s:e].sum())
                fact_masses[fact_id] = mass
            layer_fact.append(fact_masses)

            target_mass = 0.0
            for s, e in target_spans:
                target_mass += float(weights[s:e].sum())
            layer_target.append(target_mass)

            top_idx = np.argsort(weights)[-topk:][::-1]
            hit = any(
                any(s <= idx < e for s, e in target_spans) for idx in top_idx
            )
            topk_hits.append(float(hit))
            top_sources.append(
                {
                    "layer": layer_idx,
                    "head": head_idx,
                    "top_indices": top_idx.tolist(),
                    "top_values": [float(weights[i]) for i in top_idx],
                }
            )
        layer_head_fact_mass.append(layer_fact)
        layer_head_target_mass.append(layer_target)

    target_mass_mean = float(
        np.mean([x for layer in layer_head_target_mass for x in layer])
    )
    topk_hit_rate = float(np.mean(topk_hits)) if topk_hits else 0.0
    return {
        "layer_head_fact_mass": layer_head_fact_mass,
        "layer_head_target_mass": layer_head_target_mass,
        "target_fact_total_mass_mean": target_mass_mean,
        "topk_hit_rate": topk_hit_rate,
        "top_sources": top_sources,
    }


def exact_or_substring_match(pred: str, gold: str) -> bool:
    p = pred.strip().lower()
    g = gold.strip().lower()
    return p == g or g in p
