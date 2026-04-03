from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from .data_types import ConversationSample, Fact, FinalQuestion, Turn


def load_samples(path: str | Path) -> List[ConversationSample]:
    path = Path(path)
    out: List[ConversationSample] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            out.append(
                ConversationSample(
                    sample_id=raw["sample_id"],
                    task_type=raw.get("task_type", "memory_probe"),
                    facts=[Fact(**x) for x in raw["facts"]],
                    turns=[Turn(**x) for x in raw["turns"]],
                    final_question=FinalQuestion(**raw["final_question"]),
                    gold_answer=raw["gold_answer"],
                    metadata=raw.get("metadata", {}),
                )
            )
    return out


def dump_jsonl(rows: Iterable[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
