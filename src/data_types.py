from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Fact:
    fact_id: str
    text: str
    answer_target: str
    priority: int = 1
    revealed_at_turn: Optional[int] = None


@dataclass
class Turn:
    turn_id: int
    role: str
    text: str
    contains_fact_ids: List[str] = field(default_factory=list)
    source: Optional[str] = None


@dataclass
class FinalQuestion:
    role: str
    text: str
    target_fact_ids: List[str]


@dataclass
class ConversationSample:
    sample_id: str
    task_type: str
    facts: List[Fact]
    turns: List[Turn]
    final_question: FinalQuestion
    gold_answer: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def fact_map(self) -> Dict[str, Fact]:
        return {f.fact_id: f for f in self.facts}
