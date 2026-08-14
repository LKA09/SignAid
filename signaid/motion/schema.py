from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class MotionMatch:
    query: str
    concept_id: str | None
    matched_gloss: str | None
    path: Path | None
    confidence: float

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "concept_id": self.concept_id,
            "matched_gloss": self.matched_gloss,
            # Do not expose host filesystem paths through the public API.
            "path": self.path.name if self.path else None,
            "confidence": round(self.confidence, 3),
        }


@dataclass(slots=True)
class MotionRetrievalResult:
    matches: list[MotionMatch] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    @property
    def confidence(self) -> float:
        return sum(x.confidence for x in self.matches) / len(self.matches) if self.matches else 0.0

    @property
    def paths(self) -> list[Path]:
        return [x.path for x in self.matches if x.path is not None]
