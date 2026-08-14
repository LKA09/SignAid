from __future__ import annotations

import argparse
import difflib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Protocol

from signaid.emergency.taxonomy import CONCEPTS


@dataclass(slots=True)
class ParsedSignRequest:
    intent: str
    confidence: float
    gloss: list[str]
    original_text: str
    mock: bool = False
    matched_concepts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class TextToGlossBackend(Protocol):
    def parse(self, text: str) -> ParsedSignRequest: ...


class OfflineKoreanEmergencyParser:
    """Deterministic, offline Korean emergency intent/gloss parser."""

    _patterns = [
        ("EVACUATION_INSTRUCTION", re.compile(r"(엘리베이터|승강기).*(사용|이용).*(말|마|금지).*(계단).*(대피|이동)"),
         ["엘리베이터", "사용", "금지", "계단", "대피"],
         ["ELEVATOR", "DO_NOT_USE", "STAIRS", "EVACUATE"]),
        ("CHEST_PAIN", re.compile(r"(가슴|흉부|흉통).*(아프|아파|통증|답답)"),
         ["가슴", "아프다"], ["CHEST_PAIN"]),
        ("BREATHING_DIFFICULTY", re.compile(r"(숨|호흡).*(힘들|곤란|안\s*쉬|못\s*쉬)"),
         ["숨", "힘들다"], ["BREATHING_DIFFICULTY"]),
        ("CALL_119", re.compile(r"119.*(전화|신고|불러)|((전화|신고).*)119"),
         ["119", "전화"], ["CALL_119"]),
    ]

    def __init__(self, fuzzy_cutoff: float = 0.55) -> None:
        self.fuzzy_cutoff = fuzzy_cutoff

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"[^0-9가-힣a-zA-Z]+", " ", text).strip().lower()

    def parse(self, text: str) -> ParsedSignRequest:
        normalized = self._normalize(text)
        if not normalized:
            return ParsedSignRequest("UNKNOWN", 0.0, [], text)

        for intent, pattern, gloss, concept_ids in self._patterns:
            if pattern.search(normalized):
                return ParsedSignRequest(intent, 0.96, gloss, text, matched_concepts=concept_ids)

        # Preserve the order of multiple explicit emergency expressions. This
        # matters for instructions such as "불이 났어요, 계단으로 대피하세요",
        # where collapsing the whole sentence to a single fuzzy intent loses
        # most of the message.
        compact = normalized.replace(" ", "")
        exact_matches: list[tuple[int, int, str]] = []
        for concept in CONCEPTS.values():
            candidates = sorted(
                {self._normalize(value).replace(" ", "") for value in (concept.ko, *concept.aliases)},
                key=len,
                reverse=True,
            )
            for candidate in candidates:
                if not candidate:
                    continue
                position = compact.find(candidate)
                if position >= 0:
                    exact_matches.append((position, -len(candidate), concept.id))
                    break
        if exact_matches:
            exact_matches.sort()
            concept_ids = list(dict.fromkeys(item[2] for item in exact_matches))
            gloss = [word for concept_id in concept_ids for word in CONCEPTS[concept_id].gloss]
            intent = concept_ids[0] if len(concept_ids) == 1 else "COMPOUND"
            confidence = min(0.97, 0.91 + 0.01 * len(concept_ids))
            return ParsedSignRequest(intent, confidence, gloss, text, matched_concepts=concept_ids)

        scored: list[tuple[float, str]] = []
        for concept in CONCEPTS.values():
            candidates = (concept.ko, *concept.aliases)
            exact_hits = sum(1 for alias in candidates if alias.replace(" ", "") in compact)
            fuzzy = max(difflib.SequenceMatcher(None, compact, c.replace(" ", "")).ratio() for c in candidates)
            score = min(0.94, 0.65 + 0.12 * exact_hits) if exact_hits else fuzzy * 0.78
            scored.append((score, concept.id))
        confidence, intent = max(scored)
        if confidence < self.fuzzy_cutoff:
            tokens = [part for part in normalized.split() if len(part) > 1]
            return ParsedSignRequest("UNKNOWN", round(confidence, 3), tokens, text)
        return ParsedSignRequest(
            intent,
            round(confidence, 3),
            list(CONCEPTS[intent].gloss),
            text,
            matched_concepts=[intent],
        )


def main() -> None:
    cli = argparse.ArgumentParser(description="Offline Korean emergency text-to-gloss parser")
    cli.add_argument("text")
    args = cli.parse_args()
    print(json.dumps(OfflineKoreanEmergencyParser().parse(args.text).to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
