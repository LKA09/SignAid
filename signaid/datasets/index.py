from __future__ import annotations

import difflib
import json
import re
from collections import Counter
from pathlib import Path

from signaid.config import settings


class AIHubEmergencyIndex:
    """Lazy, read-only access to the filtered real AI Hub sample index."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or settings.data_dir / "index" / "emergency_samples.jsonl"
        self._records: list[dict] | None = None

    def _load(self) -> list[dict]:
        if self._records is None:
            self._records = []
            if self.path.exists():
                with self.path.open(encoding="utf-8") as source:
                    for line in source:
                        try:
                            self._records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        return self._records

    def status(self) -> dict:
        records = self._load()
        categories = Counter(str(item.get("category", "unknown")) for item in records)
        signers = {str(item.get("signer_id")) for item in records if item.get("signer_id")}
        return {
            "connected": bool(records),
            "samples": len(records),
            "categories": dict(sorted(categories.items())),
            "signers": len(signers),
            "missing_gloss": sum(not item.get("gloss") for item in records),
            "source": "AIHUB" if records else None,
            "path": self.path.name,
        }

    @staticmethod
    def _normalize(value: str) -> str:
        return re.sub(r"[^0-9가-힣a-z]+", " ", value.lower()).strip()

    def search(self, text: str, limit: int = 3) -> list[dict]:
        query = self._normalize(text)
        if not query:
            return []
        query_terms = set(query.split())
        roots = {term[:2] for term in query_terms if len(term) >= 2}
        query_compact = query.replace(" ", "")
        query_bigrams = {query_compact[i:i + 2] for i in range(max(1, len(query_compact) - 1))}
        candidates: list[tuple[float, dict]] = []
        for record in self._load():
            candidate = self._normalize(str(record.get("text", "")))
            if roots and not any(root in candidate for root in roots):
                continue
            candidate_terms = set(candidate.split())
            overlap = len(query_terms & candidate_terms) / max(1, len(query_terms))
            contains = 1.0 if query in candidate or candidate in query else 0.0
            candidate_compact = candidate.replace(" ", "")
            candidate_bigrams = {candidate_compact[i:i + 2] for i in range(max(1, len(candidate_compact) - 1))}
            char_overlap = len(query_bigrams & candidate_bigrams) / max(1, len(query_bigrams))
            if not overlap and not contains and char_overlap < 0.15:
                continue
            ratio = difflib.SequenceMatcher(None, query, candidate).ratio()
            score = 0.4 * ratio + 0.25 * overlap + 0.2 * char_overlap + 0.15 * contains
            if score >= 0.16:
                candidates.append((score, record))
        candidates.sort(key=lambda item: item[0], reverse=True)
        return [
            {
                "sample_id": record.get("sample_id"),
                "text": record.get("text"),
                "gloss": record.get("gloss", []),
                "category": record.get("category"),
                "signer_id": record.get("signer_id"),
                "similarity": round(score, 3),
                "source": "AIHUB",
            }
            for score, record in candidates[: max(1, min(limit, 10))]
        ]
