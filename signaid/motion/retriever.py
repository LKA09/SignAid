from __future__ import annotations

import difflib
from pathlib import Path

import numpy as np

from signaid.config import settings
from signaid.datasets.mock import build_mock_dataset
from signaid.emergency.taxonomy import CONCEPTS
from signaid.motion.schema import MotionMatch, MotionRetrievalResult


class MotionRetriever:
    def __init__(self, motion_dir: Path | None = None, fuzzy_cutoff: float = 0.48, auto_mock: bool = True) -> None:
        self.motion_dir = motion_dir or settings.motion_dir
        self.real_motion_dir = self.motion_dir.parent / "aihub_motions"
        self.fuzzy_cutoff = fuzzy_cutoff
        existing = next(self.motion_dir.glob("*.npz"), None)
        rebuild = existing is None
        if auto_mock and existing is not None:
            try:
                with np.load(existing) as data:
                    key = "motion" if "motion" in data else data.files[0]
                    rebuild = data[key].shape[1] < 59
            except (OSError, ValueError, KeyError):
                rebuild = True
        if auto_mock and rebuild:
            build_mock_dataset(self.motion_dir)
        self.lookup: dict[str, tuple[str, str]] = {}
        for concept in CONCEPTS.values():
            for gloss in (*concept.gloss, concept.ko, *concept.aliases):
                self.lookup[self._norm(gloss)] = (concept.id, concept.gloss[0])

    @staticmethod
    def _norm(value: str) -> str:
        return "".join(value.lower().split())

    def retrieve(self, glosses: list[str]) -> MotionRetrievalResult:
        result = MotionRetrievalResult()
        keys = list(self.lookup)
        for query in glosses:
            norm = self._norm(query)
            confidence = 1.0
            matched = norm if norm in self.lookup else None
            if matched is None:
                found = difflib.get_close_matches(norm, keys, n=1, cutoff=self.fuzzy_cutoff)
                matched = found[0] if found else None
                confidence = difflib.SequenceMatcher(None, norm, matched).ratio() if matched else 0.0
            if matched is None:
                result.missing.append(query)
                result.matches.append(MotionMatch(query, None, None, None, 0.0))
                continue
            concept_id, canonical = self.lookup[matched]
            path = self.path_for_concept(concept_id)
            if not path.exists():
                result.missing.append(query)
                result.matches.append(MotionMatch(query, concept_id, canonical, None, confidence))
            else:
                result.matches.append(MotionMatch(query, concept_id, canonical, path, confidence))
        return result

    def path_for_concept(self, concept_id: str) -> Path:
        real_path = self.real_motion_dir / f"{concept_id}.npz"
        return real_path if real_path.exists() else self.motion_dir / f"{concept_id}.npz"
