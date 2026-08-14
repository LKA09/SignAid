from __future__ import annotations

from .taxonomy import CONCEPTS


def keyword_index() -> dict[str, str]:
    result: dict[str, str] = {}
    for concept in CONCEPTS.values():
        for keyword in (concept.ko, *concept.aliases, *concept.gloss):
            result[keyword.replace(" ", "").lower()] = concept.id
    return result

