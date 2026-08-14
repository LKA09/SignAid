from __future__ import annotations

from signaid.emergency.taxonomy import CONCEPTS


GLOSS_ALIASES: dict[str, tuple[str, ...]] = {
    gloss: tuple(dict.fromkeys((gloss, *concept.aliases)))
    for concept in CONCEPTS.values()
    for gloss in concept.gloss
}

# Common surface forms produced by the offline parser.
GLOSS_ALIASES.update({
    "대피": ("대피", "피하다", "탈출"),
    "금지": ("금지", "하지 마세요", "사용하지", "이용하지"),
    "아프다": ("아프다", "아파요", "통증"),
    "힘들다": ("힘들다", "힘들어요", "곤란"),
})

