from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class EmergencyConcept:
    id: str
    ko: str
    gloss: tuple[str, ...]
    category: str
    aliases: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        value = asdict(self)
        value["gloss"] = list(self.gloss)
        value["aliases"] = list(self.aliases)
        value["motion_path"] = f"data/processed/motions/{self.id}.npz"
        return value


def _c(id: str, ko: str, gloss: tuple[str, ...], category: str, *aliases: str) -> EmergencyConcept:
    return EmergencyConcept(id, ko, gloss, category, aliases)


_CONCEPTS = [
    _c("HELP", "도와주세요", ("도움",), "general", "도와줘", "구조"),
    _c("FIRE", "불이 났어요", ("불",), "disaster", "화재", "불났어요"),
    _c("EVACUATE", "대피하세요", ("대피",), "instruction", "피하세요", "탈출"),
    _c("DANGER", "위험합니다", ("위험",), "general", "위험", "조심"),
    _c("AMBULANCE", "구급차", ("구급차",), "medical", "응급차"),
    _c("HOSPITAL", "병원", ("병원",), "medical", "응급실"),
    _c("PAIN", "아파요", ("아프다",), "medical", "통증", "아픔"),
    _c("CHEST_PAIN", "가슴이 아파요", ("가슴", "아프다"), "medical", "가슴 통증", "흉통", "가슴이 너무 아파요"),
    _c("BLEEDING", "피가 나요", ("피", "나다"), "medical", "출혈", "피나요"),
    _c("BREATHING_DIFFICULTY", "숨쉬기 힘들어요", ("숨", "힘들다"), "medical", "호흡 곤란", "숨이 안 쉬어져요", "숨을 쉬기 너무 힘들어요"),
    _c("ALLERGY", "알레르기가 있어요", ("알레르기",), "medical", "알러지"),
    _c("MEDICINE", "약이 필요해요", ("약", "필요"), "medical", "의약품", "약"),
    _c("UNCONSCIOUS", "의식이 없어요", ("의식", "없다"), "medical", "무의식", "기절"),
    _c("SEIZURE", "발작을 해요", ("발작",), "medical", "경련"),
    _c("STAIRS", "계단", ("계단",), "instruction", "층계"),
    _c("ELEVATOR", "엘리베이터", ("엘리베이터",), "instruction", "승강기"),
    _c("DO_NOT_USE", "사용하지 마세요", ("사용", "금지"), "instruction", "이용 금지", "쓰지 마세요"),
    _c("LEFT", "왼쪽", ("왼쪽",), "direction"),
    _c("RIGHT", "오른쪽", ("오른쪽",), "direction"),
    _c("WAIT", "기다리세요", ("기다리다",), "instruction", "대기"),
    _c("ACCIDENT", "사고", ("사고",), "accident", "교통사고"),
    _c("FALL", "넘어졌어요", ("넘어지다",), "accident", "추락", "낙상"),
    _c("BURN", "화상을 입었어요", ("화상",), "medical", "데었어요"),
    _c("PREGNANCY", "임신 중이에요", ("임신",), "medical", "임산부"),
    _c("CALL_119", "119에 신고해 주세요", ("119", "전화",), "general", "119", "신고"),
    _c("SMOKE", "연기가 나요", ("연기",), "disaster", "연기"),
    _c("EXPLOSION", "폭발", ("폭발",), "disaster", "폭발했어요"),
    _c("COLLAPSE", "건물이 무너졌어요", ("건물", "무너지다"), "disaster", "붕괴"),
    _c("ELECTRIC_SHOCK", "감전됐어요", ("감전",), "accident", "전기 사고"),
    _c("FLOOD", "물이 차올라요", ("홍수",), "disaster", "침수", "홍수"),
    _c("EARTHQUAKE", "지진이 났어요", ("지진",), "disaster", "지진"),
]

CONCEPTS = {concept.id: concept for concept in _CONCEPTS}

