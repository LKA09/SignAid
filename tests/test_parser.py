from signaid.text.parser import OfflineKoreanEmergencyParser


def test_chest_pain_parser():
    result = OfflineKoreanEmergencyParser().parse("가슴이 너무 아파요")
    assert result.intent == "CHEST_PAIN"
    assert result.gloss == ["가슴", "아프다"]
    assert result.confidence > 0.9


def test_breathing_parser():
    result = OfflineKoreanEmergencyParser().parse("숨을 쉬기 너무 힘들어요")
    assert result.intent == "BREATHING_DIFFICULTY"
    assert result.gloss == ["숨", "힘들다"]


def test_evacuation_instruction_parser():
    result = OfflineKoreanEmergencyParser().parse("엘리베이터를 이용하지 말고 계단으로 대피하세요")
    assert result.intent == "EVACUATION_INSTRUCTION"
    assert result.gloss == ["엘리베이터", "사용", "금지", "계단", "대피"]


def test_alias_and_unknown():
    parser = OfflineKoreanEmergencyParser()
    assert parser.parse("흉통이 있어요").intent == "CHEST_PAIN"
    assert parser.parse("   ").intent == "UNKNOWN"


def test_compound_emergency_preserves_message_order():
    result = OfflineKoreanEmergencyParser().parse("불이 났어요. 계단으로 대피하세요")
    assert result.intent == "COMPOUND"
    assert result.matched_concepts == ["FIRE", "STAIRS", "EVACUATE"]
    assert result.gloss == ["불", "계단", "대피"]
    assert result.confidence > 0.9
