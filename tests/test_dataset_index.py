import json
from pathlib import Path

from signaid.datasets.index import AIHubEmergencyIndex


def test_real_sample_index_status_and_search(tmp_path: Path):
    path = tmp_path / "samples.jsonl"
    records = [
        {"sample_id": "fire-1", "text": "화재가 발생했습니다 계단으로 대피하세요", "gloss": ["불타다", "계단", "도망"], "category": "FIRE", "signer_id": "TW01", "source": "AIHUB"},
        {"sample_id": "aid-1", "text": "심장마비 환자에게 응급처치를 실시하세요", "gloss": ["심장마비", "응급처치"], "category": "FIRSTAID", "signer_id": "TW02", "source": "AIHUB"},
    ]
    path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records), encoding="utf-8")
    index = AIHubEmergencyIndex(path)
    status = index.status()
    assert status["connected"] is True
    assert status["samples"] == 2
    assert status["signers"] == 2
    result = index.search("화재 계단 대피", limit=1)
    assert result[0]["sample_id"] == "fire-1"
    assert result[0]["source"] == "AIHUB"
