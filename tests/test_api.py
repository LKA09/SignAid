import numpy as np
from fastapi.testclient import TestClient

from app.api.main import app


client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_service_status_reports_real_capabilities():
    response = client.get("/api/status")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["text_to_sign"] is True
    assert body["real_motions"] > 0
    assert body["three_dimensional_motions"] > 0
    assert body["total_concepts"] == 31


def test_emergency_signs():
    response = client.get("/api/emergency-signs")
    assert response.status_code == 200
    assert len(response.json()) == 31


def test_text_to_sign():
    response = client.post("/api/text-to-sign", json={"text": "가슴이 너무 아파요"})
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "CHEST_PAIN"
    assert body["gloss"] == ["가슴", "아프다"]
    assert not body["missing"]
    assert all(item["path"] for item in body["motions"])
    assert all("/" not in item["path"] and "\\" not in item["path"] for item in body["motions"])


def test_mock_recognition_endpoint():
    points = np.zeros((4, 47, 3), dtype=float)
    response = client.post("/api/recognize", json={"keypoints": points.tolist(), "allow_mock": True})
    assert response.status_code == 200
    assert response.json()["mock"] is True


def test_unconfigured_recognition_never_returns_a_fake_prediction_by_default():
    points = np.zeros((4, 47, 3), dtype=float)
    response = client.post("/api/recognize", json={"keypoints": points.tolist()})
    assert response.status_code == 503


def test_unknown_text_render_uses_safe_fallback():
    response = client.post("/api/render", json={"text": "등록되지 않은 임의 문장", "gloss": []})
    assert response.status_code == 200
    body = response.json()
    assert body["fallback_motion"] is True
    assert body["renderer"] == "vrm_3d"
    assert body["url"] is None
    assert body["motion"]


def test_compound_text_to_sign_keeps_all_explicit_actions():
    response = client.post("/api/text-to-sign", json={"text": "불이 났어요 계단으로 대피하세요"})
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "COMPOUND"
    assert body["matched_concepts"] == ["FIRE", "STAIRS", "EVACUATE"]
    assert body["gloss"] == ["불", "계단", "대피"]
    assert body["supported"] is True


def test_3d_render_features_stay_frame_aligned():
    response = client.post("/api/render", json={"text": "가슴이 너무 아파요"})
    assert response.status_code == 200
    body = response.json()
    assert body["motion_source"] == "aihub_landmarks_3d"
    assert body["landmark_dimensions"] == 3
    assert body["annotation_aligned"] is True
    assert body["expert_validated"] is False
    assert len(body["motion"]) == len(body["palm_normals"])
    assert len(body["motion"]) == len(body["facial_expressions"])
    assert len(body["motion"]) == len(body["head_rotations"])


def test_recognition_payload_limits_are_enforced():
    points = np.zeros((601, 2, 3), dtype=float)
    response = client.post("/api/recognize", json={"keypoints": points.tolist(), "allow_mock": True})
    assert response.status_code == 422


def test_vrm_avatar_asset_is_served():
    response = client.get("/avatar-assets/stable/AvatarSample_C.vrm")
    assert response.status_code == 200
    assert response.content[:4] == b"glTF"
    assert len(response.content) > 1_000_000
