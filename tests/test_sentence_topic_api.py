import pytest
from fastapi.testclient import TestClient

from api.server import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_sentence_topic_rewrite_basic_schema(client):
    """
    /sentence_topic_rewrite 엔드포인트가 기본 스키마를 만족하는지 검증한다.
    데이터 품질이 아니라 구조/타입만 체크한다.
    """
    payload = {
        "paragraph": "환자는 고혈압 진단을 받았다. 약물 치료를 시작했다.",
        "lexical_overrides": {},
        "metric_hint": None,
        "options": {},
    }

    resp = client.post("/sentence_topic_rewrite", json=payload)
    assert resp.status_code == 200

    data = resp.json()

    # 필수 키 존재 여부
    for key in ["sentences", "topics", "metric_keys", "replacements", "final_text", "stats"]:
        assert key in data

    # 타입/길이 기본 검증
    sentences = data["sentences"]
    topics = data["topics"]
    metric_keys = data["metric_keys"]
    stats = data["stats"]

    assert isinstance(sentences, list)
    assert len(sentences) > 0
    assert isinstance(topics, list)
    assert len(topics) == len(sentences)
    assert isinstance(metric_keys, list)
    assert len(metric_keys) == len(sentences)

    # final_text는 문자열이어야 함
    assert isinstance(data["final_text"], str)
    assert len(data["final_text"]) > 0

    # stats 필드 기본 구조
    assert "total_tokens" in stats
    assert "replaced_tokens" in stats
    assert "replacement_ratio" in stats

    assert stats["total_tokens"] >= 0
    assert 0.0 <= stats["replacement_ratio"] <= 1.0


