from fastapi.testclient import TestClient

from tetraknowledge.api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_validation_rejects_invalid_top_k():
    response = client.post("/knowledge-bases/not-a-uuid/query", json={"question": "x", "top_k": 99})
    assert response.status_code == 422


def test_validation_rejects_unknown_pipeline_version():
    response = client.post(
        "/knowledge-bases/not-a-uuid/query",
        json={"question": "¿Qué ocurrió?", "pipeline_version": 3},
    )
    assert response.status_code == 422
