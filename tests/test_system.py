"""
System/integration checks for the crop/agri FastAPI.

These tests validate the live contract used by the React dashboard:
  - /predict-yield
  - /classify-yield (artifact optional; has fallback)
  - /forecast (artifact optional; has fallback)
  - /cluster
  - /recommend
"""

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_root_health():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json().get("message") == "ML API running"


def test_metrics_endpoint():
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.json()
    assert "generated_at" in body
    assert "artifacts" in body
    assert "yield" in body["artifacts"]
    assert "classifier" in body["artifacts"]
    assert "recommender" in body["artifacts"]
    assert "clustering" in body["artifacts"]
    assert "rainfall_forecast" in body["artifacts"]


def test_metrics_history_endpoint():
    r = client.get("/metrics/history")
    assert r.status_code == 200
    body = r.json()
    assert "count" in body
    assert "history" in body
    assert isinstance(body["history"], list)


def test_predict_yield():
    payload = {"features": {"rainfall": 200, "temperature": 22, "N": 90, "P": 40, "K": 43}}
    r = client.post("/predict-yield", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "prediction" in body


def test_classify_yield():
    payload = {"features": {"rainfall": 200, "temperature": 22, "N": 90, "P": 40, "K": 43}}
    r = client.post("/classify-yield", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body.get("prediction") in {"low", "medium", "high"}
    assert "source" in body


def test_forecast_fallback():
    r = client.post("/forecast", json={"periods": 6})
    assert r.status_code == 200
    body = r.json()
    assert body.get("periods") == 6
    assert isinstance(body.get("forecast"), list)
    assert len(body["forecast"]) == 6


def test_cluster():
    payload = {"samples": [{"rainfall": 200, "temperature": 22, "N": 90, "P": 40, "K": 43}]}
    r = client.post("/cluster", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "clusters" in body


def test_recommend():
    payload = {"N": 90, "P": 42, "K": 43, "rainfall": 200, "temperature": 21}
    r = client.post("/recommend", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "best_crop" in body
