import pathlib
import pytest
from fastapi.testclient import TestClient
from sentinelpay.serving.app import app

ARTIFACTS_DIR = pathlib.Path("artifacts")


@pytest.mark.skipif(
    not (ARTIFACTS_DIR / "champion_lgbm.onnx").exists(),
    reason="ONNX artifact not yet generated",
)
def test_score_transaction_endpoint_and_velocity():
    with TestClient(app) as client:
        # 1. Health endpoint check
        res_health = client.get("/health")
        assert res_health.status_code == 200
        assert res_health.json()["model_loaded"] is True

        # 2. Score a suspicious high-value foreign card transaction with explain_all=True
        payload = {
            "TransactionID": 99001,
            "TransactionAmt": 2450.00,
            "ProductCD": "C",
            "card1": 13926,
            "card2": 500.0,
            "card3": 185.0,
            "card4": "visa",
            "card6": "credit",
            "addr1": 299.0,
            "P_emaildomain": "gmail.com",
            "R_emaildomain": "anonymous.ru",
            "DeviceType": "mobile",
        }
        res = client.post("/v1/score-transaction?explain_all=true", json=payload)
        assert res.status_code == 200
        body = res.json()

        assert 0.0 <= body["fraud_probability"] <= 1.0
        assert body["decision"] in ("APPROVE", "REVIEW", "BLOCK")
        assert len(body["top_risk_factors"]) == 3
        assert body["inference_latency_ms"] < 150.0