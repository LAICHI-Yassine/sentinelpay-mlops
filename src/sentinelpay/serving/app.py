import json
import pathlib
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import joblib
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

from sentinelpay.explainability.shap_explainer import RealTimeFraudExplainer
from sentinelpay.serving.feature_store import OnlineVelocityStore

ARTIFACTS_DIR = pathlib.Path("artifacts")


class TransactionRequest(BaseModel):
    TransactionID: int = Field(..., example=3577539)
    TransactionAmt: float = Field(..., gt=0, example=485.50)
    ProductCD: str = Field(default="W", example="C")
    card1: int = Field(..., example=13926)
    card2: float | None = Field(default=150.0, example=500.0)
    card3: float | None = Field(default=150.0, example=185.0)
    card4: str | None = Field(default="visa", example="visa")
    card5: float | None = Field(default=226.0, example=137.0)
    card6: str | None = Field(default="debit", example="credit")
    addr1: float | None = Field(default=315.0, example=299.0)
    addr2: float | None = Field(default=87.0, example=87.0)
    dist1: float | None = Field(default=None, example=120.0)
    P_emaildomain: str | None = Field(default="gmail.com", example="gmail.com")
    R_emaildomain: str | None = Field(default=None, example="anonymous.de")
    DeviceType: str | None = Field(default="desktop", example="mobile")
    DeviceInfo: str | None = Field(default="Windows", example="SM-G960U")


class ShapReason(BaseModel):
    feature: str
    feature_value: float
    shap_log_odds_impact: float


class ScoringResponse(BaseModel):
    transaction_id: int
    fraud_probability: float
    decision: str
    threshold_used: float
    inference_latency_ms: float
    feature_store_backend: str
    top_risk_factors: list[ShapReason]


class ModelEngine:
    def __init__(self, artifacts_dir: pathlib.Path):
        with open(artifacts_dir / "model_metadata.json") as f:
            self.metadata = json.load(f)

        self.feature_cols: list[str] = self.metadata["feature_cols"]
        self.cat_cols: list[str] = self.metadata["categorical_cols"]
        self.threshold: float = float(self.metadata["optimal_threshold"])

        self.encoder = joblib.load(artifacts_dir / "ordinal_encoder.joblib")

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        self.ort_session = ort.InferenceSession(
            str(artifacts_dir / "champion_lgbm.onnx"),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.ort_session.get_inputs()[0].name
        self.explainer = RealTimeFraudExplainer(artifacts_dir)
        self.feature_store = OnlineVelocityStore()

    def build_feature_vector(self, req: TransactionRequest) -> tuple[np.ndarray, dict]:
        now_dt = datetime.now(timezone.utc)
        card_uid = f"{req.card1}_{req.card2 or 'unk'}_{req.addr1 or 'unk'}"

        # 1. Fetch live velocity features from Redis / Online Store
        vel = self.feature_store.get_and_update_card_velocity(
            card_uid=card_uid, tx_amount=req.TransactionAmt, current_ts=now_dt.timestamp()
        )

        # 2. Compute temporal & domain features matching our Polars pipeline
        hour = float(now_dt.hour)
        dow = float(now_dt.weekday() + 1)
        p_email = req.P_emaildomain or "missing"
        r_email = req.R_emaildomain or "missing"

        raw_dict = req.model_dump()
        raw_dict.update(vel)
        raw_dict.update(
            {
                "tx_amt_log": float(np.log1p(req.TransactionAmt)),
                "hour_sin": float(np.sin(hour * (2.0 * np.pi / 24.0))),
                "hour_cos": float(np.cos(hour * (2.0 * np.pi / 24.0))),
                "dow_sin": float(np.sin(dow * (2.0 * np.pi / 7.0))),
                "dow_cos": float(np.cos(dow * (2.0 * np.pi / 7.0))),
                "email_domain_mismatch": 1.0 if p_email != r_email else 0.0,
                "null_feature_count": float(
                    sum(1 for v in req.model_dump().values() if v is None)
                ),
            }
        )

        # Encode categoricals using fitted OrdinalEncoder
        if self.cat_cols:
            cat_vals = [[str(raw_dict.get(c) or "MISSING") for c in self.cat_cols]]
            encoded_cats = self.encoder.transform(cat_vals)[0]
            for c, val in zip(self.cat_cols, encoded_cats):
                raw_dict[c] = float(val)

        # Assemble exact float32 vector in training column order
        vec = np.array(
            [
                float(raw_dict[c]) if raw_dict.get(c) is not None else -999.0
                for c in self.feature_cols
            ],
            dtype=np.float32,
        ).reshape(1, -1)

        return vec, vel


engine: ModelEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    if (ARTIFACTS_DIR / "champion_lgbm.onnx").exists():
        engine = ModelEngine(ARTIFACTS_DIR)
    yield


app = FastAPI(
    title="SentinelPay Real-Time Fraud & Risk Engine",
    version="1.0.0",
    description="High-throughput ONNX Runtime + Redis Feature Store + TreeSHAP Explainability API",
    lifespan=lifespan,
)

# Expose /metrics for Prometheus & Grafana
Instrumentator().instrument(app).expose(app)


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": engine is not None,
        "feature_store": "redis" if (engine and engine.feature_store.using_redis) else "in_memory_fallback",
    }


@app.post("/v1/score-transaction", response_model=ScoringResponse)
async def score_transaction(req: TransactionRequest, explain_all: bool = False):
    t0 = time.perf_counter()
    assert engine is not None, "Model artifacts not loaded!"

    x_vec, _ = engine.build_feature_vector(req)
    ort_out = engine.ort_session.run(None, {engine.input_name: x_vec})
    prob = float(ort_out[1][0][1])

    if prob >= 0.50:
        decision = "BLOCK"
    elif prob >= engine.threshold:
        decision = "REVIEW"
    else:
        decision = "APPROVE"

    # Compute TreeSHAP reason codes if flagged (or if caller requests explain_all=True)
    reasons = []
    if decision in ("BLOCK", "REVIEW") or explain_all:
        raw_reasons = engine.explainer.explain_single_transaction(x_vec, top_k=3)
        reasons = [ShapReason(**r) for r in raw_reasons]

    latency_ms = (time.perf_counter() - t0) * 1000.0

    return ScoringResponse(
        transaction_id=req.TransactionID,
        fraud_probability=round(prob, 4),
        decision=decision,
        threshold_used=round(engine.threshold, 4),
        inference_latency_ms=round(latency_ms, 3),
        feature_store_backend="redis" if engine.feature_store.using_redis else "in_memory_fallback",
        top_risk_factors=reasons,
    )