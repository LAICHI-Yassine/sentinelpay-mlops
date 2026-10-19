# SentinelPay-MLOps: Real-Time FinTech Fraud Detection & Risk Engine

![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)
![Polars](https://img.shields.io/badge/Polars-Data_Engineering-orange.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-Optuna-green.svg)
![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-0.10ms_Inference-purple.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Redis_Feature_Store-009688.svg)
![MLOps](https://img.shields.io/badge/MLOps-MLflow_%7C_Prefect_%7C_DVC-red.svg)

An end-to-end, production-grade **Real-Time Payment Fraud Detection & MLOps Engine** trained and evaluated on **590,540 real e-commerce transactions** from the **IEEE-CIS Fraud Detection** benchmark.

Designed for strict FinTech production constraints: **zero temporal data leakage**, **sub-millisecond C++ inference via ONNX Runtime**, **real-time Redis sliding-window velocity state**, **local TreeSHAP compliance explanations**, and **automated PSI/KS drift monitoring**.

---

## System Architecture

```text
[Incoming Transaction JSON]
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│ FastAPI Async Microservice (/v1/score-transaction)      │
│  ├─► 1. Redis Online Feature Store (<1ms 1h/24h state)  │
│  ├─► 2. ONNX Runtime C++ Session (0.103ms mean latency) │
│  ├─► 3. TreeSHAP Explainer (Top-3 Risk Reason Codes)    │
│  └─► 4. Prometheus /metrics Telemetry                   │
└─────────────────────────────────────────────────────────┘
         ▲
         │ (Model Artifacts & Drift Triggers)
┌─────────────────────────────────────────────────────────┐
│ Offline MLOps Lifecycle (Prefect + MLflow + DVC)        │
│  ├─► Polars Causal Rolling Feature Engineering          │
│  ├─► Strict Out-of-Time Split (70% Train / 15% Val / 15% Test)
│  ├─► Optuna Bayesian Tuning + MLflow Model Registry     │
│  └─► PSI & KS Covariate Drift Detector                  │
└─────────────────────────────────────────────────────────┘
```

---

## Verified Production Benchmarks

### 1. Single-Transaction Inference Latency (`Batch Size = 1`, 1,000 Runs)
Compiling the trained **LightGBM** model into an **ONNX Runtime** C++ execution graph achieved a **15.1x speedup** with numerical parity down to `3.85e-08`:

| Inference Engine | Mean Latency | p99 Latency | Speedup |
| :--- | :---: | :---: | :---: |
| Standard Python `LightGBM.predict_proba()` | `1.550 ms` | `6.478 ms` | `1.0x` |
| **ONNX Runtime (`CPUExecutionProvider`)** | **`0.103 ms`** | **`0.205 ms`** | **`15.1x faster`** |

### 2. Out-of-Time Fraud Detection Performance (590,540 Transactions)
Evaluated on a strict chronological split (zero future leakage across `card_uid` rolling windows):
* **Validation ROC-AUC:** `0.9133` | **Validation PR-AUC:** `0.5586`
* **Out-of-Time Test ROC-AUC:** `0.8899` | **Test PR-AUC:** `0.4853` (**14x lift** over `3.48%` baseline fraud rate)
* **Test F2-Score:** `0.5063` (at optimal F2 probability threshold `0.1526`)
* **Fraud Recall @ 1% False Positive Rate (FPR):** `41.62%`

### 3. Real-Time Explainability (TreeSHAP)
Every transaction flagged for `REVIEW` or `BLOCK` returns the top 3 mathematical risk factors in `~26 ms`:

```json
{
  "transaction_id": 99001,
  "fraud_probability": 0.6842,
  "decision": "BLOCK",
  "threshold_used": 0.1526,
  "inference_latency_ms": 26.41,
  "feature_store_backend": "redis",
  "top_risk_factors": [
    {"feature": "card3", "feature_value": 185.0, "shap_log_odds_impact": 0.5669},
    {"feature": "card6", "feature_value": 0.0, "shap_log_odds_impact": 0.5235},
    {"feature": "card2", "feature_value": 500.0, "shap_log_odds_impact": 0.3552}
  ]
}
```

---

## Repository Structure

```text
sentinelpay-mlops/
├── .github/workflows/ci_cd.yml        # GitHub Actions CI pipeline (Pytest)
├── src/sentinelpay/
│   ├── features/
│   │   ├── ingest_raw_data.py         # Polars ingestion & table join (IEEE-CIS)
│   │   └── pipeline.py                # Causal 1h/24h rolling windows & Out-of-Time split
│   ├── training/
│   │   └── train_model.py             # Cost-sensitive LightGBM + Optuna + MLflow Registry
│   ├── explainability/
│   │   └── shap_explainer.py          # Real-time local TreeSHAP reason code engine
│   ├── serving/
│   │   ├── export_onnx.py             # LightGBM -> ONNX compiler & latency benchmark
│   │   ├── feature_store.py           # Redis sliding-window velocity store (with fail-safe)
│   │   └── app.py                     # Async FastAPI + Prometheus telemetry microservice
│   └── monitoring/
│       └── drift_detector.py          # Banking PSI & Kolmogorov-Smirnov drift monitor
├── flows/training_pipeline.py         # Prefect end-to-end orchestration flow
├── tests/                             # 5 automated Pytest suites (leakage, ONNX, API, drift)
├── Dockerfile & docker-compose.yml    # Containerized API + Redis deployment
└── pyproject.toml
```

---

## Quickstart

```bash
# 1. Install package and dev dependencies
uv pip install -e ".[dev]" scipy

# 2. Run the full automated test suite
pytest -v

# 3. Execute the Prefect MLOps pipeline (Drift check -> Optuna + MLflow -> ONNX export)
python flows/training_pipeline.py

# 4. Launch the FastAPI + Redis stack via Docker Compose
docker compose up --build
```

---

## Author

**Yassine LAICHI** — AI & Machine Learning Engineer (MLOps & GenAI)  
[GitHub](https://github.com/LAICHI-Yassine) | [LinkedIn](https://linkedin.com/in/Yassine-LAICHI)