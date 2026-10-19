import json
import pathlib
import numpy as np
import onnxruntime as ort
import pytest
from sentinelpay.explainability.shap_explainer import RealTimeFraudExplainer

ARTIFACTS_DIR = pathlib.Path("artifacts")


@pytest.mark.skipif(
    not (ARTIFACTS_DIR / "champion_lgbm.onnx").exists(),
    reason="ONNX artifact not yet generated",
)
def test_onnx_inference_and_shap_reasons():
    with open(ARTIFACTS_DIR / "model_metadata.json") as f:
        meta = json.load(f)

    n_features = len(meta["feature_cols"])
    dummy_tx = np.zeros((1, n_features), dtype=np.float32)

    # 1. Test ONNX probability output is valid [0, 1]
    sess = ort.InferenceSession(
        str(ARTIFACTS_DIR / "champion_lgbm.onnx"), providers=["CPUExecutionProvider"]
    )
    input_name = sess.get_inputs()[0].name
    out = sess.run(None, {input_name: dummy_tx})
    prob = float(out[1][0][1])
    assert 0.0 <= prob <= 1.0

    # 2. Test TreeSHAP returns top 3 structured reason codes
    explainer = RealTimeFraudExplainer(ARTIFACTS_DIR)
    reasons = explainer.explain_single_transaction(dummy_tx, top_k=3)
    assert len(reasons) == 3
    assert "feature" in reasons[0]
    assert "shap_log_odds_impact" in reasons[0]