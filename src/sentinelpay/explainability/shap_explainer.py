import json
import pathlib
import time
import joblib
import numpy as np
import polars as pl
import shap


class RealTimeFraudExplainer:
    """
    Computes exact local TreeSHAP explanations for individual flagged transactions,
    returning the top human-readable risk factors pushing the fraud score up.
    """

    def __init__(self, artifacts_dir: pathlib.Path):
        model_path = artifacts_dir / "champion_lgbm.joblib"
        meta_path = artifacts_dir / "model_metadata.json"

        self.model = joblib.load(model_path)
        with open(meta_path, "r") as f:
            self.metadata = json.load(f)

        self.feature_cols: list[str] = self.metadata["feature_cols"]
        # Initialize C++ TreeExplainer once at startup
        self.explainer = shap.TreeExplainer(self.model)

    def explain_single_transaction(
        self, x_row: np.ndarray, top_k: int = 3
    ) -> list[dict[str, float | str]]:
        """
        Accepts a (1, n_features) float32 array and returns the top_k features
        contributing most strongly toward fraud (positive SHAP values).
        """
        if x_row.ndim == 1:
            x_row = x_row.reshape(1, -1)

        shap_values = self.explainer.shap_values(x_row)
        # Handle binary classification output shapes across SHAP versions
        if isinstance(shap_values, list):
            sv = np.asarray(shap_values[1])[0]
        elif shap_values.ndim == 3:
            sv = shap_values[0, :, 1]
        else:
            sv = shap_values[0]

        # Sort features by highest positive contribution toward fraud
        top_indices = np.argsort(sv)[::-1][:top_k]

        reasons = []
        for idx in top_indices:
            feat_name = self.feature_cols[int(idx)]
            feat_val = float(x_row[0, int(idx)])
            impact = float(sv[int(idx)])
            reasons.append(
                {
                    "feature": feat_name,
                    "feature_value": round(feat_val, 4),
                    "shap_log_odds_impact": round(impact, 4),
                }
            )
        return reasons


if __name__ == "__main__":
    from sentinelpay.training.train_model import prepare_matrices

    artifacts_dir = pathlib.Path("artifacts")
    processed_dir = pathlib.Path("data/processed")

    print("Loading test split to demonstrate real-time TreeSHAP on a caught fraud transaction...")
    train_df = pl.read_parquet(processed_dir / "train.parquet", n_rows=5000)
    val_df = pl.read_parquet(processed_dir / "val.parquet", n_rows=1000)
    test_df = pl.read_parquet(processed_dir / "test.parquet", n_rows=5000)

    _, _, _, _, X_test, y_test, _, _ = prepare_matrices(train_df, val_df, test_df)

    explainer = RealTimeFraudExplainer(artifacts_dir)
    fraud_indices = np.where(y_test == 1)[0]
    sample_idx = int(fraud_indices[0])
    sample_tx = X_test[sample_idx : sample_idx + 1]

    t0 = time.perf_counter()
    reasons = explainer.explain_single_transaction(sample_tx, top_k=3)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    print(f"\nSHAP Explanation generated in {elapsed_ms:.2f} ms for Fraud Transaction #{sample_idx}:")
    for rank, r in enumerate(reasons, start=1):
        print(
            f"  #{rank}: {r['feature']} = {r['feature_value']} "
            f"(SHAP Impact: +{r['shap_log_odds_impact']:.4f})"
        )