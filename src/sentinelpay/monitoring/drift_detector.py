import json
import pathlib
import numpy as np
import polars as pl
from scipy.stats import ks_2samp

MONITORED_FEATURES = [
    "TransactionAmt",
    "tx_amt_log",
    "card_tx_count_1h",
    "card_amt_sum_1h",
    "card_tx_count_24h",
    "card_amt_mean_24h",
    "amt_to_mean_24h_ratio",
    "null_feature_count",
]


def calculate_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """
    Computes Population Stability Index (PSI) — the banking & FinTech standard
    for measuring feature distribution shift over time.
      PSI < 0.10 : No significant drift
      PSI >= 0.10: Moderate drift
      PSI >= 0.20: Severe drift (Trigger automated model retraining)
    """
    ref_clean = reference[~np.isnan(reference)]
    cur_clean = current[~np.isnan(current)]

    quantiles = np.linspace(0, 100, bins + 1)
    bin_edges = np.unique(np.percentile(ref_clean, quantiles))
    if len(bin_edges) < 3:
        return 0.0

    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref_clean, bins=bin_edges)
    cur_counts, _ = np.histogram(cur_clean, bins=bin_edges)

    ref_pct = np.clip(ref_counts / max(len(ref_clean), 1), 1e-4, 1.0)
    cur_pct = np.clip(cur_counts / max(len(cur_clean), 1), 1e-4, 1.0)

    psi_value = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi_value)


def run_drift_analysis(
    reference_df: pl.DataFrame,
    current_df: pl.DataFrame,
    output_dir: pathlib.Path,
    simulate_attack_wave: bool = False,
) -> dict:
    """
    Evaluates feature & target drift between Reference and Production windows.
    """
    if simulate_attack_wave:
        # Simulate a sudden fraud attack wave (e.g., high-value botnet card testing)
        current_df = current_df.with_columns(
            [
                (pl.col("TransactionAmt") * 2.8).alias("TransactionAmt"),
                (pl.col("TransactionAmt") * 2.8).log1p().alias("tx_amt_log"),
                (pl.col("card_tx_count_1h") * 3.5).alias("card_tx_count_1h"),
                (pl.col("amt_to_mean_24h_ratio") * 2.4).alias("amt_to_mean_24h_ratio"),
            ]
        )

    feature_report = {}
    drifted_features = []

    for col in MONITORED_FEATURES:
        if col not in reference_df.columns or col not in current_df.columns:
            continue

        ref_arr = reference_df[col].fill_null(0.0).to_numpy().astype(np.float64)
        cur_arr = current_df[col].fill_null(0.0).to_numpy().astype(np.float64)

        psi = calculate_psi(ref_arr, cur_arr)
        ks_stat, ks_pval = ks_2samp(ref_arr, cur_arr)

        is_drifted = bool(psi >= 0.15 or (ks_stat >= 0.10 and ks_pval < 0.01))
        if is_drifted:
            drifted_features.append(col)

        feature_report[col] = {
            "psi": round(psi, 4),
            "ks_statistic": round(float(ks_stat), 4),
            "ks_p_value": float(f"{ks_pval:.2e}"),
            "drift_detected": is_drifted,
        }

    ref_fraud_rate = float(reference_df["isFraud"].mean())
    cur_fraud_rate = float(current_df["isFraud"].mean())
    drift_share = len(drifted_features) / max(len(feature_report), 1)
    retrain_required = bool(drift_share >= 0.25 or len(drifted_features) >= 2)

    summary = {
        "reference_rows": reference_df.height,
        "current_rows": current_df.height,
        "reference_fraud_rate": round(ref_fraud_rate, 4),
        "current_fraud_rate": round(cur_fraud_rate, 4),
        "drifted_features_count": len(drifted_features),
        "drifted_features": drifted_features,
        "drift_share": round(drift_share, 4),
        "trigger_retraining": retrain_required,
        "features": feature_report,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "drift_report.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    processed_dir = pathlib.Path("data/processed")
    reports_dir = pathlib.Path("artifacts/reports")

    ref_df = pl.read_parquet(processed_dir / "val.parquet")
    cur_df = pl.read_parquet(processed_dir / "test.parquet")

    print("1. Evaluating Natural Out-of-Time Drift (Val vs. Test)...")
    natural_report = run_drift_analysis(ref_df, cur_df, reports_dir, simulate_attack_wave=False)
    print(
        f"   Natural Drift -> Drifted Features: {natural_report['drifted_features']} | "
        f"Retrain Required: {natural_report['trigger_retraining']}"
    )

    print("\n2. Simulating Botnet Fraud Attack Wave (Covariate Shift)...")
    attack_report = run_drift_analysis(ref_df, cur_df, reports_dir, simulate_attack_wave=True)
    print(
        f"   Attack Wave Drift -> Drifted Features ({attack_report['drifted_features_count']}): "
        f"{attack_report['drifted_features']} | Retrain Required: {attack_report['trigger_retraining']}"
    )
    print(f"   Saved full PSI & KS drift report to {reports_dir / 'drift_report.json'}")