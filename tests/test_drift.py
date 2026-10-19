import pathlib
import numpy as np
import polars as pl
from sentinelpay.monitoring.drift_detector import run_drift_analysis


def test_drift_detector_flags_attack_wave(tmp_path: pathlib.Path):
    rng = np.random.default_rng(42)
    n = 1000
    ref_df = pl.DataFrame(
        {
            "TransactionAmt": rng.lognormal(3.5, 0.8, size=n),
            "tx_amt_log": rng.normal(3.5, 0.8, size=n),
            "card_tx_count_1h": rng.integers(1, 4, size=n).astype(float),
            "card_amt_sum_1h": rng.lognormal(4.0, 0.8, size=n),
            "card_tx_count_24h": rng.integers(1, 10, size=n).astype(float),
            "card_amt_mean_24h": rng.lognormal(3.5, 0.8, size=n),
            "amt_to_mean_24h_ratio": rng.uniform(0.5, 1.5, size=n),
            "null_feature_count": rng.integers(0, 5, size=n).astype(float),
            "isFraud": rng.choice([0, 1], size=n, p=[0.96, 0.04]),
        }
    )

    # 1. Identical distributions should NOT trigger retraining
    no_drift = run_drift_analysis(ref_df, ref_df, tmp_path, simulate_attack_wave=False)
    assert no_drift["trigger_retraining"] is False

    # 2. Simulated attack wave MUST trigger retraining
    attack_drift = run_drift_analysis(ref_df, ref_df, tmp_path, simulate_attack_wave=True)
    assert attack_drift["trigger_retraining"] is True
    assert "TransactionAmt" in attack_drift["drifted_features"]