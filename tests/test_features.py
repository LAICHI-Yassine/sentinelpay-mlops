from datetime import datetime, timedelta
import polars as pl
from sentinelpay.features.pipeline import build_features, chronological_split


def _make_mock_transactions() -> pl.DataFrame:
    base_time = datetime(2026, 1, 1, 12, 0, 0)
    return pl.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4],
            "isFraud": [0, 0, 1, 0],
            "TransactionDT": [0, 600, 1800, 90000],
            "timestamp": [
                base_time,
                base_time + timedelta(minutes=10),
                base_time + timedelta(minutes=30),
                base_time + timedelta(hours=25),
            ],
            "TransactionAmt": [100.0, 200.0, 600.0, 50.0],
            "ProductCD": ["W", "W", "C", None],
            "card4": ["visa", "visa", "visa", "mastercard"],
            "card6": ["debit", "debit", "credit", "debit"],
            "P_emaildomain": ["gmail.com", "gmail.com", "gmail.com", "yahoo.com"],
            "R_emaildomain": ["gmail.com", "gmail.com", "anon.net", None],
            "DeviceType": ["desktop", "desktop", "mobile", None],
            "M4": ["M0", "M0", "M2", None],
            "M6": ["T", "T", "F", None],
            "card_uid": ["user_A", "user_A", "user_A", "user_A"],
        }
    )


def test_rolling_velocity_has_no_future_leakage():
    df = _make_mock_transactions()
    feat_df = build_features(df)

    # Transaction 1 (at t=0): 1st tx in 1h window -> count must be 1, not 3!
    # If it were 3, future transactions at t+10m and t+30m would be leaking backward.
    counts_1h = feat_df["card_tx_count_1h"].to_list()
    assert counts_1h == [1, 2, 3, 1]

    # At t=25 hours (4th transaction), the previous 3 transactions are >24h old
    counts_24h = feat_df["card_tx_count_24h"].to_list()
    assert counts_24h == [1, 2, 3, 1]


def test_chronological_split_strictly_ordered():
    df = _make_mock_transactions()
    train_df, val_df, test_df = chronological_split(df, train_ratio=0.5, val_ratio=0.25)

    assert train_df.height == 2
    assert val_df.height == 1
    assert test_df.height == 1
    assert train_df["timestamp"].max() < val_df["timestamp"].min()
    assert val_df["timestamp"].max() < test_df["timestamp"].min()