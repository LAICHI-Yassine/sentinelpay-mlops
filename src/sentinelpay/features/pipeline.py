import pathlib
import numpy as np
import polars as pl

CATEGORICAL_COLS = [
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceType",
    "M4",
    "M6",
]


def build_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Computes leak-free temporal velocity features, amount ratios, and cyclical
    time encodings using Polars native rolling windows.
    """
    # 1. Ensure strictly sorted timestamps for Polars rolling operations
    df = df.sort("timestamp")

    # 2. Compute backward-looking rolling velocity & spending habits per card_uid
    rolling_1h = (
        df.rolling(index_column="timestamp", period="1h", group_by="card_uid")
        .agg(
            [
                pl.len().alias("card_tx_count_1h"),
                pl.col("TransactionAmt").sum().alias("card_amt_sum_1h"),
            ]
        )
        .drop(["card_uid", "timestamp"])
    )

    rolling_24h = (
        df.rolling(index_column="timestamp", period="24h", group_by="card_uid")
        .agg(
            [
                pl.len().alias("card_tx_count_24h"),
                pl.col("TransactionAmt").mean().alias("card_amt_mean_24h"),
                pl.col("TransactionAmt").std().fill_null(0.0).alias("card_amt_std_24h"),
            ]
        )
        .drop(["card_uid", "timestamp"])
    )

    df = pl.concat([df, rolling_1h, rolling_24h], how="horizontal")

    # 3. Engineer high-signal domain features (ratios, cyclical time, email match)
    hour = pl.col("timestamp").dt.hour().cast(pl.Float64)
    day_of_week = pl.col("timestamp").dt.weekday().cast(pl.Float64)

    df = df.with_columns(
        [
            # Ratio of current transaction amount to user's 24h average
            (pl.col("TransactionAmt") / (pl.col("card_amt_mean_24h") + 1.0)).alias(
                "amt_to_mean_24h_ratio"
            ),
            # Log transform of skewed transaction amount
            pl.col("TransactionAmt").log1p().alias("tx_amt_log"),
            # Cyclical encoding of Hour of Day (so 23:00 and 00:00 are close in feature space)
            (hour * (2.0 * np.pi / 24.0)).sin().alias("hour_sin"),
            (hour * (2.0 * np.pi / 24.0)).cos().alias("hour_cos"),
            (day_of_week * (2.0 * np.pi / 7.0)).sin().alias("dow_sin"),
            (day_of_week * (2.0 * np.pi / 7.0)).cos().alias("dow_cos"),
            # Flag if purchaser and recipient email domains differ
            (
                pl.col("P_emaildomain").fill_null("missing")
                != pl.col("R_emaildomain").fill_null("missing")
            )
            .cast(pl.Int8)
            .alias("email_domain_mismatch"),
            # Missing value count per row (in fraud detection, missingness itself is a strong signal!)
            pl.sum_horizontal(pl.all().is_null()).cast(pl.Int16).alias("null_feature_count"),
        ]
    )

    # 4. Clean categorical columns for gradient boosting
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df = df.with_columns(
                pl.col(col).cast(pl.Utf8).fill_null("MISSING").alias(col)
            )

    return df


def chronological_split(
    df: pl.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.15
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Splits dataset strictly chronologically (Out-of-Time split) to prevent future data leakage.
    """
    df = df.sort("timestamp")
    n = df.height
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.slice(0, train_end)
    val_df = df.slice(train_end, val_end - train_end)
    test_df = df.slice(val_end, n - val_end)

    return train_df, val_df, test_df


def run_feature_pipeline(
    input_parquet: pathlib.Path, output_dir: pathlib.Path
) -> None:
    print(f"Loading ingested data from {input_parquet}...")
    df_raw = pl.read_parquet(input_parquet)

    print("Building causal rolling velocity & domain features with Polars...")
    df_features = build_features(df_raw)

    print("Performing strict Out-of-Time chronological split (70% / 15% / 15%)...")
    train_df, val_df, test_df = chronological_split(df_features)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.write_parquet(output_dir / "train.parquet", compression="zstd")
    val_df.write_parquet(output_dir / "val.parquet", compression="zstd")
    test_df.write_parquet(output_dir / "test.parquet", compression="zstd")

    print(
        f"Feature pipeline complete!\n"
        f"  Train: {train_df.height:,} rows (Fraud: {train_df['isFraud'].mean()*100:.2f}%)\n"
        f"  Val  : {val_df.height:,} rows (Fraud: {val_df['isFraud'].mean()*100:.2f}%)\n"
        f"  Test : {test_df.height:,} rows (Fraud: {test_df['isFraud'].mean()*100:.2f}%)\n"
        f"  Total Features: {train_df.width}"
    )


if __name__ == "__main__":
    raw_parquet = pathlib.Path("data/raw/ieee_transactions.parquet")
    processed_dir = pathlib.Path("data/processed")
    run_feature_pipeline(raw_parquet, processed_dir)