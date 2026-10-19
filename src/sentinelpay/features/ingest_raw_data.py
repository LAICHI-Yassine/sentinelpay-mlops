import pathlib
from datetime import datetime
import polars as pl

# We select the highest-signal, human-interpretable columns from IEEE-CIS
# so our SHAP explanations and Redis feature store are 100% meaningful in production.
TRANSACTION_COLS = [
    "TransactionID",
    "isFraud",
    "TransactionDT",
    "TransactionAmt",
    "ProductCD",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "addr2",
    "dist1",
    "P_emaildomain",
    "R_emaildomain",
    "C1",
    "C2",
    "C13",
    "C14",
    "D1",
    "D4",
    "D10",
    "D15",
    "M4",
    "M6",
]

IDENTITY_COLS = [
    "TransactionID",
    "DeviceType",
    "DeviceInfo",
    "id_01",
    "id_02",
    "id_05",
    "id_06",
    "id_30",
    "id_31",
]


def ingest_and_merge_ieee_data(raw_dir: pathlib.Path, output_path: pathlib.Path) -> pl.DataFrame:
    tx_path = raw_dir / "train_transaction.csv"
    id_path = raw_dir / "train_identity.csv"

    if not tx_path.exists() or not id_path.exists():
        raise FileNotFoundError(
            f"Missing raw IEEE-CIS files in {raw_dir}. "
            "Make sure train_transaction.csv and train_identity.csv are downloaded."
        )

    print("1. Reading raw IEEE-CIS transaction & identity CSVs with Polars...")
    df_tx = pl.read_csv(tx_path, columns=TRANSACTION_COLS, null_values=["", "NA", "nan"])
    df_id = pl.read_csv(id_path, columns=IDENTITY_COLS, null_values=["", "NA", "nan"])

    print("2. Joining transaction and identity tables on TransactionID...")
    df = df_tx.join(df_id, on="TransactionID", how="left")

    # Reference epoch start for TransactionDT (seconds elapsed from reference point)
    base_epoch = int(datetime(2025, 11, 1, 0, 0, 0).timestamp())

    print("3. Standardizing timestamps and constructing composite card_uid...")
    df = (
        df.with_columns(
            [
                # Convert TransactionDT (seconds offset) into a proper Datetime column
                pl.from_epoch(pl.lit(base_epoch) + pl.col("TransactionDT"), time_unit="s").alias(
                    "timestamp"
                ),
                # Create a synthetic composite cardholder ID from card1 + card2 + addr1
                (
                    pl.col("card1").cast(pl.Utf8).fill_null("unk")
                    + "_"
                    + pl.col("card2").cast(pl.Utf8).fill_null("unk")
                    + "_"
                    + pl.col("addr1").cast(pl.Utf8).fill_null("unk")
                ).alias("card_uid"),
            ]
        )
        .sort("timestamp")
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path, compression="zstd")

    fraud_rate = df["isFraud"].mean() * 100
    print(
        f"Success! Saved real-world dataset to {output_path}\n"
        f"Total Transactions: {df.height:,} | Features: {df.width} | Fraud Rate: {fraud_rate:.2f}%"
    )
    return df


if __name__ == "__main__":
    # Remove the old synthetic script if it still exists
    raw_directory = pathlib.Path("data/raw")
    output_parquet = pathlib.Path("data/raw/ieee_transactions.parquet")
    ingest_and_merge_ieee_data(raw_directory, output_parquet)