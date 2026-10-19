import pathlib
import polars as pl
from prefect import flow, task

from sentinelpay.features.pipeline import run_feature_pipeline
from sentinelpay.monitoring.drift_detector import run_drift_analysis
from sentinelpay.serving.export_onnx import export_and_benchmark_onnx
from sentinelpay.training.train_model import run_training_pipeline


@task(name="Build_Polars_Causal_Features", retries=1)
def feature_engineering_task(raw_parquet: pathlib.Path, processed_dir: pathlib.Path) -> None:
    run_feature_pipeline(raw_parquet, processed_dir)


@task(name="Check_Data_Drift_PSI_KS")
def drift_detection_task(processed_dir: pathlib.Path, reports_dir: pathlib.Path) -> dict:
    ref_df = pl.read_parquet(processed_dir / "val.parquet")
    cur_df = pl.read_parquet(processed_dir / "test.parquet")
    return run_drift_analysis(ref_df, cur_df, reports_dir, simulate_attack_wave=False)


@task(name="Train_Optuna_LightGBM_MLflow")
def model_training_task(
    processed_dir: pathlib.Path, artifacts_dir: pathlib.Path, n_trials: int
) -> None:
    run_training_pipeline(processed_dir, artifacts_dir, n_trials=n_trials)


@task(name="Compile_and_Benchmark_ONNX")
def onnx_export_task(artifacts_dir: pathlib.Path) -> None:
    export_and_benchmark_onnx(artifacts_dir)


@flow(name="SentinelPay_End_To_End_MLOps_Pipeline", log_prints=True)
def sentinelpay_mlops_flow(
    run_features: bool = False,
    n_optuna_trials: int = 5,
) -> dict:
    raw_parquet = pathlib.Path("data/raw/ieee_transactions.parquet")
    processed_dir = pathlib.Path("data/processed")
    artifacts_dir = pathlib.Path("artifacts")
    reports_dir = artifacts_dir / "reports"

    if run_features or not (processed_dir / "train.parquet").exists():
        feature_engineering_task(raw_parquet, processed_dir)

    drift_summary = drift_detection_task(processed_dir, reports_dir)
    print(f"Drift Check Complete -> Drifted features: {drift_summary['drifted_features']}")

    model_training_task(processed_dir, artifacts_dir, n_trials=n_optuna_trials)
    onnx_export_task(artifacts_dir)

    return drift_summary


if __name__ == "__main__":
    sentinelpay_mlops_flow(run_features=False, n_optuna_trials=5)