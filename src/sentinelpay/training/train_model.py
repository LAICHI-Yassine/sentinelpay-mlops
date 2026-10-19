import json
import pathlib
import joblib
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import optuna
import polars as pl
from sklearn.metrics import (
    average_precision_score,
    fbeta_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import OrdinalEncoder

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Columns that must NOT be used as input features (IDs, raw timestamps, target)
DROP_COLS = ["TransactionID", "isFraud", "TransactionDT", "timestamp", "card_uid"]

CATEGORICAL_COLS = [
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceType",
    "DeviceInfo",
    "id_30",
    "id_31",
    "M4",
    "M6",
]


def recall_at_fixed_fpr(y_true: np.ndarray, y_prob: np.ndarray, max_fpr: float = 0.01) -> float:
    """
    Banking industry metric: Maximum fraud Recall achievable while keeping
    False Positive Rate (legitimate customer friction) <= 1%.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    valid_idx = np.where(fpr <= max_fpr)[0]
    return float(tpr[valid_idx[-1]]) if len(valid_idx) > 0 else 0.0


def find_optimal_f2_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """
    Finds the probability threshold that maximizes F2-Score (weighing Recall 2x higher than Precision).
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # F2 formula: (1 + 2^2) * (P * R) / (4 * P + R)
    numerator = 5.0 * precisions[:-1] * recalls[:-1]
    denominator = (4.0 * precisions[:-1]) + recalls[:-1] + 1e-10
    f2_scores = numerator / denominator
    best_idx = int(np.argmax(f2_scores))
    return float(thresholds[best_idx]), float(f2_scores[best_idx])


def prepare_matrices(
    train_df: pl.DataFrame, val_df: pl.DataFrame, test_df: pl.DataFrame
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], OrdinalEncoder
]:
    feature_cols = [c for c in train_df.columns if c not in DROP_COLS]
    cat_cols = [c for c in CATEGORICAL_COLS if c in feature_cols]

    # Fit OrdinalEncoder on train categoricals, mapping unknown future categories to -1
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float32)

    def _transform_split(df: pl.DataFrame, fit: bool = False) -> tuple[np.ndarray, np.ndarray]:
        y = df["isFraud"].to_numpy().astype(np.int32)
        df_feat = df.select(feature_cols).to_pandas()
        if cat_cols:
            df_feat[cat_cols] = df_feat[cat_cols].astype(str)
            if fit:
                df_feat[cat_cols] = encoder.fit_transform(df_feat[cat_cols])
            else:
                df_feat[cat_cols] = encoder.transform(df_feat[cat_cols])
        # Fill remaining numeric NaNs with -999.0 for ONNX & Tree compatibility
        X = df_feat.fillna(-999.0).to_numpy(dtype=np.float32)
        return X, y

    X_train, y_train = _transform_split(train_df, fit=True)
    X_val, y_val = _transform_split(val_df, fit=False)
    X_test, y_test = _transform_split(test_df, fit=False)

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, encoder


def run_training_pipeline(
    processed_dir: pathlib.Path,
    artifacts_dir: pathlib.Path,
    n_trials: int = 15,
) -> None:
    print("1. Loading processed Out-of-Time splits...")
    train_df = pl.read_parquet(processed_dir / "train.parquet")
    val_df = pl.read_parquet(processed_dir / "val.parquet")
    test_df = pl.read_parquet(processed_dir / "test.parquet")

    print("2. Encoding features to float32 matrices (ONNX-ready)...")
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, encoder = prepare_matrices(
        train_df, val_df, test_df
    )

    # Calculate class imbalance ratio for cost-sensitive learning
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    imbalance_ratio = neg_count / max(pos_count, 1)

    # Configure local SQLite MLflow backend so Model Registry works out of the box
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("SentinelPay_Fraud_Detection")

    print(f"3. Starting Optuna Bayesian Hyperparameter Search ({n_trials} trials)...")

    with mlflow.start_run(run_name="Optuna_LightGBM_Study") as parent_run:

        def objective(trial: optuna.Trial) -> float:
            params = {
                "objective": "binary",
                "metric": "average_precision",
                "boosting_type": "gbdt",
                "n_estimators": 800,
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 31, 160),
                "max_depth": trial.suggest_int("max_depth", 5, 12),
                "min_child_samples": trial.suggest_int("min_child_samples", 50, 400),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                "scale_pos_weight": trial.suggest_float(
                    "scale_pos_weight", 1.0, min(imbalance_ratio, 15.0)
                ),
                "n_jobs": -1,
                "random_state": 42,
                "verbose": -1,
            }

            with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
                model = lgb.LGBMClassifier(**params)
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)],
                )

                val_probs = model.predict_proba(X_val)[:, 1]
                val_pr_auc = float(average_precision_score(y_val, val_probs))
                val_roc_auc = float(roc_auc_score(y_val, val_probs))

                mlflow.log_params(params)
                mlflow.log_metrics({"val_pr_auc": val_pr_auc, "val_roc_auc": val_roc_auc})

                print(
                    f"  [Trial {trial.number:02d}] Val PR-AUC: {val_pr_auc:.4f} | "
                    f"Val ROC-AUC: {val_roc_auc:.4f} | Best Iter: {model.best_iteration_}"
                )
                return val_pr_auc

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        print("\n4. Training Final Champion Model with Best Hyperparameters...")
        best_params = {
            **study.best_params,
            "objective": "binary",
            "metric": "average_precision",
            "boosting_type": "gbdt",
            "n_estimators": 1200,
            "n_jobs": -1,
            "random_state": 42,
            "verbose": -1,
        }

        champion_model = lgb.LGBMClassifier(**best_params)
        champion_model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )

        # Evaluate on Out-of-Time Val & Test sets
        val_probs = champion_model.predict_proba(X_val)[:, 1]
        test_probs = champion_model.predict_proba(X_test)[:, 1]

        opt_threshold, val_f2 = find_optimal_f2_threshold(y_val, val_probs)
        test_preds = (test_probs >= opt_threshold).astype(int)

        metrics = {
            "val_pr_auc": float(average_precision_score(y_val, val_probs)),
            "val_roc_auc": float(roc_auc_score(y_val, val_probs)),
            "test_pr_auc": float(average_precision_score(y_test, test_probs)),
            "test_roc_auc": float(roc_auc_score(y_test, test_probs)),
            "test_f2_score": float(fbeta_score(y_test, test_preds, beta=2)),
            "test_recall_at_1pct_fpr": recall_at_fixed_fpr(y_test, test_probs, max_fpr=0.01),
            "optimal_f2_threshold": opt_threshold,
        }

        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)

        # Save local production artifacts for ONNX conversion, SHAP, and FastAPI
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(champion_model, artifacts_dir / "champion_lgbm.joblib")
        joblib.dump(encoder, artifacts_dir / "ordinal_encoder.joblib")

        metadata = {
            "feature_cols": feature_cols,
            "categorical_cols": [c for c in CATEGORICAL_COLS if c in feature_cols],
            "optimal_threshold": opt_threshold,
            "metrics": metrics,
            "best_iteration": int(champion_model.best_iteration_),
        }
        with open(artifacts_dir / "model_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Register model in MLflow Model Registry
        mlflow.lightgbm.log_model(
            lgb_model=champion_model,
            artifact_path="model",
            registered_model_name="sentinelpay-fraud-lgbm",
        )

        print("\n" + "=" * 65)
        print("CHAMPION MODEL OUT-OF-TIME TEST RESULTS (SAVE THESE FOR YOUR CV!)")
        print("=" * 65)
        print(f"  Test ROC-AUC               : {metrics['test_roc_auc']:.4f}")
        print(f"  Test PR-AUC                : {metrics['test_pr_auc']:.4f}")
        print(f"  Test F2-Score              : {metrics['test_f2_score']:.4f}")
        print(f"  Test Recall @ 1% FPR       : {metrics['test_recall_at_1pct_fpr']*100:.2f}%")
        print(f"  Optimal Decision Threshold : {metrics['optimal_f2_threshold']:.4f}")
        print("=" * 65)


if __name__ == "__main__":
    run_training_pipeline(
        processed_dir=pathlib.Path("data/processed"),
        artifacts_dir=pathlib.Path("artifacts"),
        n_trials=12,
    )