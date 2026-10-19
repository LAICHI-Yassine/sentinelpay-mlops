import json
import pathlib
import time
import joblib
import numpy as np
import onnxruntime as ort
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType


def export_and_benchmark_onnx(artifacts_dir: pathlib.Path) -> None:
    model_path = artifacts_dir / "champion_lgbm.joblib"
    meta_path = artifacts_dir / "model_metadata.json"
    onnx_path = artifacts_dir / "champion_lgbm.onnx"

    print("1. Loading trained Champion LightGBM model and metadata...")
    model = joblib.load(model_path)
    with open(meta_path, "r") as f:
        metadata = json.load(f)

    n_features = len(metadata["feature_cols"])
    print(f"2. Converting LightGBM ({n_features} features) to ONNX graph...")

    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    onnx_model = onnxmltools.convert_lightgbm(
        model, initial_types=initial_type, target_opset=15
    )
    onnxmltools.utils.save_model(onnx_model, str(onnx_path))

    # Initialize C++ ONNX Runtime session
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess = ort.InferenceSession(
        str(onnx_path), sess_options=sess_options, providers=["CPUExecutionProvider"]
    )
    input_name = sess.get_inputs()[0].name

    # Verify numerical parity between LightGBM and ONNX Runtime
    rng = np.random.default_rng(42)
    dummy_input = rng.standard_normal((1, n_features), dtype=np.float32)

    lgb_prob = float(model.predict_proba(dummy_input)[0, 1])
    onnx_out = sess.run(None, {input_name: dummy_input})
    # onnxmltools outputs [label, [{0: p0, 1: p1}]]
    onnx_prob = float(onnx_out[1][0][1])

    max_diff = abs(lgb_prob - onnx_prob)
    print(f"3. Numerical Parity Check -> LightGBM: {lgb_prob:.6f} | ONNX: {onnx_prob:.6f} (Diff: {max_diff:.2e})")
    assert max_diff < 1e-4, "ONNX output diverged from LightGBM!"

    # Benchmark single-transaction real-time inference latency (1,000 requests)
    n_runs = 1000
    print(f"4. Benchmarking single-transaction real-time latency over {n_runs} calls...")

    lgb_latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = model.predict_proba(dummy_input)
        lgb_latencies.append((time.perf_counter() - t0) * 1000.0)

    onnx_latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = sess.run(None, {input_name: dummy_input})
        onnx_latencies.append((time.perf_counter() - t0) * 1000.0)

    lgb_p99 = float(np.percentile(lgb_latencies, 99))
    onnx_p99 = float(np.percentile(onnx_latencies, 99))
    lgb_mean = float(np.mean(lgb_latencies))
    onnx_mean = float(np.mean(onnx_latencies))
    speedup = lgb_mean / max(onnx_mean, 1e-6)

    metadata["benchmark"] = {
        "lgbm_mean_ms": round(lgb_mean, 4),
        "onnx_mean_ms": round(onnx_mean, 4),
        "lgbm_p99_ms": round(lgb_p99, 4),
        "onnx_p99_ms": round(onnx_p99, 4),
        "speedup_factor": round(speedup, 2),
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 65)
    print("REAL-TIME INFERENCE LATENCY BENCHMARK (BATCH SIZE = 1)")
    print("=" * 65)
    print(f"  Standard Python LightGBM : Mean = {lgb_mean:.3f} ms | p99 = {lgb_p99:.3f} ms")
    print(f"  ONNX Runtime (C++ Graph) : Mean = {onnx_mean:.3f} ms | p99 = {onnx_p99:.3f} ms")
    print(f"  Latency Speedup Factor   : {speedup:.1f}x faster!")
    print("=" * 65)


if __name__ == "__main__":
    export_and_benchmark_onnx(pathlib.Path("artifacts"))