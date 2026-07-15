#!/usr/bin/env python3
"""Compare ONNX Runtime CPU use with worker spinning enabled or disabled."""

from __future__ import annotations

import argparse
import resource
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--spinning", choices=("on", "off"), required=True)
    parser.add_argument("--rate", type=float, required=True)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--threads", type=int, default=2)
    return parser.parse_args()


def cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def main() -> None:
    args = parse_args()
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    allow_spinning = "1" if args.spinning == "on" else "0"
    options.add_session_config_entry(
        "session.intra_op.allow_spinning", allow_spinning
    )
    options.add_session_config_entry(
        "session.inter_op.allow_spinning", allow_spinning
    )

    session = ort.InferenceSession(
        str(args.model),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    model_input = session.get_inputs()[0]
    shape = [value if isinstance(value, int) else 1 for value in model_input.shape]
    input_data = np.zeros(shape, dtype=np.float32)

    for _ in range(5):
        session.run(None, {model_input.name: input_data})

    interval = 1.0 / args.rate
    latencies: list[float] = []
    start_wall = time.perf_counter()
    start_cpu = cpu_seconds()
    deadline = start_wall
    while time.perf_counter() - start_wall < args.duration:
        inference_start = time.perf_counter()
        session.run(None, {model_input.name: input_data})
        latencies.append(time.perf_counter() - inference_start)
        deadline += interval
        remaining = deadline - time.perf_counter()
        if remaining > 0.0:
            time.sleep(remaining)

    wall_seconds = time.perf_counter() - start_wall
    process_cpu_seconds = cpu_seconds() - start_cpu
    sorted_latencies = sorted(latencies)
    p95_index = min(len(sorted_latencies) - 1, int(len(sorted_latencies) * 0.95))
    print(
        f"spinning={args.spinning} rate_hz={args.rate:.1f} "
        f"threads={args.threads} samples={len(latencies)} "
        f"cpu_percent={100.0 * process_cpu_seconds / wall_seconds:.1f} "
        f"cpu_seconds={process_cpu_seconds:.3f} wall_seconds={wall_seconds:.3f} "
        f"latency_mean_ms={1000.0 * statistics.mean(latencies):.1f} "
        f"latency_p95_ms={1000.0 * sorted_latencies[p95_index]:.1f}"
    )


if __name__ == "__main__":
    main()
