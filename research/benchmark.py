#!/usr/bin/env python3
"""
Benchmark int8 NAX prefill vs baseline for mlx-vlm server.

Measures prefill (TTFT) and decode (TPOT) performance with and without
the --int8-prefill flag that routes prefill matmuls through W8A8 int8
GEMMs on M5 neural accelerators.

Usage:
    python research/benchmark.py [--port PORT] [--iterations N]

The script will:
1. Download the model to research/models/ if not already present
2. Run two benchmark passes: baseline (no int8) and int8-prefill enabled
3. Compare prefill speedup across context sizes (256, 2k, 10k, 20k tokens)
4. Report TTFT, prefill TPS, decode TPS, and speedup ratio
5. Save JSON results to research/results/
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import requests
from openai import OpenAI

# Default model paths
DEFAULT_MODEL = "mlx-community/Qwen3.6-27B-4bit"
DEFAULT_PORT = 8080

# Local model directory
MODELS_DIR = Path(__file__).parent / "models"

# Synthetic prompt template — repeated to reach target token count
# ~1 token per word, so we use a ~150-word block and repeat it
PROMPT_BLOCK = (
    "The following is a paragraph used to generate long context for benchmarking. "
    "Deep learning models have transformed the field of artificial intelligence, "
    "enabling breakthroughs in natural language processing, computer vision, and "
    "reinforcement learning. Transformers, introduced in 2017, revolutionized sequence "
    "modeling by replacing recurrence with self-attention mechanisms. This architecture "
    "allows models to process all tokens in parallel during training, leading to faster "
    "convergence and better scaling with model size. The attention mechanism computes "
    "weighted sums of values where the weights are determined by query-key similarity. "
    "Multi-head attention further improves representation by attending to different "
    "subspaces simultaneously. Positional encodings are added to inject order information "
    "since the self-attention operation itself is permutation-invariant. Modern variants "
    "include rotary positional embeddings, alibi, and learned positional biases. "
    "During inference, autoregressive generation processes one token at a time, "
    "requiring efficient KV cache management to avoid recomputing past activations. "
    "Speculative decoding techniques can accelerate generation by drafting multiple "
    "tokens with a smaller model and verifying them in a single forward pass. "
    "Quantization methods like 4-bit and 8-bit reduce memory footprint significantly, "
    "making large models feasible on consumer hardware like Apple Silicon. "
    "The trade-off between precision and speed is an active area of research. "
)


def _make_prompt(target_tokens: int, question: str) -> list[dict]:
    """Build a prompt with approximately target_tokens of context + a short question."""
    block_tokens = len(PROMPT_BLOCK.split())  # ~150 words ≈ 150 tokens
    repeats = max(1, target_tokens // block_tokens)
    context = PROMPT_BLOCK * repeats
    return [
        {
            "role": "user",
            "content": f"{context}\n\nQuestion: {question}",
        }
    ]


def _count_tokens(messages: list[dict]) -> int:
    """Approximate token count from messages (1 token ≈ 1 word)."""
    return sum(len(str(m["content"]).split()) for m in messages)


def _average_results(results: list[dict]) -> list[dict]:
    """Average results across iterations, grouping by prompt name."""
    grouped: dict[str, list[dict]] = {}
    for r in results:
        grouped.setdefault(r["prompt"], []).append(r)

    averaged = []
    for name, runs in grouped.items():
        avg = {
            "prompt": name,
            "context_tokens": runs[0]["context_tokens"],
            "iterations": len(runs),
            "ttft_s": round(sum(r["ttft_s"] for r in runs) / len(runs), 3),
            "ttft_ms": round(sum(r["ttft_ms"] for r in runs) / len(runs), 2),
            "tpot_ms": round(sum(r["tpot_ms"] for r in runs) / len(runs), 2),
            "prefill_tps": round(sum(r["prefill_tps"] for r in runs) / len(runs), 2),
            "decode_tps": round(sum(r["decode_tps"] for r in runs) / len(runs), 2),
            "generated_tokens": sum(r["generated_tokens"] for r in runs),
        }
        averaged.append(avg)
    return averaged


# Benchmark prompts: (name, target_context_tokens, max_output_tokens)
BENCHMARK_PROMPTS = [
    ("short  (256 ctx)",   256,  50),
    ("medium (2k ctx)",   2000, 100),
    ("long   (10k ctx)", 10000, 100),
    ("xlong  (20k ctx)", 20000, 100),
]


def download_model(model_id: str, local_dir: Path) -> Path:
    """Download a model from Hugging Face to the local directory."""
    from huggingface_hub import snapshot_download

    target_dir = local_dir / model_id.split("/")[-1]
    if target_dir.exists():
        print(f"  Model already exists at {target_dir}")
        return target_dir

    print(f"  Downloading {model_id} to {target_dir}...")
    snapshot_download(
        repo_id=model_id,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
    )
    print(f"  Download complete.")
    return target_dir


def get_python_executable() -> str:
    """Find the correct Python executable, preferring the project's .venv."""
    venv_python = str(Path(__file__).parent.parent / ".venv" / "bin" / "python")
    if Path(venv_python).exists():
        return venv_python
    return sys.executable


def start_server(
    model_path: str,
    port: int,
    int8_prefill: bool = False,
) -> subprocess.Popen:
    """Start the mlx-vlm server and wait for it to be ready."""
    python_exe = get_python_executable()
    cmd = [
        python_exe,
        "-m",
        "mlx_vlm.server",
        "--model",
        model_path,
        "--port",
        str(port),
    ]

    if int8_prefill:
        cmd.append("--int8-prefill")

    print(f"\nStarting server: {' '.join(cmd)}")
    print("(This may take a while to load the model...)\n")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait for server to be ready
    max_wait = 600  # 10 minutes max
    start = time.time()

    while time.time() - start < max_wait:
        try:
            resp = requests.get(f"http://localhost:{port}/health", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                print(f"Server is ready: {data}")
                return proc
        except (requests.ConnectionError, requests.Timeout):
            time.sleep(2)

    print("ERROR: Server failed to start within timeout")
    proc.terminate()
    sys.exit(1)


def stop_server(proc: subprocess.Popen):
    """Stop the server gracefully."""
    print("\nStopping server...")
    try:
        proc.terminate()
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        print("  Server did not stop gracefully, forcing kill...")
        proc.kill()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            print("  WARNING: Server process could not be killed.")
    print("Server stopped.")


def run_benchmark(
    client: OpenAI, name: str, messages: list[dict], max_tokens: int, run_index: int, model_id: str
) -> dict:
    """Run a single benchmark iteration and collect timing metrics."""
    start_time = time.perf_counter()
    first_token_time = None
    token_times = []
    total_tokens = 0
    last_checkpoint = time.perf_counter()

    stream = client.chat.completions.create(
        model=model_id,
        messages=messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=0.0,
    )

    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta:
            delta = chunk.choices[0].delta
            if delta.content:
                now = time.perf_counter()
                if first_token_time is None:
                    first_token_time = now
                token_times.append(now - last_checkpoint)
                last_checkpoint = now
                total_tokens += 1

    end_time = time.perf_counter()

    total_time = end_time - start_time
    ttft = (first_token_time - start_time) * 1000 if first_token_time else 0  # ms

    # Calculate TPOT (average inter-token latency)
    tpot = 0
    decode_tps = 0
    if len(token_times) > 1:
        decode_intervals = token_times[1:]
        avg_interval = sum(decode_intervals) / len(decode_intervals)
        tpot = avg_interval * 1000  # ms
        decode_tps = 1000 / tpot if tpot > 0 else 0

    # Prompt tokens (approximate)
    prompt_tokens = _count_tokens(messages)
    prefill_tps = (prompt_tokens / (ttft / 1000)) if ttft > 0 else 0

    return {
        "run": run_index,
        "prompt": name,
        "context_tokens": prompt_tokens,
        "generated_tokens": total_tokens,
        "total_time_s": round(total_time, 3),
        "ttft_ms": round(ttft, 2),
        "ttft_s": round(ttft / 1000, 3),
        "tpot_ms": round(tpot, 2),
        "prefill_tps": round(prefill_tps, 2),
        "decode_tps": round(decode_tps, 2),
    }


def print_table(label: str, results: list[dict], model_id: str):
    """Print a results table for one mode (baseline or int8)."""
    print(f"\n{'=' * 90}")
    print(f"  {label}")
    print(f"{'=' * 90}")
    print(f"Model:  {model_id}")
    print("-" * 90)
    print(
        f"{'Context':<14} {'Ctx Tok':>8} {'TTFT(s)':>8} {'TTFT(ms)':>10} "
        f"{'Prefill TPS':>12} {'Decode TPS':>12}"
    )
    print("-" * 90)

    for r in results:
        print(
            f"{r['prompt']:<14} {r['context_tokens']:>8} "
            f"{r['ttft_s']:>8.3f} {r['ttft_ms']:>10.2f} "
            f"{r['prefill_tps']:>12.2f} {r['decode_tps']:>12.2f}"
        )

    print(f"{'=' * 90}\n")


def print_comparison(baseline: list[dict], int8: list[dict], model_id: str):
    """Print side-by-side comparison of baseline vs int8-prefill (averaged)."""
    print(f"\n{'=' * 100}")
    print("  COMPARISON: int8 NAX Prefill Speedup")
    print(f"{'=' * 100}")
    print(f"Model:  {model_id}")
    print(f"Date:   {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 100)
    print(
        f"{'Context':<14} {'Baseline TPS':>14} {'int8 TPS':>14} "
        f"{'Speedup':>10} {'Baseline TTFT':>14} {'int8 TTFT':>14}"
    )
    print("-" * 100)

    for b, i in zip(baseline, int8):
        if b["prefill_tps"] > 0:
            speedup = i["prefill_tps"] / b["prefill_tps"]
        else:
            speedup = 0.0
        print(
            f"{b['prompt']:<14} {b['prefill_tps']:>14.2f} {i['prefill_tps']:>14.2f} "
            f"{speedup:>9.2f}x {b['ttft_s']:>14.3f}s {i['ttft_s']:>14.3f}s"
        )

    print(f"{'=' * 100}\n")

    # Save results
    results_path = Path(__file__).parent / "results"
    results_path.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_int8_{timestamp}.json"
    output_file = results_path / filename

    output = {
        "model": model_id,
        "baseline": baseline,
        "int8_prefill": int8,
        "timestamp": timestamp,
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark int8 NAX prefill vs baseline"
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="Server port (default: 8080)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model ID or path (default: mlx-community/Qwen3.6-27B-4bit)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of benchmark iterations per prompt (default: 3)",
    )
    args = parser.parse_args()

    model_id = args.model or DEFAULT_MODEL

    # Step 1: Download model
    print("=" * 60)
    print("Step 1: Preparing model")
    print("=" * 60)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = download_model(model_id, MODELS_DIR)
    local_model_path = str(model_path)

    proc = None
    try:
        # Step 2: Run benchmarks for both modes
        print("\n" + "=" * 60)
        print("Step 2: Running benchmarks")
        print("=" * 60)

        all_baseline = []
        all_int8 = []

        for label, int8_flag in [("Baseline (no int8)", False), ("int8 NAX Prefill", True)]:
            print(f"\n{'=' * 60}")
            print(f"Mode: {label}")
            print(f"{'=' * 60}")

            # Start (or restart) server with the correct flag
            if proc is not None:
                stop_server(proc)
            proc = start_server(local_model_path, args.port, int8_prefill=int8_flag)

            client = OpenAI(
                base_url=f"http://localhost:{args.port}/v1", api_key="not-needed"
            )

            # Warmup
            print("\n  Warmup (not counted)...")
            warmup_messages = _make_prompt(256, "What is AI?")
            run_benchmark(client, "warmup", warmup_messages, 50, 0, local_model_path)
            print("  Warmup complete.\n")

            # Benchmarks
            mode_results = []
            for name, ctx_tokens, max_out in BENCHMARK_PROMPTS:
                messages = _make_prompt(
                    ctx_tokens,
                    "Summarize the key ideas in the text above in 3 sentences.",
                )
                actual_ctx = _count_tokens(messages)
                print(f"  {name} ({actual_ctx} context tokens)")
                for i in range(1, args.iterations + 1):
                    print(f"    Run {i}/{args.iterations}...")
                    result = run_benchmark(
                        client, name, messages, max_out, i, local_model_path
                    )
                    mode_results.append(result)
                    print(
                        f"      TTFT: {result['ttft_s']:.3f}s | "
                        f"Prefill: {result['prefill_tps']:.2f} tok/s | "
                        f"Decode: {result['decode_tps']:.2f} tok/s"
                    )

            if not int8_flag:
                all_baseline = mode_results
            else:
                all_int8 = mode_results

        # Average across iterations
        avg_baseline = _average_results(all_baseline)
        avg_int8 = _average_results(all_int8)

        # Print results
        print_table("Baseline (no int8)", avg_baseline, model_id)
        print_table("int8 NAX Prefill", avg_int8, model_id)
        print_comparison(avg_baseline, avg_int8, model_id)

    finally:
        if proc is not None:
            stop_server(proc)


if __name__ == "__main__":
    main()