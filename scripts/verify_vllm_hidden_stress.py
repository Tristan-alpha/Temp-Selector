"""Stress-test vLLM hidden state extraction across 4 rounds of 128 math problems.

Verifies that repeated generate + hidden state extraction doesn't crash vLLM.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/verify_vllm_hidden_stress.py
    CUDA_VISIBLE_DEVICES=0 python scripts/verify_vllm_hidden_stress.py --num-samples 128 --rounds 4
"""
from __future__ import annotations

import argparse
import atexit
import gc
import json
import os
import shutil
import tempfile
import time
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

# Allow direct execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TMPDIR: str | None = None


def cleanup():
    global TMPDIR
    if TMPDIR and os.path.isdir(TMPDIR):
        shutil.rmtree(TMPDIR, ignore_errors=True)
        print(f"[cleanup] removed {TMPDIR}")


atexit.register(cleanup)


def load_math_problems(path: str, n: int) -> List[Dict[str, Any]]:
    """Load first `n` math problems from a JSONL file."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if len(rows) >= n:
                break
    return rows


def format_prompt(tokenizer, problem: str) -> str:
    """Format a math problem with the chat template."""
    messages = [
        {"role": "system", "content": (
            "You are a math reasoning assistant.\n\n"
            "Formatting rules:\n"
            "- Solve the problem step by step.\n"
            "- Each step must be written as a separate paragraph.\n"
            "- Separate every step with exactly two newline characters.\n"
            "- Do not use numbering, bullets, or any markers at the start of a step.\n"
            "- The final paragraph must include the final boxed answer written as \\boxed{}.\n"
        )},
        {"role": "user", "content": problem},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    except Exception:
        return "[SYSTEM]\n" + messages[0]["content"] + "\n\n[USER]\n" + problem + "\n\n[ASSISTANT]\n"


def print_gpu_memory(label: str):
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            alloc = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"  [{label}] GPU {i}: alloc={alloc:.2f} GiB  reserved={reserved:.2f} GiB")


def main():
    global TMPDIR

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/xuezhe/models/Qwen3-8B")
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--data-path", default="data/math-5-sub-200.jsonl")
    parser.add_argument("--gpu", default=None, help="GPU device to use (e.g. 0, 1)")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    print(f"=== vLLM hidden state stress test ===")
    print(f"Model: {args.model}")
    print(f"Samples per round: {args.num_samples}")
    print(f"Rounds: {args.rounds}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Data: {args.data_path}")

    # ---- Load problems ----
    data_full = os.path.join(os.path.dirname(__file__), "..", args.data_path)
    problems = load_math_problems(data_full, args.num_samples)
    print(f"\nLoaded {len(problems)} math problems")

    # ---- Load LLM with extract_hidden_states ----
    from transformers import AutoConfig
    from vllm import LLM, SamplingParams
    from safetensors import safe_open

    hf_cfg = AutoConfig.from_pretrained(args.model)
    num_layers = hf_cfg.num_hidden_layers  # 1-indexed last layer
    hidden_dim = hf_cfg.hidden_size
    print(f"Model layers: {num_layers}, hidden dim: {hidden_dim}")

    TMPDIR = tempfile.mkdtemp(prefix="vllm_stress_", dir="/dev/shm")
    print(f"Hidden state tmpdir: {TMPDIR}")

    print("\nLoading LLM ...")
    t0 = time.perf_counter()
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        max_model_len=args.max_tokens + 2048,
        enable_chunked_prefill=False,
        speculative_config={
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,
            "draft_model_config": {
                "hf_config": {
                    "eagle_aux_hidden_state_layer_ids": [num_layers],
                }
            },
        },
        kv_transfer_config={
            "kv_connector": "ExampleHiddenStatesConnector",
            "kv_role": "kv_producer",
            "kv_connector_extra_config": {
                "shared_storage_path": TMPDIR,
            },
        },
    )
    tokenizer = llm.get_tokenizer()
    print(f"LLM ready in {time.perf_counter() - t0:.1f}s")
    print_gpu_memory("after-init")

    # ---- Format prompts ----
    prompts = [format_prompt(tokenizer, p["problem"]) for p in problems]

    # ---- Run stress test ----
    total_hs_files = 0
    total_hs_bytes = 0
    errors: List[str] = []

    for rnd in range(args.rounds):
        print(f"\n{'='*60}")
        print(f"Round {rnd + 1}/{args.rounds}")
        print(f"{'='*60}")

        sampling_params = [SamplingParams(max_tokens=args.max_tokens, temperature=0.7,
                                          top_p=1.0, top_k=0)] * len(prompts)

        # Generate
        t0 = time.perf_counter()
        torch.cuda.synchronize()
        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
        torch.cuda.synchronize()
        gen_time = time.perf_counter() - t0
        print(f"  Generation: {gen_time:.1f}s  ({gen_time/len(prompts):.2f}s/sample)")

        # Extract hidden states from each output
        round_hs = 0
        round_bytes = 0
        hs_shapes = []
        for i, out in enumerate(outputs):
            hs_path = out.kv_transfer_params.get("hidden_states_path")
            if hs_path is None:
                errors.append(f"Round {rnd+1} sample {i}: no hidden_states_path")
                continue

            try:
                with safe_open(hs_path, "pt") as f:
                    hs = f.get_tensor("hidden_states")  # [seq_len, 1, hidden_dim]
                hs_shapes.append(tuple(hs.shape))
                round_bytes += hs.element_size() * hs.numel()
                round_hs += 1
            except Exception as e:
                errors.append(f"Round {rnd+1} sample {i}: read error: {e}")
            finally:
                try:
                    os.remove(hs_path)
                except OSError:
                    pass

        total_hs_files += round_hs
        total_hs_bytes += round_bytes

        print(f"  Hidden states: {round_hs}/{len(outputs)} extracted")
        print(f"  HS data: {round_bytes / 1024**2:.1f} MiB total")
        if hs_shapes:
            shapes = set(hs_shapes)
            print(f"  HS shapes: {shapes}")

        # Check for leftover files in tmpdir
        leftover = os.listdir(TMPDIR)
        if leftover:
            print(f"  WARNING: {len(leftover)} leftover files in tmpdir: {leftover[:5]}...")
        else:
            print(f"  Leftover files in tmpdir: 0 (clean)")

        print_gpu_memory(f"round-{rnd+1}")

        # Force Python-side GC between rounds
        gc.collect()
        torch.cuda.empty_cache()

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Total rounds: {args.rounds}")
    print(f"Total generations: {args.rounds * args.num_samples}")
    print(f"Total HS files extracted: {total_hs_files}")
    print(f"Total HS data: {total_hs_bytes / 1024**3:.2f} GiB")
    print(f"Errors: {len(errors)}")
    if errors:
        for e in errors[:10]:
            print(f"  - {e}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

    # Final tmpdir state
    final_leftover = os.listdir(TMPDIR)
    print(f"Final leftover files: {len(final_leftover)}")
    if final_leftover:
        print(f"  Files: {final_leftover}")

    print_gpu_memory("final")

    cleanup()
    TMPDIR = None

    if errors:
        print("\n*** TEST FAILED: errors detected ***")
        sys.exit(1)
    else:
        print("\n*** TEST PASSED: no errors ***")


if __name__ == "__main__":
    main()
