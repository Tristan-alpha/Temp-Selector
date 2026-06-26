"""Benchmark: GPU vs CPU cat for realistic eval logprob patterns.

Tests the exact pattern from extract_from_ids (CHUNK=1024) + batch_build_segment_obs_from_lp.
Only tests n_resp=32 (single chunk, the current eval config) since multi-chunk
doesn't apply with segment_size=32.
"""

from __future__ import annotations

import time
import torch


def bench(fn, warmup=20, trials=100):
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(trials):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / trials * 1000


def main():
    have_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0") if have_gpu else torch.device("cpu")
    gpu_name = torch.cuda.get_device_name(0) if have_gpu else "N/A"
    print(f"GPU: {gpu_name}")

    B = 1500
    n_resp = 32
    top_k = 4096
    CHUNK = 1024
    n_chunks = (n_resp + CHUNK - 1) // CHUNK
    data_mb = B * n_resp * (top_k + 1) * 4 / 1e6
    print(f"B={B}  n_resp={n_resp}  top_k={top_k}  n_chunks={n_chunks}  data={data_mb:.0f}MB")
    assert n_chunks == 1, "segment_size < CHUNK, should be 1 chunk"
    print()

    # Build all chain data as single tensors (no chunking needed since n_resp=32 < 1024)
    # Each chain: [32, 4097] float32
    all_chains = torch.randn(B, n_resp, top_k + 1, dtype=torch.float32)

    print("── Pattern 1: single cat(1-chunk) ≡ identity ──")
    # A: "cat" 1 chunk on CPU → stack → .to(device)
    def cat_cpu_stack():
        cpu_tensors = [all_chains[i] for i in range(B)]  # identity "cat"
        return torch.stack(cpu_tensors).to(device)

    # B: "cat" 1 chunk on GPU → stack on GPU
    def cat_gpu_stack():
        gpu_tensors = [all_chains[i].to(device, non_blocking=True) for i in range(B)]
        return torch.stack(gpu_tensors)

    # C: stack on CPU → .to(device) in one shot (what batch_build_segment_obs_from_lp does)
    def stack_cpu_to_gpu():
        return all_chains.to(device)

    ta = bench(cat_cpu_stack, warmup=10, trials=50)
    if have_gpu:
        tb = bench(cat_gpu_stack, warmup=10, trials=50)
        tc = bench(stack_cpu_to_gpu, warmup=10, trials=50)
        print(f"  A (list+stack→GPU, 1500 small ops): {ta:.1f} ms")
        print(f"  B (list .to GPU + stack, 1500 small transfers): {tb:.1f} ms")
        print(f"  C (bulk .to GPU, single transfer):     {tc:.1f} ms")
        print(f"  C is {tb/tc:.0f}x faster than B, {ta/tc:.0f}x faster than A")

    print()
    print("── Pattern 2: what if n_resp > CHUNK (hypothetical multi-chunk) ──")
    # Simulate multi-chunk with smaller B
    B2 = 100
    big_n = 2048
    n_chunks2 = (big_n + CHUNK - 1) // CHUNK  # = 2
    print(f"B={B2}  n_resp={big_n}  n_chunks={n_chunks2}  data={B2*big_n*(top_k+1)*4/1e6:.0f}MB")

    # Build N randomly-sized chunks per chain
    import random
    rng = random.Random(42)
    chunks_per_chain = []
    for i in range(B2):
        chain = []
        remaining = big_n
        for ci in range(n_chunks2):
            sz = min(CHUNK, remaining)
            chain.append(torch.randn(sz, top_k + 1, dtype=torch.float32))
            remaining -= sz
        rng.shuffle(chain)  # shuffle so sizes vary (more realistic)
        chunks_per_chain.append(chain)

    def cat_cpu_stack2():
        result = torch.stack([torch.cat(c, dim=0) for c in chunks_per_chain]).to(device)
        return result

    def cat_gpu_stack2():
        result = torch.stack([
            torch.cat([c.to(device, non_blocking=True) for c in chunks], dim=0)
            for chunks in chunks_per_chain
        ])
        return result

    ta2 = bench(cat_cpu_stack2, warmup=5, trials=20)
    if have_gpu:
        tb2 = bench(cat_gpu_stack2, warmup=5, trials=20)
        print(f"  A (cat CPU, stack→GPU): {ta2:.1f} ms")
        print(f"  B (cat GPU, stack GPU): {tb2:.1f} ms")
        print(f"  diff: {tb2-ta2:+.1f} ms  ({tb2/ta2:.2f}x)")

    del chunks_per_chain, all_chains
    if have_gpu:
        torch.cuda.empty_cache()

    print()
    print("── Conclusion ──")
    print(f"  Current config (n_resp=32, 1 chunk):")
    print(f"    C (single .to(device)) is optimal — what batch_build_segment_obs_from_lp already does.")
    print(f"  Hypothetical multi-chunk (n_resp=2048, 2 chunks):")
    print(f"    GPU cat might help, but segment_size=2048 won't happen in eval.")


if __name__ == "__main__":
    main()
