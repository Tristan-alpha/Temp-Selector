"""Verify vLLM hidden state extraction with speculative decode + immediate file cleanup.

Usage: CUDA_VISIBLE_DEVICES=7 conda run -n tfinder python scripts/verify_vllm_hidden.py
"""
import atexit
import os
import shutil
import tempfile
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

TMPDIR = None


def cleanup():
    global TMPDIR
    if TMPDIR and os.path.isdir(TMPDIR):
        shutil.rmtree(TMPDIR, ignore_errors=True)
        print(f"[cleanup] removed {TMPDIR}")


atexit.register(cleanup)


def main():
    global TMPDIR

    from vllm import LLM, SamplingParams
    from safetensors import safe_open
    import torch

    model_path = "/home/xuezhe/models/Qwen3-8B"
    TMPDIR = tempfile.mkdtemp(prefix="vllm_hs_")
    print(f"tmpdir: {TMPDIR}")

    print("Loading LLM with speculative extract_hidden_states ...")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        max_model_len=10240,
        enforce_eager=True,
        speculative_config={
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,
            "draft_model_config": {
                "hf_config": {
                    "eagle_aux_hidden_state_layer_ids": [28],
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
    tok = llm.get_tokenizer()
    print("LLM ready.\n")

    prompt = "What is 2+2? Answer:"
    response = "The answer is 4."
    prompt_ids = tok.encode(prompt, add_special_tokens=False)
    resp_ids = tok.encode(response, add_special_tokens=False)
    full_ids = prompt_ids + resp_ids
    P = len(prompt_ids)
    R = len(resp_ids)
    print(f"P={P} R={R} full={P+R}")

    # ---- Round 1: extract + delete + verify tensor alive ----
    print("\n=== Round 1 ===")
    out = llm.generate([full_ids], SamplingParams(max_tokens=1))[0]
    hs_path = out.kv_transfer_params["hidden_states_path"]
    print(f"  file: {os.path.basename(hs_path)}")
    print(f"  tmpdir: {os.listdir(TMPDIR)}")

    with safe_open(hs_path, "pt") as f:
        token_ids = f.get_tensor("token_ids")
        hs = f.get_tensor("hidden_states")
    print(f"  token_ids shape: {token_ids.shape}")  # [seq_len]
    print(f"  hidden_states shape: {hs.shape}")     # [seq_len, num_layers, hidden_dim]
    # hs shape is [seq_len, num_layers, hidden_dim]
    # Squeeze layers dim (we only have 1 layer), slice response
    hs_1layer = hs[:, -1, :]          # [seq_len, hidden_dim] — last (only) layer
    resp_hs = hs_1layer[P - 1 :]       # response portion
    print(f"  resp_hs shape: {resp_hs.shape}  (expect >= {R})")

    os.remove(hs_path)
    print(f"  deleted file, tensor mean: {resp_hs.float().mean().item():.4f}")

    # ---- Round 2: second generate after cleanup ----
    print("\n=== Round 2 (post-delete) ===")
    out2 = llm.generate([full_ids], SamplingParams(max_tokens=1))[0]
    hs_path2 = out2.kv_transfer_params["hidden_states_path"]
    print(f"  file: {os.path.basename(hs_path2)}")
    with safe_open(hs_path2, "pt") as f:
        hs2 = f.get_tensor("hidden_states")
    rh2 = hs2[:, -1, :][P - 1 :]
    print(f"  resp_hs shape: {rh2.shape}  mean: {rh2.float().mean().item():.4f}")
    os.remove(hs_path2)

    # ---- Round 3: batch of 2, delete each after read ----
    print("\n=== Round 3 (batch x2) ===")
    prompts = ["What is 2+2? Answer:", "The capital of France is"]
    responses = ["The answer is 4.", "Paris."]
    batch_ids = []
    batch_P = []
    for p, r in zip(prompts, responses):
        pid = tok.encode(p, add_special_tokens=False)
        rid = tok.encode(r, add_special_tokens=False)
        batch_ids.append(pid + rid)
        batch_P.append(len(pid))

    outs = llm.generate(batch_ids, [SamplingParams(max_tokens=1)] * 2)
    for i, o in enumerate(outs):
        p = o.kv_transfer_params["hidden_states_path"]
        with safe_open(p, "pt") as f:
            h = f.get_tensor("hidden_states")
        rh = h[:, -1, :][batch_P[i] - 1 :]
        print(f"  [{i}] path={os.path.basename(p)} shape={h.shape} resp={rh.shape} mean={rh.float().mean().item():.4f}")
        os.remove(p)  # ONLY delete this file, not others

    # ---- Round 4: generate 3 rounds at speed ----
    print("\n=== Round 4 (3x speed) ===")
    for rnd in range(3):
        t0 = time.perf_counter()
        out = llm.generate([full_ids], SamplingParams(max_tokens=1))[0]
        t1 = time.perf_counter()
        p = out.kv_transfer_params["hidden_states_path"]
        with safe_open(p, "pt") as f:
            h = f.get_tensor("hidden_states")
        os.remove(p)
        print(f"  [{rnd}] {t1-t0:.3f}s  hs_shape={h.shape}")

    cleanup()
    TMPDIR = None
    print("\nDone.")


if __name__ == "__main__":
    main()
