"""Compare 3 methods for getting response-token logprobs from vLLM:

A: native prompt_logprobs (baseline)
B: hidden_states → safetensors lm_head → manual logprobs
C: hidden_states → apply_model → model.compute_logits → compute_topk_logprobs

GPU 7, tfinder env.
"""
import atexit, os, shutil, tempfile, time
TMPDIR = None


def cleanup():
    global TMPDIR
    if TMPDIR and os.path.isdir(TMPDIR):
        shutil.rmtree(TMPDIR, ignore_errors=True)


atexit.register(cleanup)


def main():
    global TMPDIR
    from vllm import LLM, SamplingParams
    import torch, json, glob
    from safetensors import safe_open

    model_path = "/home/xuezhe/models/Qwen3-8B"
    TMPDIR = tempfile.mkdtemp(prefix="vllm_hs_cmp_")

    # ---- Load lm_head + norm from safetensors (for path B) ----
    lm_weight = norm_weight = None
    for sf in sorted(glob.glob(f"{model_path}/*.safetensors")):
        with safe_open(sf, framework="pt", device="cpu") as f:
            if "lm_head.weight" in f.keys():
                lm_weight = f.get_tensor("lm_head.weight")
            if "model.norm.weight" in f.keys():
                norm_weight = f.get_tensor("model.norm.weight")
    assert lm_weight is not None and norm_weight is not None

    # ---- vLLM ----
    cfg = json.load(open(f"{model_path}/config.json"))
    eps = cfg.get("rms_norm_eps", 1e-6)
    print(f"Loading LLM  (rms_norm_eps={eps}) ...")
    llm = LLM(
        model=model_path, tensor_parallel_size=1, gpu_memory_utilization=0.90,
        max_model_len=10240, enforce_eager=True,
        speculative_config={
            "method": "extract_hidden_states", "num_speculative_tokens": 1,
            "draft_model_config": {"hf_config": {"eagle_aux_hidden_state_layer_ids": [36]}},
        },
        kv_transfer_config={
            "kv_connector": "ExampleHiddenStatesConnector", "kv_role": "kv_producer",
            "kv_connector_extra_config": {"shared_storage_path": TMPDIR},
        },
        enable_chunked_prefill=False,
    )
    tok = llm.get_tokenizer()

    prompt = "What is 2+2? Answer:"
    response = "The answer is 4."
    p_ids = tok.encode(prompt, add_special_tokens=False)
    r_ids = tok.encode(response, add_special_tokens=False)
    full_ids = p_ids + r_ids
    P, R = len(p_ids), len(r_ids)
    print(f"P={P} R={R}")

    # ================================================================
    # A: native prompt_logprobs
    # ================================================================
    print("\n=== A: native prompt_logprobs ===")
    t0 = time.perf_counter()
    out = llm.generate(
        [full_ids],
        SamplingParams(max_tokens=1, prompt_logprobs=16, top_p=1.0, top_k=0, temperature=1.0),
    )[0]
    t_native = time.perf_counter() - t0
    native_lp = [[v.logprob for v in row.values()] for row in out.prompt_logprobs[P:] if row]
    print(f"  time={t_native:.4f}s  entries={len(native_lp)}")

    # ================================================================
    # B: hidden_states → safetensors lm_head → manual logprobs
    # ================================================================
    print("\n=== B: manual (safetensors lm_head) ===")
    t0 = time.perf_counter()
    out_hs = llm.generate([full_ids], SamplingParams(max_tokens=1, top_p=1.0, top_k=0))[0]
    with safe_open(out_hs.kv_transfer_params["hidden_states_path"], "pt") as f:
        hs = f.get_tensor("hidden_states")  # [seq_len, 1, 4096]
    os.remove(out_hs.kv_transfer_params["hidden_states_path"])
    t_hs = time.perf_counter() - t0

    t0 = time.perf_counter()
    hs_gpu = hs[:, -1, :].cuda()
    resp_hs = hs_gpu[P - 1 : P - 1 + R]
    lm_gpu = lm_weight.cuda().to(hs_gpu.dtype)
    nw_gpu = norm_weight.cuda().to(hs_gpu.dtype)

    resp_normed = torch.nn.functional.rms_norm(
        resp_hs.float(), [resp_hs.shape[-1]], weight=nw_gpu.float(), eps=eps,
    ).to(hs_gpu.dtype)
    logits = torch.nn.functional.linear(resp_normed, lm_gpu)
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    b_topk_vals, _ = torch.topk(logprobs, 16, dim=-1)
    resp_t = torch.tensor(r_ids, device="cuda")
    b_sampled = logprobs[torch.arange(R), resp_t]
    t_compute = time.perf_counter() - t0
    print(f"  hidden_extract={t_hs:.4f}s  compute={t_compute:.4f}s")

    # ================================================================
    # C: apply_model → model.compute_logits → compute_topk_logprobs
    # ================================================================
    print("\n=== C: apply_model compute_logits ===")
    from vllm.v1.worker.gpu.sample.logprob import compute_topk_logprobs
    resp_hs_cpu = hs[:, -1, :][P - 1 : P - 1 + R].cpu()
    resp_t_cpu = torch.tensor(r_ids)

    def _make_fn(h_cpu, t_cpu, k):
        def _fn(model):
            h = h_cpu.to(next(model.parameters()).device, non_blocking=True)
            ids = t_cpu.to(h.device, non_blocking=True)
            normed = model.model.norm(h)
            logits = model.compute_logits(normed)
            result = compute_topk_logprobs(logits, k, ids)
            return torch.stack([result.logprobs.cpu(), result.logprob_token_ids.cpu().float()])
        return _fn

    t0 = time.perf_counter()
    raw = llm.apply_model(_make_fn(resp_hs_cpu, resp_t_cpu, 16))[0]
    t_am = time.perf_counter() - t0
    c_lp = raw[0]  # [R, 17] logprobs
    print(f"  apply_model={t_am:.4f}s  shape={list(c_lp.shape)}")

    # ================================================================
    # C2: without norm (test if Qwen3 needs it)
    # ================================================================
    print("\n=== C2: apply_model WITHOUT norm ===")
    def _make_fn_no_norm(h_cpu, t_cpu, k):
        def _fn(model):
            h = h_cpu.to(next(model.parameters()).device, non_blocking=True)
            ids = t_cpu.to(h.device, non_blocking=True)
            logits = model.compute_logits(h)  # no norm
            result = compute_topk_logprobs(logits, k, ids)
            return torch.stack([result.logprobs.cpu(), result.logprob_token_ids.cpu().float()])
        return _fn

    raw2 = llm.apply_model(_make_fn_no_norm(resp_hs_cpu, resp_t_cpu, 16))[0]
    c2_lp = raw2[0]
    c2_sampled = c2_lp[:, 0]

    # ================================================================
    # Compare
    # ================================================================
    print("\n=== Correctness (vs native) ===")
    native_sampled = [r[0] for r in native_lp]
    c_sampled = c_lp[:, 0]

    for label, vals in (("C with norm", c_sampled), ("C2 WITHOUT norm", c2_sampled)):
        d = max(abs(native_sampled[i] - vals[i].item()) for i in range(R))
        ok = "✓" if d < 1e-6 else f"✗ ({d:.2e})"
        print(f"  {label:18s} sampled max_diff={ok}")
    b_sampled_cpu = b_sampled.cpu()

    for label, vals in (("B manual", b_sampled_cpu), ("C compute_logits", c_sampled)):
        d = max(abs(native_sampled[i] - vals[i].item()) for i in range(R))
        ok = "✓" if d < 1e-6 else f"✗ ({d:.2e})"
        print(f"  {label:18s} sampled max_diff={ok}")

    n_topk = [sorted(r, reverse=True) for r in native_lp]
    b_topk = [sorted(b_topk_vals[i].cpu().tolist(), reverse=True) for i in range(R)]
    # C returns 17 cols (1 sampled + 16 top-k), native has 16 (1+15). Align.
    c_topk = [sorted(c_lp[i, 1:].tolist()[:len(n_topk[i])], reverse=True) for i in range(R)]

    for label, mt in (("B manual", b_topk), ("C compute_logits", c_topk)):
        d = max(abs(a - b) for i in range(R) for a, b in zip(n_topk[i], mt[i]))
        ok = "✓" if d < 1e-6 else f"✗ ({d:.2e})"
        print(f"  {label:18s} topk  max_diff={ok}")

    print(f"\n  native (16vals) top3: {[f'{x:.6f}' for x in n_topk[0][:3]]}")
    print(f"  B               top3: {[f'{x:.6f}' for x in b_topk[0][:3]]}")
    print(f"  C (16 top-k)    top3: {[f'{x:.6f}' for x in c_topk[0][:3]]}")

    # ================================================================
    # Speed
    # ================================================================
    print("\n=== Speed (30 rounds each) ===")
    for _ in range(5):
        _ = llm.generate([full_ids], SamplingParams(max_tokens=1, top_p=1.0, top_k=0))

    for label, sp in (
        ("A native_logprobs", SamplingParams(max_tokens=1, prompt_logprobs=16, top_p=1.0, top_k=0)),
        ("B hidden_states ", SamplingParams(max_tokens=1, top_p=1.0, top_k=0)),
    ):
        times = []
        for _ in range(30):
            t0 = time.perf_counter()
            out = llm.generate([full_ids], sp)[0]
            if "hidden" in label:
                with safe_open(out.kv_transfer_params["hidden_states_path"], "pt") as f:
                    f.get_tensor("hidden_states")
                os.remove(out.kv_transfer_params["hidden_states_path"])
            else:
                _ = out.prompt_logprobs
            times.append(time.perf_counter() - t0)
        print(f"  {label}: avg={sum(times)/len(times):.4f}s  min={min(times):.4f}s")

    cleanup(); TMPDIR = None
    print("\nDone.")


if __name__ == "__main__":
    main()
