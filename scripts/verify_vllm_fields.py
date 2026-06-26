"""Verify vLLM logprob & hidden state interfaces.  tfinder env, GPU 7."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"


def main():
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    model_path = "/home/xuezhe/models/Qwen3-8B"

    print("Loading LLM ...")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        max_model_len=10240,
        enforce_eager=True,
        max_logprobs=4096,
    )
    tok = llm.get_tokenizer()

    prompt = "What is 2+2? Answer:"
    response = "The answer is 4."
    prompt_ids = tok.encode(prompt, add_special_tokens=False)
    resp_ids = tok.encode(response, add_special_tokens=False)
    full_ids = prompt_ids + resp_ids
    P = len(prompt_ids)
    R = len(resp_ids)
    print(f"P={P} R={R} full_len={P+R}")

    # === Test 1: prompt_logprobs=256 ===
    print("\n=== max_tokens=1, prompt_logprobs=256 ===")
    out = llm.generate(
        [full_ids],
        SamplingParams(max_tokens=1, prompt_logprobs=256),
    )[0]
    _dump_prompt_logprobs(out, P, R)

    # === Test 2: prompt_logprobs=4096 ===
    print("\n=== max_tokens=1, prompt_logprobs=4096 ===")
    out = llm.generate(
        [full_ids],
        SamplingParams(max_tokens=1, prompt_logprobs=4096),
    )[0]
    _dump_prompt_logprobs(out, P, R)

    # === Test 3: LLM max_logprobs check ===
    print("\n=== LLM constructor params ===")
    import inspect
    sig = inspect.signature(LLM.__init__)
    for name, param in sig.parameters.items():
        if 'logprob' in name.lower() or name in ('self', 'kwargs'):
            continue
    # Print all params with logprob mention
    for name, param in sig.parameters.items():
        if 'logprob' in name.lower():
            print(f"  {name}: default={param.default}")

    # === Test 3: LLM constructor logprob-related params ===
    print("\n=== LLM.__init__ params containing 'logprob' ===")
    import inspect
    sig = inspect.signature(LLM.__init__)
    for name, param in sig.parameters.items():
        if 'logprob' in name.lower():
            print(f"  {name}: default={param.default}")


def _dump_prompt_logprobs(out, P, R):
    plp = out.prompt_logprobs
    if plp is None:
        print(f"  prompt_logprobs is None")
        return
    print(f"  prompt_logprobs len={len(plp)}")
    non_none = sum(1 for x in plp if x is not None)
    print(f"  non-None entries: {non_none}")

    # Slice response portion [P:]
    resp_slice = plp[P:]
    nn_slice = [x for x in resp_slice if x is not None]
    print(f"  [{P}:] len={len(resp_slice)} non_none={len(nn_slice)}  (need >= {R})")

    # Show a sample entry
    sample = next((x for x in resp_slice if x is not None), None)
    if sample and isinstance(sample, dict):
        items = list(sample.items())[:3]
        print(f"  sample entry: {items}")


if __name__ == "__main__":
    main()
