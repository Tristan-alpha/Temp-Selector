"""Online PPO training — the policy controls vLLM generation temperature
segment-by-segment, receiving real math-verify rewards.

Usage (single-process; vLLM tensor parallelism handles multi-GPU):
    CUDA_VISIBLE_DEVICES=0,1 python -m ppo.training --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import random
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from features.segmenter import segment_pooling
from features.vectorizer import token_to_vec, compute_entropy
from utils.jsonl import sample_prefix
from mil.model import MILModel
from ppo.model import PolicyValueNet, sample_action, compute_gae, load_mil_encoder_for_warmstart
from utils.exp_logger import log_exception, setup_experiment_logger
from inference.vllm_hidden_extractor import VLLMHiddenStateExtractor


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_prompts(path: str, logger=None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                if logger:
                    logger.warning("skipping malformed JSON at line %d", line_no)
    return rows


def _extract_segment_obs(token_ids: List[int], logprobs_list: List[Any],
                         obs_dim: int, top_k: int,
                         hidden_states: Optional[List[List[float]]] = None,
                         feature_mode: str = "basic") -> Optional[List[float]]:
    """Mean-pool per-token features into one segment observation vector.

    When hidden_states is provided (feature_mode=hidden_states or all),
    mean-pools the hidden state vectors and concatenates with standard
    logprob/entropy/topk_logits features (or returns hidden-only for
    hidden_states mode).
    """
    if not token_ids or not logprobs_list:
        return None

    # Standard logprob features
    token_obs: List[List[float]] = []
    for tid, lp_item in zip(token_ids, logprobs_list):
        if lp_item is None:
            continue
        if isinstance(lp_item, dict):
            selected_obj = lp_item.get(tid)
            if selected_obj is None:
                selected_lp = float(max(lp_item.values(), key=lambda v: v.logprob).logprob) if lp_item else -20.0
            else:
                selected_lp = float(selected_obj.logprob)
            logprob_vals = sorted([float(v.logprob) for v in lp_item.values()], reverse=True)[:top_k]
        else:
            selected_lp = float(getattr(lp_item, 'logprob', -20.0))
            logprob_vals = [selected_lp] * top_k
        entropy_val = compute_entropy(logprob_vals)
        obs = token_to_vec({"logprob": selected_lp, "entropy": entropy_val, "topk_logits": logprob_vals}, obs_dim)
        token_obs.append(obs)

    std_vec: List[float] = []
    if feature_mode != "hidden_states":
        if not token_obs:
            return None
        avg = [0.0] * obs_dim
        for row in token_obs:
            for i, v in enumerate(row):
                avg[i] += v
        denom = float(len(token_obs))
        std_vec = [v / denom for v in avg]

    # Hidden state features
    if hidden_states is not None and feature_mode in {"hidden_states", "all"}:
        n_hs = len(hidden_states)
        if n_hs == 0:
            return std_vec if std_vec else None
        hs_dim = len(hidden_states[0])
        hs_pooled = [0.0] * hs_dim
        for hs in hidden_states:
            for j, v in enumerate(hs):
                hs_pooled[j] += v
        hs_pooled = [v / n_hs for v in hs_pooled]
        if feature_mode == "hidden_states":
            return hs_pooled
        # all: concatenate
        return std_vec + hs_pooled

    return std_vec if std_vec else None


def load_train_prompts(dataset_path: str) -> list:
    """Extract unique (question, answer) pairs from labeled train dataset JSONL."""
    seen: set = set()
    prompts: list = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            sid = str(row.get("sample_id", ""))
            prefix = sample_prefix(sid)
            if prefix in seen:
                continue
            seen.add(prefix)
            prompts.append({
                "question": row.get("prompt", ""),
                "answer": row.get("metadata", {}).get("gold_answer", ""),
            })
    return prompts


def train_ppo(
    config_path: str,
    train_path: str,
    mil_ckpt: str | None = None,
    tensor_parallel_size: int | str = "auto",
    run_name: str | None = None,
    log_dir: str = "logs",
) -> None:
    cfg = _load_config(config_path)
    logger, _log_path, final_run_name = setup_experiment_logger(
        component="train_ppo", run_name=run_name, log_dir=log_dir, config=cfg,
    )

    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)

    obs_dim = int(cfg["data"]["instance_dim"])
    hidden_dim = int(cfg["mil"]["model"]["hidden_dim"])
    temp_bins = [float(x) for x in cfg["data"]["temp_bins"]]
    n_actions = len(temp_bins)
    segment_size = int(cfg["data"]["segment_size"])
    max_new_tokens = int(cfg["inference"]["max_new_tokens"])
    top_k_logits = int(cfg["inference"]["top_k_logits"])
    num_votes = int(cfg["inference"].get("num_votes", 1))
    system_prompt = cfg["inference"].get("system_prompt", "")
    use_math_chat = bool(cfg["inference"].get("use_math_chat_prompt", True))
    feature_mode = cfg["inference"].get("feature_mode", "basic")

    # Hidden state extractor (two-pass prefill trick)
    hs_extractor = None
    if feature_mode in {"hidden_states", "all"}:
        layer_ids = cfg["inference"].get("eagle_aux_hidden_state_layer_ids", [28])
        hs_extractor = VLLMHiddenStateExtractor(
            model_name_or_path=cfg["inference"]["model_name_or_path"],
            layer_ids=[int(x) for x in layer_ids],
            tensor_parallel_size=1,
            gpu_memory_utilization=0.30,
        )
        logger.info("hidden_state_extractor ready feature_mode=%s", feature_mode)

    max_iterations = int(cfg["ppo"]["training"]["max_iterations"])
    early_stop_patience = int(cfg["ppo"]["training"]["early_stop_patience"])
    rollout_size = int(cfg["ppo"]["training"].get("online_rollout_size", 32))
    ppo_epochs = int(cfg["ppo"]["training"]["ppo_epochs"])
    mini_batch_size = int(cfg["ppo"]["training"]["mini_batch_size"])
    policy_hidden_dim = int(cfg["ppo"]["model"]["hidden_dim"])
    val_ratio = float(cfg["ppo"]["training"].get("val_ratio", 0.2))
    clip_eps = float(cfg["ppo"]["training"]["clip_eps"])
    value_coef = float(cfg["ppo"]["training"]["value_coef"])
    entropy_coef = float(cfg["ppo"]["training"]["entropy_coef"])
    gamma = float(cfg["ppo"]["training"]["gamma"])
    lam = float(cfg["ppo"]["training"]["gae_lambda"])
    lr = float(cfg["ppo"]["training"]["lr"])
    shaping_coef = float(cfg["ppo"]["training"].get("shaping_coef", 0.0))

    all_prompts = load_train_prompts(train_path)
    logger.info("train_prompts=%d", len(all_prompts))

    # ---- vLLM engine ----
    from vllm import LLM

    if isinstance(tensor_parallel_size, str) and tensor_parallel_size == "auto":
        import os as _os
        visible = _os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if visible:
            devices = [d.strip() for d in visible.split(",") if d.strip() and d.strip() != "-1"]
            tp_size = max(1, len(devices))
        else:
            tp_size = 1
    else:
        tp_size = max(1, int(tensor_parallel_size))
    max_model_len = max_new_tokens + 2048
    gpu_mem = float(cfg["inference"].get("gpu_memory_utilization", 0.90))
    llm = LLM(
        model=cfg["inference"]["model_name_or_path"],
        tensor_parallel_size=tp_size, max_model_len=max_model_len,
        gpu_memory_utilization=gpu_mem,
    )
    tokenizer = llm.get_tokenizer()
    logger.info("vLLM ready tp_size=%d max_model_len=%d", tp_size, max_model_len)

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # ---- PPO policy ----
    policy = PolicyValueNet(obs_dim=obs_dim, n_actions=n_actions, hidden=policy_hidden_dim).to(device)

    # Best-fixed temperature for pi head init
    best_fixed_temp_idx = n_actions // 2
    from features.dataset_eval import load_temperature_labels
    temp_labels = load_temperature_labels(cfg["paths"]["val_dataset"])
    if temp_labels:
        per_temp_acc = {t: sum(lbls) / len(lbls) for t, lbls in temp_labels.items() if lbls}
        if per_temp_acc:
            best_temp = max(per_temp_acc, key=per_temp_acc.get)
            for i, t in enumerate(temp_bins):
                if abs(t - best_temp) < 1e-6:
                    best_fixed_temp_idx = i
                    break
    logger.info("best_fixed_temp=%.1f idx=%d", temp_bins[best_fixed_temp_idx], best_fixed_temp_idx)

    mil_model = None
    if mil_ckpt is not None:
        warm = load_mil_encoder_for_warmstart(mil_ckpt, device)
        if warm:
            try:
                policy.load_state_dict(warm, strict=False)
                logger.info("mil_warmstart loaded %d params", len(warm))
            except RuntimeError:
                logger.warning("mil_warmstart skipped — shape mismatch")
                warm = None
        try:
            mil_ckpt_data = torch.load(mil_ckpt, map_location=device, weights_only=False)
            mil_model = MILModel(
                input_dim=obs_dim, hidden_dim=hidden_dim,
                aggregator=cfg["mil"]["model"].get("aggregator", "attention"),
                use_position=cfg["mil"]["model"].get("use_position", True),
                use_gru=cfg["mil"]["model"].get("use_gru", True),
            ).to(device)
            mil_model.load_state_dict(mil_ckpt_data["mil"])
            mil_model.eval()
            logger.info("MIL model loaded for shaping rewards")
        except Exception:
            mil_model = None

    with torch.no_grad():
        policy.pi.bias.fill_(-5.0)
        policy.pi.bias[best_fixed_temp_idx] = 5.0
    logger.info("pi_head biased toward temp index %d (T=%.1f)", best_fixed_temp_idx, temp_bins[best_fixed_temp_idx])

    opt = optim.Adam(policy.parameters(), lr=lr)

    def render_prompt(question: str) -> str:
        if not use_math_chat:
            return question
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except Exception:
            return f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{question}\n\n[ASSISTANT]\n"

    from utils.answer_verifier import verify_answer, self_consistency_correct

    best_val_value = float("inf")
    patience_counter = 0
    best_state: dict | None = None

    for it in range(max_iterations):
        rng = random.Random(seed + it * 1000)
        batch_prompts = rng.sample(all_prompts, min(rollout_size, len(all_prompts)))
        N = len(batch_prompts)
        rendered = [render_prompt(p.get("question") or p.get("problem") or p.get("prompt", "")) for p in batch_prompts]
        gold_answers = [p.get("answer", "") for p in batch_prompts]

        V = num_votes
        generated: List[List[str]] = [[""] * V for _ in range(N)]
        active = [True] * N
        segment_obs: List[Optional[torch.Tensor]] = [None] * N
        ep_prefixes: List[str] = [rendered[i] for i in range(N)]  # accumulated for HS extraction

        ep_obs: List[List[torch.Tensor]] = [[] for _ in range(N)]
        ep_actions: List[List[torch.Tensor]] = [[] for _ in range(N)]
        ep_logprobs: List[List[torch.Tensor]] = [[] for _ in range(N)]
        ep_values: List[List[torch.Tensor]] = [[] for _ in range(N)]
        ep_correct: List[int] = [-1] * N  # 1 = majority correct, 0 = majority wrong, -1 = unknown

        max_rounds = max_new_tokens // segment_size + 1
        from vllm import SamplingParams

        for _seg_idx in range(max_rounds):
            round_prompts = []
            round_params = []
            round_indices = []

            for i in range(N):
                if not active[i]:
                    continue
                round_prompts.append(rendered[i] + generated[i][0])

                if segment_obs[i] is None:
                    # First segment: no prior generated tokens → no features to
                    # build an observation from.  Use a fixed default temperature.
                    # The dummy values below are safely skipped during PPO batch
                    # construction (range(1, n_steps)) — they only keep the
                    # per-episode lists aligned so t=0 is the first decision.
                    temp = 0.7
                    ep_obs[i].append(torch.zeros(obs_dim))
                    ep_actions[i].append(torch.tensor(0))
                    ep_logprobs[i].append(torch.tensor(0.0))
                    ep_values[i].append(torch.tensor(0.0))
                else:
                    obs_t = segment_obs[i].unsqueeze(0).to(device)
                    with torch.no_grad():
                        logits, value = policy(obs_t)
                        action, logp = sample_action(logits.squeeze(0))
                    temp = temp_bins[int(action.item())]
                    ep_obs[i].append(segment_obs[i].cpu())
                    ep_actions[i].append(action.cpu())
                    ep_logprobs[i].append(logp.cpu())
                    ep_values[i].append(value.squeeze(0).cpu())

                round_params.append(SamplingParams(
                    n=V, temperature=temp, max_tokens=segment_size, logprobs=top_k_logits,
                ))
                round_indices.append(i)

            if not round_indices:
                break

            outputs = llm.generate(round_prompts, round_params, use_tqdm=False)

            for j, i in enumerate(round_indices):
                req_outputs = outputs[j].outputs
                for v in range(min(V, len(req_outputs))):
                    generated[i][v] += req_outputs[v].text

                out0 = req_outputs[0]
                new_tokens = out0.token_ids

                finish_reason = getattr(out0, 'finish_reason', None)
                if (tokenizer.eos_token_id is not None and tokenizer.eos_token_id in new_tokens) or \
                   finish_reason == 'stop' or not new_tokens:
                    active[i] = False
                    ep_correct[i] = 1 if self_consistency_correct(generated[i], gold_answers[i]) else 0
                    continue

                if out0.logprobs:
                    # Accumulate prefix for hidden state extraction
                    seg_text = req_outputs[0].text
                    if seg_text:
                        ep_prefixes[i] += seg_text

                    # Extract hidden states for this segment's token positions
                    seg_hidden_states = None
                    if hs_extractor is not None and seg_text:
                        seg_hidden_states = hs_extractor.extract(
                            [ep_prefixes[i]], [seg_text]
                        )[0]  # one result per (prefix, text) pair

                    obs = _extract_segment_obs(
                        new_tokens, out0.logprobs, obs_dim, top_k_logits,
                        hidden_states=seg_hidden_states,
                        feature_mode=feature_mode,
                    )
                    segment_obs[i] = torch.tensor(obs, dtype=torch.float32) if obs else None

        for i in range(N):
            if ep_correct[i] == -1:
                ep_correct[i] = 1 if self_consistency_correct(generated[i], gold_answers[i]) else 0

        # ---- Build PPO batch ----
        # Walk each episode's recorded steps.  range(1, n_steps) skips t=0
        # (the first segment, which had no prior observation to base a decision
        # on).  The last step of each episode is marked done=True and receives
        # the terminal majority-vote reward (±1); intermediate steps receive 0
        # (or a MIL shaping reward if available).  done flags prevent GAE from
        # propagating advantage across episode boundaries.
        all_obs: List[torch.Tensor] = []
        all_actions: List[torch.Tensor] = []
        all_logprobs: List[torch.Tensor] = []
        all_rewards: List[float] = []
        all_dones: List[float] = []
        all_values: List[torch.Tensor] = []

        for i in range(N):
            n_steps = len(ep_actions[i])
            if n_steps <= 1:
                continue
            for t in range(1, n_steps):
                all_obs.append(ep_obs[i][t])
                all_actions.append(ep_actions[i][t])
                all_logprobs.append(ep_logprobs[i][t])
                all_values.append(ep_values[i][t])
                done = (t == n_steps - 1)
                all_dones.append(float(done))
                if done:
                    reward = 1.0 if ep_correct[i] > 0 else -1.0
                else:
                    reward = 0.0
                    if mil_model is not None and shaping_coef > 0:
                        seg_obs_t = ep_obs[i][t].unsqueeze(0).unsqueeze(0).to(device)
                        with torch.no_grad():
                            mil_out = mil_model(seg_obs_t)
                        reward = shaping_coef * float((1.0 - torch.sigmoid(mil_out["inst_logit"])).item())
                all_rewards.append(reward)

        if len(all_obs) < mini_batch_size:
            logger.info("iter=%d too_few_steps=%d skipping_update", it + 1, len(all_obs))
            continue

        obs_t = torch.stack(all_obs).to(device)
        act_t = torch.stack(all_actions).to(device)
        logp_t = torch.stack(all_logprobs).to(device)
        rew_t = torch.tensor(all_rewards, device=device, dtype=torch.float32)
        don_t = torch.tensor(all_dones, device=device, dtype=torch.float32)
        val_t = torch.stack(all_values).to(device)

        adv, ret = compute_gae(rew_t, don_t, val_t, gamma, lam)

        total_steps = len(all_obs)
        perm = torch.randperm(total_steps)
        n_train = int(total_steps * (1.0 - val_ratio))
        train_idx = perm[:n_train]
        val_idx = perm[n_train:]

        sum_policy = 0.0; sum_value = 0.0; sum_entropy = 0.0; n_updates = 0
        for _ in range(ppo_epochs):
            for start in range(0, n_train, mini_batch_size):
                end = min(start + mini_batch_size, n_train)
                mb_idx = train_idx[start:end]

                logits, values = policy(obs_t[mb_idx])
                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(act_t[mb_idx])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logp - logp_t[mb_idx])
                surr1 = ratio * adv[mb_idx]
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv[mb_idx]
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = ((ret[mb_idx] - values) ** 2).mean()
                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                opt.step()

                sum_policy += policy_loss.item()
                sum_value += value_loss.item()
                sum_entropy += entropy.item()
                n_updates += 1

        avg_policy = sum_policy / max(1, n_updates)
        avg_value = sum_value / max(1, n_updates)
        avg_entropy = sum_entropy / max(1, n_updates)

        val_value = 0.0
        if len(val_idx) > 0:
            with torch.no_grad():
                _logits_val, values_val = policy(obs_t[val_idx])
                val_value = ((ret[val_idx] - values_val) ** 2).mean().item()

        mean_reward = rew_t.mean().item()
        online_acc = sum(ep_correct) / max(1, len(ep_correct))

        logger.info(
            "iter=%d loss=%.4f policy=%.4f value=%.4f val_value=%.4f entropy=%.4f reward=%.4f acc=%.4f steps=%d updates=%d",
            it + 1,
            avg_policy + value_coef * avg_value - entropy_coef * avg_entropy,
            avg_policy, avg_value, val_value, avg_entropy,
            mean_reward, online_acc, total_steps, n_updates,
        )

        # ---- early stopping on val_value ----
        if val_value < best_val_value:
            best_val_value = val_value
            patience_counter = 0
            best_state = {"policy_value": policy.state_dict(), "config": cfg}
            logger.info("new_best val_value=%.4f", best_val_value)
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                logger.info("early_stop val_value=%.4f best=%.4f", val_value, best_val_value)
                break

    if best_state is None:
        best_state = {"policy_value": policy.state_dict(), "config": cfg}
    ckpt_path = cfg["paths"]["ppo_ckpt"]
    torch.save(best_state, ckpt_path)
    logger.info("saved_checkpoint=%s best_val_value=%.4f run_name=%s", ckpt_path, best_val_value, final_run_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-data", default=None, help="Override paths.train_dataset from config")
    parser.add_argument("--mil-ckpt", default=None, help="Override paths.mil_ckpt from config")
    parser.add_argument("--tensor-parallel-size", default="auto")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--log-dir", default="logs")
    args = parser.parse_args()
    cfg = _load_config(args.config)
    train_path = args.train_data or cfg["paths"]["train_dataset"]
    mil_ckpt = args.mil_ckpt or cfg["paths"]["mil_ckpt"]
    try:
        train_ppo(args.config, train_path, mil_ckpt=mil_ckpt, tensor_parallel_size=args.tensor_parallel_size, run_name=args.run_name, log_dir=args.log_dir)
    except Exception as exc:
        cfg = _load_config(args.config)
        logger, _, _ = setup_experiment_logger(component="train_ppo", run_name=args.run_name, log_dir=args.log_dir, config=cfg)
        log_exception(logger, exc)
        raise


if __name__ == "__main__":
    main()
