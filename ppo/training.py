"""Online PPO training — the policy controls vLLM generation temperature
segment-by-segment, receiving real math-verify rewards.

Usage (single-process; vLLM tensor parallelism handles multi-GPU):
    CUDA_VISIBLE_DEVICES=0,1 python -m ppo.training --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import json
import random
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.optim as optim
import yaml

from features.segmenter import build_segment_obs_from_lp
from utils.jsonl import sample_prefix
from mil.model import MILModel
from ppo.model import PolicyValueNet, sample_action, compute_gae, load_mil_encoder_for_warmstart
from utils.exp_logger import log_exception, setup_experiment_logger
from inference.vllm_runner import VLLMFeatureExporter


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def _decide_temperature(
    segment_obs: torch.Tensor | None,
    policy: PolicyValueNet,
    temp_bins: list[float],
    device: torch.device,
    deterministic: bool,
) -> tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select a temperature for the next generation round.

    When ``segment_obs`` is None (first round — no prior generated tokens),
    returns a default temperature of 0.7 with dummy action/logp/value.
    When ``deterministic=True``, uses argmax (for validation).
    When ``deterministic=False``, uses ``sample_action`` (for training).
    """
    if segment_obs is None:
        return 0.7, torch.tensor(0), torch.tensor(0.0), torch.tensor(0.0)

    obs_t = segment_obs[-1:].to(device)  # [1, obs_dim] — last segment, batch of 1
    assert obs_t.dim() == 2, \
        f"_decide_temperature: obs_t must be 2D [1, D], got {obs_t.shape}"
    with torch.no_grad():
        logits, value = policy(obs_t)
    assert logits.dim() == 2 and logits.shape[0] == 1, \
        f"_decide_temperature: logits must be [1, n_actions], got {logits.shape}"
    if deterministic:
        action = logits.argmax(dim=-1)
        logp = torch.tensor(0.0)
    else:
        action, logp = sample_action(logits.squeeze(0))
    temp = temp_bins[int(action.item())]
    return temp, action.cpu(), logp.cpu(), value.squeeze(0).cpu()


def _process_generated_features(
    feat_dict: dict,
    tokenizer: Any,
    segment_size: int,
    instance_dim: int,
    device: torch.device,
    segment_mode: str,
    hs_needed: bool,
    pooling_mode: str,
) -> tuple[str, bool, torch.Tensor | None]:
    """Process the output of ``generate_with_features`` for one chain.

    Returns ``(text_delta, is_done, next_segment_obs)``.  When the chain has
    terminated (EOS / stop / empty tokens), ``is_done`` is True and
    ``next_segment_obs`` is None.  Otherwise the segment observation for the
    next round is built via ``build_segment_obs_from_lp``.
    """
    text_delta = feat_dict["text"]
    new_tokens = feat_dict["token_ids"]
    finish_reason = feat_dict["finish_reason"]

    if (tokenizer.eos_token_id is not None and tokenizer.eos_token_id in new_tokens) or \
       finish_reason == "stop" or not new_tokens:
        return text_delta, True, None

    # APC can evict hidden states → logprobs may be None
    if feat_dict.get("logprobs") is None:
        return text_delta, True, None

    extra = [feat_dict["hidden_states"]] if hs_needed else None
    obs = build_segment_obs_from_lp(
        feat_dict["logprobs"], feat_dict["tokens"], feat_dict["text"],
        segment_size, instance_dim, device=device, extra_parts=extra,
        segment_mode=segment_mode,
        include_topk=(not hs_needed),
        pooling_mode=pooling_mode,
    )
    return text_delta, False, obs.cpu()


def train_ppo(
    config_path: str,
    train_path: str,
    mil_ckpt: str | None = None,
    parallel_size: int | None = None,
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

    # ---- Metrics JSONL ----
    metrics_path = f"{log_dir}/{final_run_name}_ppo_metrics.jsonl"
    metrics_fh = open(metrics_path, "a", encoding="utf-8")
    logger.info("metrics_jsonl=%s", metrics_path)

    instance_dim = int(cfg["data"]["instance_dim"])
    segment_size = int(cfg["data"]["segment_size"])
    pooling_mode = cfg["data"].get("segment_pooling", "mean")
    model_obs_dim = instance_dim * segment_size if pooling_mode == "concat" else instance_dim
    temp_bins = [float(x) for x in cfg["data"]["temp_bins"]]
    n_actions = len(temp_bins)
    segment_mode = cfg["data"].get("segment_mode", "fixed_window")
    if segment_mode not in ("fixed_window",):
        raise ValueError(
            f"segment_mode='{segment_mode}' is not yet supported in PPO training. "
            f"Only 'fixed_window' is currently implemented. "
            f"Step/punctuation modes would produce variable segment counts per round, "
            f"which breaks torch.stack at batch construction time. "
            f"To support this, the per-step observation storage and batch assembly "
            f"in train_ppo() need to be updated to handle variable-K segments."
        )
    max_new_tokens = int(cfg["inference"]["max_new_tokens"])
    top_k_logprobs = int(cfg["inference"]["top_k_logprobs"])
    num_votes = int(cfg["inference"].get("num_votes", 1))
    system_prompt = cfg["inference"].get("system_prompt", "")
    use_math_chat = bool(cfg["inference"].get("use_math_chat_prompt", True))
    feature_mode = cfg["inference"].get("feature_mode", "topk_logprobs")
    hs_needed = feature_mode == "hidden_states"

    max_iterations = int(cfg["ppo"]["training"]["max_iterations"])
    early_stop_patience = int(cfg["ppo"]["training"]["early_stop_patience"])
    rollout_size = int(cfg["ppo"]["training"].get("online_rollout_size", 32))
    ppo_epochs = int(cfg["ppo"]["training"]["ppo_epochs"])
    mini_batch_size = int(cfg["ppo"]["training"]["mini_batch_size"])
    policy_hidden_dim = int(cfg["ppo"]["model"]["hidden_dim"])
    val_size = int(cfg["ppo"]["training"].get("val_size", 16))
    clip_eps = float(cfg["ppo"]["training"]["clip_eps"])
    value_coef = float(cfg["ppo"]["training"]["value_coef"])
    entropy_coef = float(cfg["ppo"]["training"]["entropy_coef"])
    gamma = float(cfg["ppo"]["training"]["gamma"])
    lam = float(cfg["ppo"]["training"]["gae_lambda"])
    lr = float(cfg["ppo"]["training"]["lr"])

    all_prompts = load_train_prompts(train_path)
    logger.info("train_prompts=%d", len(all_prompts))

    val_prompts = load_train_prompts(cfg["paths"]["val_dataset"])
    val_rng = random.Random(seed)
    val_fixed = val_rng.sample(val_prompts, min(val_size, len(val_prompts)))
    logger.info("val_fixed=%d", len(val_fixed))

    # ---- Inference engine ----
    model_path = cfg["inference"]["model_name_or_path"]
    gpu_mem = float(cfg["inference"].get("gpu_memory_utilization", 0.90))

    runner = VLLMFeatureExporter(
        model_name_or_path=model_path,
        max_new_tokens=max_new_tokens,
        parallel_size=parallel_size,
        gpu_memory_utilization=gpu_mem,
        reserve_training_gpu=True,
    )
    tokenizer = runner.tokenizer
    logger.info("VLLMFeatureExporter ready")

    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    device = torch.device(f"cuda:{max(0, n_gpu - 1)}") if n_gpu > 0 else torch.device("cpu")

    # ---- PPO policy ----
    policy = PolicyValueNet(obs_dim=model_obs_dim, n_actions=n_actions, hidden=policy_hidden_dim).to(device)

    # Best-fixed temperature for pi head init
    best_fixed_temp_idx = n_actions // 2
    from features.dataset_eval import load_temperature_labels
    temp_labels = load_temperature_labels(train_path)
    if temp_labels:
        per_temp_acc = {t: sum(lbls) / len(lbls) for t, lbls in temp_labels.items() if lbls}
        if per_temp_acc:
            best_temp = max(per_temp_acc, key=per_temp_acc.get)
            temp_to_idx = {t: i for i, t in enumerate(temp_bins)}
            best_fixed_temp_idx = temp_to_idx.get(best_temp, n_actions // 2)
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
            hidden_dim = int(cfg["mil"]["model"]["hidden_dim"])
            mil_model = MILModel(
                input_dim=model_obs_dim, hidden_dim=hidden_dim,
                aggregator=cfg["mil"]["model"].get("aggregator", "attention"),
                use_position=cfg["mil"]["model"].get("use_position", True),
                use_gru=cfg["mil"]["model"].get("use_gru", True),
                gated_attention=cfg["mil"]["model"].get("gated_attention", False),
                num_heads=int(cfg["mil"]["model"].get("num_heads", 1)),
            ).to(device)
            mil_model.load_state_dict(mil_ckpt_data["mil"])
            mil_model.eval()
            logger.info("MIL model loaded for shaping rewards")
        except (FileNotFoundError, RuntimeError, KeyError) as e:
            logger.warning("MIL model not loaded for shaping rewards: %s", e)
            mil_model = None

    with torch.no_grad():
        policy.pi.bias.fill_(-5.0)
        policy.pi.bias[best_fixed_temp_idx] = 5.0
    logger.info("pi_head biased toward temp index %d (T=%.1f)", best_fixed_temp_idx, temp_bins[best_fixed_temp_idx])

    opt = optim.Adam(policy.parameters(), lr=lr)

    from utils.answer_verifier import self_consistency_correct

    best_val_value = float("-inf")
    patience_counter = 0
    best_state: dict | None = None

    for it in range(max_iterations):
        rng = random.Random(seed + it * 1000)
        batch_prompts = rng.sample(all_prompts, min(rollout_size, len(all_prompts)))
        N = len(batch_prompts)
        rendered = [
            runner.render_messages(runner.build_math_messages(
                p.get("question") or p.get("problem") or p.get("prompt", ""),
                system_prompt=system_prompt,
            )) if use_math_chat else (p.get("question") or p.get("problem") or p.get("prompt", ""))
            for p in batch_prompts
        ]
        gold_answers = [p.get("answer", "") for p in batch_prompts]

        V = num_votes
        generated: List[List[str]] = [[""] * V for _ in range(N)]
        active: List[List[bool]] = [[True] * V for _ in range(N)]
        segment_obs: List[List[Optional[torch.Tensor]]] = [[None] * V for _ in range(N)]
        ep_obs: List[List[List[torch.Tensor]]] = [[[] for _ in range(V)] for _ in range(N)]
        ep_actions: List[List[List[torch.Tensor]]] = [[[] for _ in range(V)] for _ in range(N)]
        ep_logprobs: List[List[List[torch.Tensor]]] = [[[] for _ in range(V)] for _ in range(N)]
        ep_values: List[List[List[torch.Tensor]]] = [[[] for _ in range(V)] for _ in range(N)]
        ep_correct: List[int] = [-1] * N  # 1 = majority correct, 0 = majority wrong, -1 = unknown

        iter_temp_counts: Dict[float, int] = {t: 0 for t in temp_bins}

        max_rounds = max_new_tokens // segment_size

        for _seg_idx in range(max_rounds):
            round_prompts: List[str] = []
            round_temps: List[float] = []
            round_map: List[Tuple[int, int]] = []  # (prompt_idx, chain_idx)

            for i in range(N):
                for v in range(V):
                    if not active[i][v]:
                        continue
                    round_prompts.append(rendered[i] + generated[i][v])

                    temp, action, logp, value = _decide_temperature(
                        segment_obs[i][v], policy, temp_bins, device, deterministic=False,
                    )
                    # ep_obs stores 1D [obs_dim] tensors uniformly:
                    # step 0 (dummy) and steps 1+ (squeezed from [1, obs_dim]).
                    # This keeps torch.stack friendly — no spurious dim-1 axis.
                    ep_obs[i][v].append(
                        torch.zeros(model_obs_dim)
                        if segment_obs[i][v] is None
                        else segment_obs[i][v].squeeze(0).cpu()
                    )
                    ep_actions[i][v].append(action)
                    ep_logprobs[i][v].append(logp)
                    ep_values[i][v].append(value)

                    round_temps.append(temp)
                    round_map.append((i, v))
                    iter_temp_counts[float(temp)] += 1

            if not round_map:
                break

            feats = runner.generate_with_features(
                round_prompts, round_temps, segment_size,
                top_k=top_k_logprobs,
                return_logprobs=True,
                return_hidden=hs_needed,
            )

            for j, (i, v) in enumerate(round_map):
                f = feats[j]
                text_delta, done, next_obs = _process_generated_features(
                    f, tokenizer, segment_size, instance_dim, device,
                    segment_mode, hs_needed, pooling_mode,
                )
                generated[i][v] += text_delta
                if done:
                    active[i][v] = False
                    continue
                segment_obs[i][v] = next_obs

        for i in range(N):
            if ep_correct[i] == -1:
                ep_correct[i] = 1 if self_consistency_correct(generated[i], gold_answers[i]) else 0

        # ---- Build PPO batch ----
        # Each chain is an independent episode.  range(1, n_steps) skips t=0
        # (the first segment, which had no prior observation to base a decision
        # on).  The last step of each chain is marked done=True and receives
        # the terminal majority-vote reward (±1) shared across all chains of
        # the same prompt.  Intermediate steps receive 0 (or MIL shaping reward).
        all_obs: List[torch.Tensor] = []
        all_actions: List[torch.Tensor] = []
        all_logprobs: List[torch.Tensor] = []
        all_rewards: List[float] = []
        all_dones: List[float] = []
        all_values: List[torch.Tensor] = []

        for i in range(N):
            terminal_reward = 1.0 if ep_correct[i] > 0 else -1.0
            for v in range(V):
                n_steps = len(ep_actions[i][v])
                if n_steps <= 1:
                    continue

                # Distribute terminal reward across all steps.
                # Correct chains: uniform — MIL attention has no error to localize.
                # Incorrect chains with MIL: attention-weighted credit assignment.
                # No MIL: uniform fallback.
                n_eff = n_steps - 1  # skip dummy step 0
                if terminal_reward < 0 and mil_model is not None and n_eff > 0:
                    # ep_obs entries are already 1D [obs_dim]; stack → [K, obs_dim]
                    full_bag = torch.stack(ep_obs[i][v][1:]).unsqueeze(0).to(device)
                    with torch.no_grad():
                        raw_w = mil_model(full_bag)["attn_w"].squeeze(0)
                    weights = raw_w / (raw_w.sum() + 1e-8)
                else:
                    weights = torch.full((n_eff,), 1.0 / max(n_eff, 1))

                for t in range(1, n_steps):
                    # ep_obs entries are uniformly 1D [obs_dim] (see storage contract above)
                    all_obs.append(ep_obs[i][v][t])
                    all_actions.append(ep_actions[i][v][t])
                    all_logprobs.append(ep_logprobs[i][v][t])
                    all_values.append(ep_values[i][v][t])
                    done = (t == n_steps - 1)
                    all_dones.append(float(done))
                    reward = terminal_reward * float(weights[t - 1])
                    all_rewards.append(reward)

        if len(all_obs) < mini_batch_size:
            logger.info("iter=%d too_few_steps=%d skipping_update", it + 1, len(all_obs))
            continue

        obs_t = torch.stack(all_obs).to(device)
        # Shape contract: obs_t must be 2D [N, obs_dim].  A 3D [N, 1, obs_dim]
        # corrupts downstream Categorical.log_prob and value loss broadcasting.
        assert obs_t.dim() == 2, \
            f"PPO batch: obs_t must be 2D [N, D], got {obs_t.shape}. " \
            f"Did ep_obs storage miss a squeeze?"
        act_t = torch.stack(all_actions).to(device)
        logp_t = torch.stack(all_logprobs).to(device)
        rew_t = torch.tensor(all_rewards, device=device, dtype=torch.float32)
        don_t = torch.tensor(all_dones, device=device, dtype=torch.float32)
        val_t = torch.stack(all_values).to(device)

        # All batch tensors must be 1D to avoid silent broadcast in compute_gae
        # and Categorical.log_prob.  obs_t is the exception: 2D [N, obs_dim].
        N = obs_t.shape[0]
        for name, t in [("act", act_t), ("logp", logp_t), ("rew", rew_t),
                         ("don", don_t), ("val", val_t)]:
            assert t.dim() == 1, \
                f"PPO batch: {name}_t must be 1D [N], got {t.shape}"
            assert t.shape[0] == N, \
                f"PPO batch: {name}_t length {t.shape[0]} != obs_t length {N}"

        adv, ret = compute_gae(rew_t, don_t, val_t, gamma, lam)

        total_steps = len(all_obs)
        perm = torch.randperm(total_steps)

        sum_policy = 0.0; sum_value = 0.0; sum_entropy = 0.0; n_updates = 0
        sum_clipped = 0; sum_total_ratio = 0
        for _ in range(ppo_epochs):
            for start in range(0, total_steps, mini_batch_size):
                end = min(start + mini_batch_size, total_steps)
                mb_idx = perm[start:end]

                logits, values = policy(obs_t[mb_idx])
                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(act_t[mb_idx])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logp - logp_t[mb_idx])
                clipped_mask = (ratio < 1.0 - clip_eps) | (ratio > 1.0 + clip_eps)
                sum_clipped += clipped_mask.sum().item()
                sum_total_ratio += ratio.numel()

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

        mean_reward = rew_t.mean().item()
        train_acc = sum(ep_correct) / max(1, len(ep_correct))

        # ---- val rollout on fixed set ----
        val_acc = 0.0
        if val_fixed:
            val_N = len(val_fixed)
            val_rendered = [
                runner.render_messages(runner.build_math_messages(
                    p.get("question") or p.get("problem") or p.get("prompt", ""),
                    system_prompt=system_prompt,
                )) if use_math_chat else (p.get("question") or p.get("problem") or p.get("prompt", ""))
                for p in val_fixed
            ]
            val_gold = [p.get("answer", "") for p in val_fixed]
            val_generated: List[List[str]] = [[""] * V for _ in range(val_N)]
            val_active: List[List[bool]] = [[True] * V for _ in range(val_N)]
            val_seg_obs: List[List[Optional[torch.Tensor]]] = [[None] * V for _ in range(val_N)]

            with torch.no_grad():
                for _seg_idx in range(max_rounds):
                    val_prompts: List[str] = []
                    val_temps: List[float] = []
                    val_map: List[Tuple[int, int]] = []

                    for i in range(val_N):
                        for v in range(V):
                            if not val_active[i][v]:
                                continue
                            val_prompts.append(val_rendered[i] + val_generated[i][v])

                            temp, _, _, _ = _decide_temperature(
                                val_seg_obs[i][v], policy, temp_bins, device, deterministic=True,
                            )
                            val_temps.append(temp)
                            val_map.append((i, v))

                    if not val_map:
                        break

                    val_feats = runner.generate_with_features(
                        val_prompts, val_temps, segment_size,
                        top_k=top_k_logprobs,
                        return_logprobs=True,
                        return_hidden=hs_needed,
                    )

                    for j, (i, v) in enumerate(val_map):
                        f = val_feats[j]
                        text_delta, done, next_obs = _process_generated_features(
                            f, tokenizer, segment_size, instance_dim, device,
                            segment_mode, hs_needed, pooling_mode,
                        )
                        val_generated[i][v] += text_delta
                        if done:
                            val_active[i][v] = False
                        else:
                            val_seg_obs[i][v] = next_obs

            val_correct = sum(
                1 if self_consistency_correct(val_generated[i], val_gold[i]) else 0
                for i in range(val_N)
            )
            val_acc = val_correct / val_N

        logger.info(
            "iter=%d loss=%.4f policy=%.4f value=%.4f entropy=%.4f reward=%.4f train_acc=%.4f val_acc=%.4f steps=%d updates=%d",
            it + 1,
            avg_policy + value_coef * avg_value - entropy_coef * avg_entropy,
            avg_policy, avg_value, avg_entropy,
            mean_reward, train_acc, val_acc, total_steps, n_updates,
        )

        # ---- Iteration metrics ----
        reward_pos_ratio = sum(1 for c in ep_correct if c > 0) / max(1, len(ep_correct))
        all_temps_flat: List[float] = []
        seg_lengths: List[int] = []
        n_prompts = len(batch_prompts)
        for i in range(n_prompts):
            for v in range(V):
                n_steps = len(ep_actions[i][v])
                if n_steps > 1:
                    seg_lengths.append(n_steps - 1)  # exclude dummy t=0
                for t_idx in range(1, n_steps):
                    # Reconstruct temperature from the action index
                    act_val = int(ep_actions[i][v][t_idx].item())
                    all_temps_flat.append(float(temp_bins[act_val]))
        temp_mean = float(sum(all_temps_flat) / max(1, len(all_temps_flat)))
        temp_std = float((sum((x - temp_mean) ** 2 for x in all_temps_flat) / max(1, len(all_temps_flat))) ** 0.5)
        seg_mean = sum(seg_lengths) / max(1, len(seg_lengths))
        seg_min = min(seg_lengths) if seg_lengths else 0
        seg_max = max(seg_lengths) if seg_lengths else 0
        adv_mean = float(adv.mean().item())
        adv_std = float(adv.std().item())
        clip_fraction = sum_clipped / max(1, sum_total_ratio)

        metrics_fh.write(json.dumps({
            "iter": it + 1,
            "total_loss": round(avg_policy + value_coef * avg_value - entropy_coef * avg_entropy, 4),
            "policy_loss": round(avg_policy, 4),
            "value_loss": round(avg_value, 4),
            "entropy": round(avg_entropy, 4),
            "reward_mean": round(mean_reward, 4),
            "reward_pos_ratio": round(reward_pos_ratio, 4),
            "train_acc": round(train_acc, 4),
            "val_acc": round(val_acc, 4),
            "temp_dist": {f"{t:.1f}": iter_temp_counts[t] for t in temp_bins},
            "temp_mean": round(temp_mean, 4),
            "temp_std": round(temp_std, 4),
            "segments_mean": round(seg_mean, 2),
            "segments_min": seg_min,
            "segments_max": seg_max,
            "advantage_mean": round(adv_mean, 4),
            "advantage_std": round(adv_std, 4),
            "clip_fraction": round(clip_fraction, 4),
            "total_steps": total_steps,
        }) + "\n")
        metrics_fh.flush()

        # ---- early stopping on val accuracy ----
        if val_acc > best_val_value:
            best_val_value = val_acc
            patience_counter = 0
            best_state = {"policy_value": {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}, "config": cfg}
            logger.info("new_best val_acc=%.4f", best_val_value)
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                logger.info("early_stop val_acc=%.4f best=%.4f", val_acc, best_val_value)
                break

    if best_state is None:
        best_state = {"policy_value": {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}, "config": cfg}
    ckpt_path = cfg["paths"]["ppo_ckpt"]
    torch.save(best_state, ckpt_path)
    logger.info("saved_checkpoint=%s best_val_acc=%.4f run_name=%s", ckpt_path, best_val_value, final_run_name)

    metrics_fh.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-data", default=None, help="Override paths.train_dataset from config")
    parser.add_argument("--mil-ckpt", default=None, help="Override paths.mil_ckpt from config")
    parser.add_argument("--parallel-size", type=int, default=None,
                        help="Override inference.parallel_size from config")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--log-dir", default="logs")
    args = parser.parse_args()
    cfg = _load_config(args.config)
    train_path = args.train_data or cfg["paths"]["train_dataset"]
    mil_ckpt = args.mil_ckpt or cfg["paths"]["mil_ckpt"]
    try:
        train_ppo(args.config, train_path, mil_ckpt=mil_ckpt,
                  parallel_size=args.parallel_size, run_name=args.run_name, log_dir=args.log_dir)
    except Exception as exc:
        cfg = _load_config(args.config)
        logger, _, _ = setup_experiment_logger(component="train_ppo", run_name=args.run_name, log_dir=args.log_dir, config=cfg)
        log_exception(logger, exc)
        raise


if __name__ == "__main__":
    main()
