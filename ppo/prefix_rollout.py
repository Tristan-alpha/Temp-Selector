"""Shared online rollout engine for prefix-value PPO training and evaluation."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import torch

from features.segmenter import batch_build_masked_concat_segment_obs_from_lp
from mil.prefix_value import PrefixRecurrentState, PrefixValueModel, calibrated_probability, potential_reward
from ppo.model import PrefixPolicyValueNet, sample_action
from utils.answer_verifier import extract_answer, verify_answer, verify_answer_by_value
from utils.calibration import answer_entropy


NO_ANSWER = "<NO_ANSWER>"


@dataclass
class PrefixTransition:
    observation: torch.Tensor
    action: torch.Tensor
    logprob: torch.Tensor
    value: torch.Tensor
    temperature: float
    segment_index: int
    phi_before: float
    phi_after: float
    done: bool
    reward: float = 0.0
    terminal_reward: float = 0.0
    shaping_reward: float = 0.0
    final_correct: int = 0


@dataclass
class PrefixRolloutResult:
    majority_correct: List[int]
    individual_correct: List[List[int]]
    generated: List[List[str]]
    extracted_answers: List[List[str]]
    majority_answers: List[str]
    majority_counts: List[int]
    sc_confidences: List[float]
    answer_entropies: List[float]
    temperatures: List[List[float]]
    segment_counts: List[List[int]]
    token_counts: List[List[int]]
    transitions: List[List[List[PrefixTransition]]] = field(default_factory=list)


class PrefixRolloutEngine:
    def __init__(self, runner, value_model: PrefixValueModel,
                 calibration_temperature: float, device: torch.device,
                 temp_bins: List[float], segment_size: int, token_dim: int,
                 top_k_logprobs: int, num_votes: int,
                 max_new_tokens: int, gamma: float, shaping_coef: float,
                 system_prompt: str, use_math_chat: bool):
        self.runner = runner
        self.value_model = value_model
        self.calibration_temperature = calibration_temperature
        self.device = device
        self.temp_bins = temp_bins
        self.segment_size = segment_size
        self.token_dim = token_dim
        self.top_k_logprobs = top_k_logprobs
        self.num_votes = num_votes
        self.max_new_tokens = max_new_tokens
        self.gamma = gamma
        self.shaping_coef = shaping_coef
        self.system_prompt = system_prompt
        self.use_math_chat = use_math_chat
        self.tokenizer = runner.tokenizer

    def _render(self, question: str) -> str:
        if not self.use_math_chat:
            return question
        return self.runner.render_messages(
            self.runner.build_math_messages(question, system_prompt=self.system_prompt),
        )

    @torch.no_grad()
    def rollout(self, prompts_data: List[Dict[str, Any]], policy: PrefixPolicyValueNet,
                stochastic: bool, rng: random.Random,
                collect_transitions: bool = True,
                generation_seed: int | None = None,
                fixed_temperature: float | None = None,
                random_temperature: bool = False) -> PrefixRolloutResult:
        n_prompts = len(prompts_data)
        votes = self.num_votes
        rendered = [self._render(str(p.get("question", p.get("prompt", "")))) for p in prompts_data]
        gold = [str(p.get("answer", "")) for p in prompts_data]
        generated = [[""] * votes for _ in range(n_prompts)]
        active = [[True] * votes for _ in range(n_prompts)]
        states: List[List[Optional[PrefixRecurrentState]]] = [[None] * votes for _ in range(n_prompts)]
        hidden_obs: List[List[Optional[torch.Tensor]]] = [[None] * votes for _ in range(n_prompts)]
        potentials: List[List[Optional[float]]] = [[None] * votes for _ in range(n_prompts)]
        temperatures: List[List[float]] = [[] for _ in range(n_prompts)]
        segment_counts = [[0] * votes for _ in range(n_prompts)]
        token_counts = [[0] * votes for _ in range(n_prompts)]
        transitions: List[List[List[PrefixTransition]]] = [
            [[] for _ in range(votes)] for _ in range(n_prompts)
        ]
        use_prompt_hidden = int(getattr(self.value_model, "prompt_dim", 0)) > 0

        max_rounds = max(1, (self.max_new_tokens + self.segment_size - 1) // self.segment_size)
        for segment_idx in range(max_rounds):
            round_prompts: List[str] = []
            round_temps: List[float] = []
            round_map: List[Tuple[int, int]] = []
            pending: List[Optional[Dict[str, Any]]] = []
            for i in range(n_prompts):
                for vote in range(votes):
                    if not active[i][vote]:
                        continue
                    round_prompts.append(rendered[i] + generated[i][vote])
                    if fixed_temperature is not None:
                        temperature = float(fixed_temperature)
                        decision = None
                    elif random_temperature:
                        temperature = float(rng.choice(self.temp_bins))
                        decision = None
                    elif hidden_obs[i][vote] is None:
                        temperature = 0.7
                        decision = None
                    else:
                        obs = hidden_obs[i][vote].unsqueeze(0).to(self.device)
                        logits, value = policy(obs)
                        if stochastic:
                            action, logprob = sample_action(logits.squeeze(0))
                        else:
                            action = logits.argmax(dim=-1).squeeze(0)
                            dist = torch.distributions.Categorical(logits=logits.squeeze(0))
                            logprob = dist.log_prob(action)
                        temperature = self.temp_bins[int(action.item())]
                        decision = {
                            "observation": obs.squeeze(0).cpu(),
                            "action": action.cpu(),
                            "logprob": logprob.cpu(),
                            "value": value.squeeze(0).cpu(),
                            "phi_before": float(potentials[i][vote]),
                            "temperature": float(temperature),
                        }
                    temperatures[i].append(temperature)
                    round_temps.append(temperature)
                    round_map.append((i, vote))
                    pending.append(decision)

            if not round_map:
                break
            features = self.runner.generate_with_features(
                round_prompts, round_temps, self.segment_size,
                top_k=self.top_k_logprobs,
                return_logprobs=True, return_hidden=False,
                return_prompt_hidden=use_prompt_hidden,
                device=self.device,
                seeds=[
                    (generation_seed + segment_idx * n_prompts * votes + i * votes + vote)
                    if generation_seed is not None else rng.randrange(2**31)
                    for i, vote in round_map
                ],
            )

            valid_positions: List[int] = []
            lp_tensors: List[torch.Tensor] = []
            token_lists: List[List[str]] = []
            texts: List[str] = []
            prompt_hiddens: List[torch.Tensor] = []
            done_flags: List[bool] = []
            for pos, ((i, vote), item) in enumerate(zip(round_map, features)):
                generated[i][vote] += item["text"]
                n_tokens = len(item["token_ids"])
                token_counts[i][vote] += n_tokens
                segment_counts[i][vote] += 1
                done = (
                    not item["token_ids"] or item["finish_reason"] == "stop" or
                    (self.tokenizer.eos_token_id is not None and
                     self.tokenizer.eos_token_id in item["token_ids"])
                )
                done_flags.append(done)
                if item["logprobs"] is not None and n_tokens > 0:
                    valid_positions.append(pos)
                    lp_tensors.append(item["logprobs"])
                    token_lists.append(item["tokens"])
                    texts.append(item["text"])
                    if use_prompt_hidden:
                        prompt_hidden = item.get("prompt_hidden")
                        if prompt_hidden is None:
                            raise RuntimeError("prompt-aware PVM rollout requires prompt_hidden")
                        prompt_hiddens.append(prompt_hidden)

            masked_list = batch_build_masked_concat_segment_obs_from_lp(
                lp_tensors, token_lists, texts,
                segment_size=self.segment_size, token_dim=self.token_dim,
                device=self.device, segment_mode="fixed_window",
            ) if lp_tensors else []
            position_to_masked = dict(zip(valid_positions, masked_list))
            position_to_prompt_hidden = (
                dict(zip(valid_positions, prompt_hiddens)) if use_prompt_hidden else {}
            )

            # Active chains in the same generation round share a prefix position.
            step_positions = [pos for pos in valid_positions]
            if step_positions:
                step_features = torch.stack([
                    position_to_masked[pos].features[0] for pos in step_positions
                ]).to(self.device)
                step_masks = torch.stack([
                    position_to_masked[pos].token_mask[0] for pos in step_positions
                ]).to(self.device)
                hidden_parts = []
                for pos in step_positions:
                    i, vote = round_map[pos]
                    state = states[i][vote]
                    if state is None:
                        if use_prompt_hidden:
                            hidden_parts.append(self.value_model.initial_hidden(
                                position_to_prompt_hidden[pos].unsqueeze(0).to(self.device)
                            ))
                        else:
                            hidden_parts.append(torch.zeros(
                                1, 1, self.value_model.hidden_dim, device=self.device,
                            ))
                    else:
                        hidden_parts.append(state.hidden.to(self.device))
                hidden = torch.cat(hidden_parts, dim=1)
                logits, encoded, next_hidden = self.value_model.step_batch(
                    step_features, step_masks, hidden=hidden, position=segment_idx,
                )
                probs = calibrated_probability(logits, self.calibration_temperature)
                for batch_idx, pos in enumerate(step_positions):
                    i, vote = round_map[pos]
                    states[i][vote] = PrefixRecurrentState(
                        next_hidden[:, batch_idx:batch_idx + 1].detach(), segment_idx + 1,
                    )
                    hidden_obs[i][vote] = encoded[batch_idx].detach()
                    potentials[i][vote] = float(probs[batch_idx].item())

            for pos, ((i, vote), decision, done) in enumerate(zip(round_map, pending, done_flags)):
                phi_after = potentials[i][vote]
                if decision is not None:
                    transitions[i][vote].append(PrefixTransition(
                        observation=decision["observation"],
                        action=decision["action"],
                        logprob=decision["logprob"],
                        value=decision["value"],
                        temperature=decision["temperature"],
                        segment_index=segment_idx,
                        phi_before=decision["phi_before"],
                        phi_after=float(phi_after if phi_after is not None else decision["phi_before"]),
                        done=done,
                    ))
                if done:
                    active[i][vote] = False

        majority_correct: List[int] = []
        individual_correct: List[List[int]] = []
        extracted_answers: List[List[str]] = []
        majority_answers: List[str] = []
        majority_counts: List[int] = []
        sc_confidences: List[float] = []
        answer_entropies: List[float] = []
        for i in range(n_prompts):
            answers = [
                answer if (answer := extract_answer(text)) is not None else NO_ANSWER
                for text in generated[i]
            ]
            counts = Counter(answers)
            majority_answer, majority_count = (
                counts.most_common(1)[0] if counts else (NO_ANSWER, 0)
            )
            majority = int(
                majority_answer != NO_ANSWER and
                verify_answer_by_value(majority_answer, gold[i])
            )
            majority_correct.append(majority)
            individual_correct.append([int(verify_answer(text, gold[i])) for text in generated[i]])
            extracted_answers.append(answers)
            majority_answers.append(str(majority_answer))
            majority_counts.append(int(majority_count))
            sc_confidences.append(float(majority_count / max(1, len(generated[i]))))
            answer_entropies.append(answer_entropy(answers))
            for vote in range(votes):
                vote_correct = int(individual_correct[i][vote])
                terminal = 1.0 if vote_correct else -1.0
                chain = transitions[i][vote]
                if chain:
                    chain[-1].done = True
                for transition in chain:
                    reward = float(potential_reward(
                        transition.phi_before, transition.phi_after,
                        gamma=self.gamma, shaping_coef=self.shaping_coef,
                        terminal_reward=terminal if transition.done else None,
                    ).item())
                    terminal_component = terminal if transition.done else 0.0
                    transition.reward = reward
                    transition.terminal_reward = float(terminal_component)
                    transition.shaping_reward = float(reward - terminal_component)
                    transition.final_correct = vote_correct

        if not collect_transitions:
            transitions = []
        return PrefixRolloutResult(
            majority_correct=majority_correct,
            individual_correct=individual_correct,
            generated=generated,
            extracted_answers=extracted_answers,
            majority_answers=majority_answers,
            majority_counts=majority_counts,
            sc_confidences=sc_confidences,
            answer_entropies=answer_entropies,
            temperatures=temperatures,
            segment_counts=segment_counts,
            token_counts=token_counts,
            transitions=transitions,
        )
