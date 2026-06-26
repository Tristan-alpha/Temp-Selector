#!/usr/bin/env python3
"""Extract prefix-end layer hidden states for layer-wise value lens probes.

This script intentionally keeps the multi-layer extraction path separate from
``VLLMFeatureExporter`` because the production exporter is configured around the
final-layer feature path used by PVM/PPO training.
"""

from __future__ import annotations

import argparse
import atexit
import gc
import json
import os
import shutil
import socket
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import yaml
from transformers import AutoConfig, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.vllm_runner import _cleanup_hidden_states_file, _load_hidden_states_file
from utils.jsonl import load_jsonl


@dataclass(frozen=True)
class PrefixRecord:
    record_index: int
    split: str
    problem_id: str
    source_sample_id: str
    prefix_segments: int
    prefix_stage: str
    prefix_token_end: int
    n_correct: int
    n_total: int
    source_individual_label: int


@dataclass(frozen=True)
class ExtractionJob:
    source_sample_id: str
    prompt_ids: list[int]
    response_ids: list[int]
    max_prefix_token_end: int
    prefix_token_ends: tuple[int, ...] = ()


class _LayerEndpointLogitFeatureFn:
    """Compute endpoint logit-lens features for layer hidden states in vLLM."""

    def __init__(self, hidden_states_cpu: torch.Tensor, token_ids_cpu: torch.Tensor):
        self.hidden_states_cpu = hidden_states_cpu
        self.token_ids_cpu = token_ids_cpu

    def __call__(self, model):
        dev = next(model.parameters()).device
        h = self.hidden_states_cpu.to(dev, non_blocking=True)
        ids = self.token_ids_cpu.to(dev, non_blocking=True)
        outputs = []
        for layer_pos in range(h.shape[1]):
            normed = model.model.norm(h[:, layer_pos, :])
            logits = model.compute_logits(normed).float()
            log_probs = torch.log_softmax(logits, dim=-1)
            probs = torch.exp(log_probs)
            entropy = -(probs * log_probs).sum(dim=-1)
            sampled = log_probs.gather(1, ids.unsqueeze(1)).squeeze(1)
            top2 = torch.topk(log_probs, k=2, dim=-1).values
            top1 = top2[:, 0]
            margin = top2[:, 0] - top2[:, 1]
            outputs.append(torch.stack([entropy, sampled, top1, margin], dim=1))
            del logits, log_probs, probs, top2
        return torch.stack(outputs, dim=1).cpu()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(data), indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(json_safe(row), sort_keys=True) + "\n")


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def continuation_path(cfg: Mapping[str, Any], split: str) -> Path:
    key = f"{split}_continuations"
    try:
        return Path(str(cfg["paths"][key]))
    except KeyError as exc:
        raise KeyError(f"config paths.{key} is required for split={split}") from exc


def dataset_path(cfg: Mapping[str, Any], split: str) -> Path:
    key = f"{split}_dataset"
    try:
        return Path(str(cfg["paths"][key]))
    except KeyError as exc:
        raise KeyError(f"config paths.{key} is required for split={split}") from exc


def load_prefix_records(path: Path, split: str) -> list[PrefixRecord]:
    records: list[PrefixRecord] = []
    for idx, row in enumerate(read_jsonl(path)):
        n_total = int(row["n_total"])
        records.append(PrefixRecord(
            record_index=idx,
            split=split,
            problem_id=str(row["problem_id"]),
            source_sample_id=str(row["source_sample_id"]),
            prefix_segments=int(row["prefix_segments"]),
            prefix_stage=str(row.get("prefix_stage", "unknown")),
            prefix_token_end=int(row["prefix_token_end"]),
            n_correct=int(row["n_correct"]),
            n_total=n_total,
            source_individual_label=int(row.get("source_individual_label", -1)),
        ))
    return records


def prefix_metadata_row(row: PrefixRecord) -> dict[str, Any]:
    observed = row.n_correct / max(1, row.n_total)
    posterior = (row.n_correct + 0.5) / (row.n_total + 1.0)
    return {
        "record_index": row.record_index,
        "split": row.split,
        "problem_id": row.problem_id,
        "source_sample_id": row.source_sample_id,
        "prefix_segments": row.prefix_segments,
        "prefix_stage": row.prefix_stage,
        "prefix_token_end": row.prefix_token_end,
        "n_correct": row.n_correct,
        "n_total": row.n_total,
        "observed_success_rate": observed,
        "target": posterior,
        "source_individual_label": row.source_individual_label,
    }


def load_source_rows(path: Path, needed_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
    needed = set(str(item) for item in needed_ids)
    found: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not needed:
                break
            row = json.loads(line)
            sid = str(row.get("sample_id", ""))
            if sid in needed:
                found[sid] = row
                needed.remove(sid)
    if needed:
        preview = ", ".join(sorted(needed)[:5])
        raise KeyError(f"{path} is missing {len(needed)} source rows: {preview}")
    return found


def build_extraction_jobs(
    prefix_records: Sequence[PrefixRecord],
    source_rows: Mapping[str, Mapping[str, Any]],
    tokenizer: Any,
) -> list[ExtractionJob]:
    max_by_source: dict[str, int] = defaultdict(int)
    ends_by_source: dict[str, set[int]] = defaultdict(set)
    for row in prefix_records:
        max_by_source[row.source_sample_id] = max(
            max_by_source[row.source_sample_id],
            int(row.prefix_token_end),
        )
        if int(row.prefix_token_end) > 0:
            ends_by_source[row.source_sample_id].add(int(row.prefix_token_end))

    jobs: list[ExtractionJob] = []
    for source_sample_id in sorted(max_by_source):
        source = source_rows[source_sample_id]
        prompt = source.get("metadata", {}).get("rendered_prompt") or source.get("prompt", "")
        encoded = tokenizer(prompt, add_special_tokens=False)
        prompt_ids = [int(item) for item in encoded.input_ids]
        response_ids = [int(item) for item in source.get("token_ids", [])]
        max_end = min(max_by_source[source_sample_id], len(response_ids))
        if max_end <= 0:
            continue
        endpoint_ends = tuple(sorted({
            min(int(end), max_end)
            for end in ends_by_source[source_sample_id]
            if int(end) > 0
        }))
        jobs.append(ExtractionJob(
            source_sample_id=source_sample_id,
            prompt_ids=prompt_ids,
            response_ids=response_ids[:max_end],
            max_prefix_token_end=max_end,
            prefix_token_ends=endpoint_ends,
        ))
    return jobs


def _endpoint_from_store(store: Any, endpoint: int) -> torch.Tensor:
    if isinstance(store, Mapping):
        if endpoint in store:
            return store[endpoint]
        candidates = [int(key) for key in store if int(key) <= int(endpoint)]
        if not candidates:
            raise RuntimeError(f"no cached endpoint <= {endpoint}")
        return store[max(candidates)]
    return store[endpoint]


def gather_prefix_endpoints(
    prefix_records: Sequence[PrefixRecord],
    source_hidden: Mapping[str, Any],
    source_logit_features: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather prefix-end vectors from per-source response hidden tensors.

    Full source tensors are ``[response_tokens, n_layers, hidden_dim]`` for tests
    and compatibility.  The extraction path stores sparse endpoint maps keyed by
    0-indexed response-token endpoint to avoid retaining all token hidden states.
    """

    hidden_rows: list[torch.Tensor] = []
    logit_rows: list[torch.Tensor] = []
    for row in prefix_records:
        hidden = source_hidden[row.source_sample_id]
        logit_features = source_logit_features[row.source_sample_id]
        if isinstance(hidden, Mapping):
            if not hidden:
                raise RuntimeError(f"empty hidden endpoint map for {row.source_sample_id}")
            endpoint = int(row.prefix_token_end) - 1
        else:
            if hidden.shape[0] <= 0:
                raise RuntimeError(f"empty hidden tensor for {row.source_sample_id}")
            endpoint = min(int(row.prefix_token_end), int(hidden.shape[0])) - 1
        if endpoint < 0:
            raise RuntimeError(f"invalid prefix_token_end={row.prefix_token_end} for {row.source_sample_id}")
        hidden_rows.append(_endpoint_from_store(hidden, endpoint))
        logit_rows.append(_endpoint_from_store(logit_features, endpoint))
    return torch.stack(hidden_rows, dim=0), torch.stack(logit_rows, dim=0)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _batched(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), max(1, int(batch_size))):
        yield items[start:start + max(1, int(batch_size))]


def extract_layer_chunk(
    *,
    jobs: Sequence[ExtractionJob],
    model_path: Path,
    layer_ids: Sequence[int],
    max_model_len: int,
    gpu_memory_utilization: float,
    parallel_size: int,
    batch_size: int,
    endpoint_chunk_size: int,
    enforce_eager: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from vllm import LLM, SamplingParams

    hs_tmpdir = tempfile.mkdtemp(prefix="layer_value_lens_hs_", dir="/dev/shm")
    atexit.register(lambda: shutil.rmtree(hs_tmpdir, ignore_errors=True))
    started = time.perf_counter()
    print(
        f"[layer-cache] loading vLLM model={model_path} layers={list(layer_ids)} tp={parallel_size}",
        flush=True,
    )
    llm = LLM(
        model=str(model_path),
        tensor_parallel_size=int(parallel_size),
        max_model_len=int(max_model_len),
        gpu_memory_utilization=float(gpu_memory_utilization),
        enforce_eager=bool(enforce_eager),
        enable_chunked_prefill=False,
        speculative_config={
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,
            "draft_model_config": {
                "hf_config": {
                    "eagle_aux_hidden_state_layer_ids": [int(x) for x in layer_ids],
                }
            },
        },
        kv_transfer_config={
            "kv_connector": "ExampleHiddenStatesConnector",
            "kv_role": "kv_producer",
            "kv_port": _free_port(),
            "kv_connector_extra_config": {
                "shared_storage_path": hs_tmpdir,
            },
        },
    )
    load_seconds = time.perf_counter() - started
    params = [SamplingParams(max_tokens=1, top_p=1.0, top_k=0, temperature=1.0)]
    source_hidden: dict[str, Any] = {}
    source_logit_features: dict[str, Any] = {}
    processed = 0
    for batch in _batched(list(jobs), batch_size):
        outputs = llm.generate(
            [job.prompt_ids + job.response_ids for job in batch],
            params * len(batch),
            use_tqdm=False,
        )
        for job, output in zip(batch, outputs):
            hs_path = output.kv_transfer_params.get("hidden_states_path") if output.kv_transfer_params else None
            if hs_path is None:
                raise RuntimeError(f"vLLM did not return hidden_states_path for {job.source_sample_id}")
            try:
                data = _load_hidden_states_file(hs_path)
                hs = data["hidden_states"]
            finally:
                _cleanup_hidden_states_file(hs_path)

            n_resp = len(job.response_ids)
            if hs.ndim != 3 or hs.shape[1] != len(layer_ids):
                raise RuntimeError(
                    f"expected hidden states [seq,{len(layer_ids)},hidden], got {tuple(hs.shape)}"
                )
            if hs.shape[0] < n_resp + 1:
                raise RuntimeError(
                    f"hidden states too short for {job.source_sample_id}: hs={hs.shape[0]} response={n_resp}"
                )
            response_hs = hs[-(n_resp + 1):-1].cpu()
            token_ids = torch.tensor(job.response_ids, dtype=torch.long)
            endpoint_indices = sorted({
                min(max(1, int(end)), n_resp) - 1
                for end in job.prefix_token_ends
            })
            if not endpoint_indices:
                raise RuntimeError(f"no endpoint indices for {job.source_sample_id}")
            endpoint_hs = response_hs[endpoint_indices]
            endpoint_token_ids = token_ids[endpoint_indices]
            chunks: list[torch.Tensor] = []
            for start in range(0, len(endpoint_indices), int(endpoint_chunk_size)):
                end = min(start + int(endpoint_chunk_size), len(endpoint_indices))
                raw = llm.apply_model(_LayerEndpointLogitFeatureFn(
                    endpoint_hs[start:end],
                    endpoint_token_ids[start:end],
                ))[0]
                chunks.append(raw)
            endpoint_logit = torch.cat(chunks, dim=0).to(torch.float32)
            source_hidden[job.source_sample_id] = {
                int(endpoint): vector.to(torch.float16)
                for endpoint, vector in zip(endpoint_indices, endpoint_hs)
            }
            source_logit_features[job.source_sample_id] = {
                int(endpoint): vector
                for endpoint, vector in zip(endpoint_indices, endpoint_logit)
            }
            processed += 1
            print(
                f"[layer-cache] {processed}/{len(jobs)} {job.source_sample_id} tokens={n_resp}",
                flush=True,
            )

    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    shutil.rmtree(hs_tmpdir, ignore_errors=True)
    return source_hidden, source_logit_features, {
        "layer_ids": [int(x) for x in layer_ids],
        "load_seconds": load_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "n_sources": len(source_hidden),
    }


def layer_chunks(layer_ids: Sequence[int], chunk_size: int) -> list[list[int]]:
    size = max(1, int(chunk_size))
    return [list(layer_ids[start:start + size]) for start in range(0, len(layer_ids), size)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--splits", default="train,val",
                        help="Comma-separated continuation/dataset splits to extract.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer-chunk-size", type=int, default=6)
    parser.add_argument("--layers", default="all",
                        help="Comma-separated 1-indexed layer ids, or 'all'.")
    parser.add_argument("--parallel-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--endpoint-chunk-size", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=10240)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    parser.add_argument("--enforce-eager", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_path = Path(str(cfg["inference"]["model_name_or_path"]))
    hf_cfg = AutoConfig.from_pretrained(str(model_path))
    num_layers = int(hf_cfg.num_hidden_layers)
    hidden_size = int(hf_cfg.hidden_size)
    if args.layers == "all":
        selected_layers = list(range(1, num_layers + 1))
    else:
        selected_layers = [int(item) for item in args.layers.split(",") if item.strip()]
    if not selected_layers:
        raise ValueError("no layers selected")
    invalid = [layer for layer in selected_layers if layer < 1 or layer > num_layers]
    if invalid:
        raise ValueError(f"invalid 1-indexed layer ids for model with {num_layers} layers: {invalid}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    split_names = [item.strip() for item in str(args.splits).split(",") if item.strip()]
    prefix_records_by_split: dict[str, list[PrefixRecord]] = {}
    jobs_by_split: dict[str, list[ExtractionJob]] = {}
    problem_ids_by_split: dict[str, set[str]] = {}
    source_counts: dict[str, int] = {}

    for split in split_names:
        continuations = continuation_path(cfg, split)
        dataset = dataset_path(cfg, split)
        prefix_records = load_prefix_records(continuations, split)
        source_rows = load_source_rows(dataset, {row.source_sample_id for row in prefix_records})
        jobs = build_extraction_jobs(prefix_records, source_rows, tokenizer)
        prefix_records_by_split[split] = prefix_records
        jobs_by_split[split] = jobs
        problem_ids_by_split[split] = {row.problem_id for row in prefix_records}
        source_counts[split] = len(jobs)
        write_jsonl(args.output_dir / f"{split}_metadata.jsonl", [
            prefix_metadata_row(row) for row in prefix_records
        ])
        print(
            f"[layer-cache] split={split} prefixes={len(prefix_records)} sources={len(jobs)} "
            f"problems={len(problem_ids_by_split[split])}",
            flush=True,
        )

    overlap: dict[str, int] = {}
    if "train" in problem_ids_by_split and "val" in problem_ids_by_split:
        overlap["train_val_problem_overlap"] = len(problem_ids_by_split["train"] & problem_ids_by_split["val"])

    chunks = layer_chunks(selected_layers, args.layer_chunk_size)
    chunk_meta: list[dict[str, Any]] = []
    for chunk_index, chunk in enumerate(chunks):
        print(f"[layer-cache] chunk {chunk_index + 1}/{len(chunks)} layers={chunk}", flush=True)
        combined_jobs: list[ExtractionJob] = []
        seen_sources: set[str] = set()
        for split in split_names:
            for job in jobs_by_split[split]:
                if job.source_sample_id in seen_sources:
                    continue
                seen_sources.add(job.source_sample_id)
                combined_jobs.append(job)
        source_hidden, source_logit_features, extraction_meta = extract_layer_chunk(
            jobs=combined_jobs,
            model_path=model_path,
            layer_ids=chunk,
            max_model_len=int(args.max_model_len),
            gpu_memory_utilization=float(args.gpu_memory_utilization),
            parallel_size=int(args.parallel_size),
            batch_size=int(args.batch_size),
            endpoint_chunk_size=int(args.endpoint_chunk_size),
            enforce_eager=bool(args.enforce_eager),
        )
        for split in split_names:
            prefix_hidden, logit_features = gather_prefix_endpoints(
                prefix_records_by_split[split],
                source_hidden,
                source_logit_features,
            )
            first_layer = int(chunk[0])
            last_layer = int(chunk[-1])
            cache_name = f"{split}_layers_{first_layer:04d}_{last_layer:04d}.pt"
            torch.save({
                "split": split,
                "layer_ids": [int(x) for x in chunk],
                "prefix_hidden": prefix_hidden.to(torch.float16),
                "logit_features": logit_features.to(torch.float32),
                "logit_feature_names": ["entropy", "sampled_logprob", "top1_logprob", "top1_margin"],
            }, args.output_dir / cache_name)
            entry = {
                "split": split,
                "cache_file": cache_name,
                "n_prefixes": int(prefix_hidden.shape[0]),
                "hidden_size": int(prefix_hidden.shape[-1]),
                "combined_sources": int(extraction_meta["n_sources"]),
                **extraction_meta,
            }
            chunk_meta.append(entry)
            print(
                f"[layer-cache] wrote {args.output_dir / cache_name} shape={tuple(prefix_hidden.shape)}",
                flush=True,
            )

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": "scripts/extract_layer_value_lens_cache.py",
        "config": str(args.config),
        "model_path": str(model_path),
        "num_hidden_layers": num_layers,
        "hidden_size": hidden_size,
        "layer_ids": selected_layers,
        "layer_indexing": "1-indexed transformer layer ids as passed to vLLM eagle_aux_hidden_state_layer_ids",
        "splits": {
            split: {
                "continuations": str(continuation_path(cfg, split)),
                "dataset": str(dataset_path(cfg, split)),
                "n_prefixes": len(prefix_records_by_split[split]),
                "n_sources": source_counts[split],
                "n_problem_ids": len(problem_ids_by_split[split]),
                "metadata": f"{split}_metadata.jsonl",
            }
            for split in split_names
        },
        "overlap": overlap,
        "chunks": chunk_meta,
        "runtime": {
            "started_at": datetime.fromtimestamp(started).isoformat(timespec="seconds"),
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "elapsed_seconds": time.time() - started,
        },
        "args": vars(args) | {"config": str(args.config), "output_dir": str(args.output_dir)},
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "splits": manifest["splits"],
        "overlap": overlap,
        "n_chunks": len(chunk_meta),
    }, indent=2))


if __name__ == "__main__":
    main()
