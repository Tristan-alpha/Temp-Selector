from inference.vllm_runner import VLLMFeatureExporter


class _ChunkingExporter(VLLMFeatureExporter):
    def __init__(self):
        super().__init__(
            model_name_or_path="unused",
            max_batch_size=2,
        )
        self.calls = []

    def _generate_with_features_once(
        self, prompts, temperatures, segment_size, top_k=4096,
        return_logprobs=False, return_hidden=False, device=None, seeds=None,
    ):
        self.calls.append((list(prompts), list(temperatures), None if seeds is None else list(seeds)))
        return [
            {
                "token_ids": [seed if seed is not None else idx],
                "tokens": ["x"],
                "text": prompt,
                "finish_reason": "length",
                "logprobs": None,
                "hidden_states": None,
            }
            for idx, (prompt, seed) in enumerate(zip(
                prompts,
                seeds if seeds is not None else [None] * len(prompts),
            ))
        ]


def test_generate_with_features_micro_batches_preserve_order_and_seeds():
    exporter = _ChunkingExporter()
    result = exporter.generate_with_features(
        prompts=["a", "b", "c", "d", "e"],
        temperatures=[0.1, 0.2, 0.3, 0.4, 0.5],
        segment_size=64,
        seeds=[10, 11, 12, 13, 14],
    )
    assert exporter.calls == [
        (["a", "b"], [0.1, 0.2], [10, 11]),
        (["c", "d"], [0.3, 0.4], [12, 13]),
        (["e"], [0.5], [14]),
    ]
    assert [item["text"] for item in result] == ["a", "b", "c", "d", "e"]
    assert [item["token_ids"][0] for item in result] == [10, 11, 12, 13, 14]
