## ADDED Requirements

### Requirement: CLI-only parallel_size

The `VLLMFeatureExporter` SHALL receive `parallel_size` exclusively from CLI `--parallel-size` arguments, not from YAML config files. Each script SHALL accept `--parallel-size` as an optional `int` (default `None` for auto-detection).

#### Scenario: parallel_size from CLI

- **WHEN** a script is invoked with `--parallel-size 4`
- **THEN** `parallel_size=4` SHALL be passed to `VLLMFeatureExporter`

#### Scenario: parallel_size default

- **WHEN** a script is invoked without `--parallel-size`
- **THEN** `parallel_size=None` SHALL be passed to `VLLMFeatureExporter` (auto-detect all GPUs)

## MODIFIED Requirements

### Requirement: Constructor parameters

The `VLLMFeatureExporter.__init__` SHALL accept:
- `parallel_size: int | None` (default `None`), provided by CLI `--parallel-size` argument
- `reserve_training_gpu: bool` (default `False`), replacing `engine_preset: str`

The `parallel_size` value SHALL NOT be read from YAML config files.

#### Scenario: Default construction

- **WHEN** `VLLMFeatureExporter(model_name_or_path=...)` is called with no GPU-related parameters
- **THEN** all available GPUs SHALL be used for vLLM with no reservation
