## ADDED Requirements

### Requirement: GPU count from torch only

The system SHALL use `torch.cuda.device_count()` as the sole source of truth for available GPU count when determining vLLM tensor-parallel size. It SHALL NOT parse `CUDA_VISIBLE_DEVICES` manually.

#### Scenario: Auto-detect all GPUs

- **WHEN** `parallel_size` is `None` (or omitted) and `torch.cuda.device_count()` returns `N > 0`
- **THEN** tensor-parallel size SHALL be `N`

#### Scenario: Explicit parallel_size

- **WHEN** `parallel_size` is an integer `K > 0`
- **THEN** tensor-parallel size SHALL be `K` (regardless of `torch.cuda.device_count()`)

### Requirement: No-GPU error

The system SHALL raise `RuntimeError` if `torch.cuda.device_count()` returns 0.

#### Scenario: Zero GPUs detected

- **WHEN** `torch.cuda.device_count()` returns 0
- **THEN** `_resolve_parallel_size` SHALL raise `RuntimeError` with a message indicating no GPUs are available

### Requirement: Training GPU reservation

The system SHALL support reserving one GPU for training via a `reserve_training_gpu: bool` parameter (default `False`). When `True`, tensor-parallel size SHALL be reduced by 1.

#### Scenario: Reservation with sufficient GPUs

- **WHEN** `reserve_training_gpu=True` and `torch.cuda.device_count()` returns 2 or more
- **THEN** tensor-parallel size SHALL be `N - 1`

#### Scenario: Reservation with insufficient GPUs

- **WHEN** `reserve_training_gpu=True` and `torch.cuda.device_count()` returns 1
- **THEN** `_resolve_parallel_size` SHALL raise `RuntimeError` indicating no GPUs remain after reservation

### Requirement: CLI-only parallel_size

The `VLLMFeatureExporter` SHALL receive `parallel_size` exclusively from CLI `--parallel-size` arguments, not from YAML config files. Each script SHALL accept `--parallel-size` as an optional `int` (default `None` for auto-detection).

#### Scenario: parallel_size from CLI

- **WHEN** a script is invoked with `--parallel-size 4`
- **THEN** `parallel_size=4` SHALL be passed to `VLLMFeatureExporter`

#### Scenario: parallel_size default

- **WHEN** a script is invoked without `--parallel-size`
- **THEN** `parallel_size=None` SHALL be passed to `VLLMFeatureExporter` (auto-detect all GPUs)

### Requirement: Constructor parameters

The `VLLMFeatureExporter.__init__` SHALL accept:
- `parallel_size: int | None` (default `None`), provided by CLI `--parallel-size` argument
- `reserve_training_gpu: bool` (default `False`), replacing `engine_preset: str`

The `parallel_size` value SHALL NOT be read from YAML config files.

#### Scenario: Default construction

- **WHEN** `VLLMFeatureExporter(model_name_or_path=...)` is called with no GPU-related parameters
- **THEN** all available GPUs SHALL be used for vLLM with no reservation

### Requirement: Config directory structure

Config files SHALL be organized into `configs/dataset/` and `configs/training/`. Dataset configs SHALL include a `split:` section with `val_ratio` and `test_ratio`. `build_dataset.py` SHALL read split parameters from config, with CLI args as overrides. Split seed SHALL reuse the global `seed` key.

#### Scenario: Split params from config

- **WHEN** `build_dataset.py` is invoked with `--config configs/dataset/full.yaml`
- **THEN** `val_ratio` and `test_ratio` SHALL be read from config `split:` section; seed SHALL use global `seed`
