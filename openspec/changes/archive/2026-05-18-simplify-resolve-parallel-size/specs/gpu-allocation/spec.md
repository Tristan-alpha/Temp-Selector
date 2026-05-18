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

### Requirement: Constructor parameters

The `VLLMFeatureExporter.__init__` SHALL accept:
- `parallel_size: int | None` (default `None`), replacing `int | str | None`
- `reserve_training_gpu: bool` (default `False`), replacing `engine_preset: str`

#### Scenario: Default construction

- **WHEN** `VLLMFeatureExporter(model_name_or_path=...)` is called with no GPU-related parameters
- **THEN** all available GPUs SHALL be used for vLLM with no reservation
