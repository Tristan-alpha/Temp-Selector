## ADDED Requirements

### Requirement: features/DESIGN.md exists
A `features/DESIGN.md` file SHALL exist explaining the features module's design rationale.

#### Scenario: File present in features/
- **WHEN** checking `features/DESIGN.md`
- **THEN** the file exists and is non-empty

### Requirement: features/DESIGN.md covers data pipeline and design tradeoffs
The file SHALL explain: the full data pipeline from JSONL to BagSample to instance tensor, the three segmentation modes (fixed_window, step, punctuation) and why step is the default, the vectorizer's feature merging strategy (logprob + entropy + top-k logits into fixed-dim vectors), majority voting computation and its role as a unified signal across all stages, and the data format contract between Stage 1, 2, and 3.

#### Scenario: Design rationale sections present
- **WHEN** reading features/DESIGN.md
- **THEN** it covers data pipeline, segmentation tradeoffs, vectorizer strategy, and inter-stage data contract
