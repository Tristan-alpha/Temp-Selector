## ADDED Requirements

### Requirement: mil/DESIGN.md exists
A `mil/DESIGN.md` file SHALL exist explaining the MIL module's design rationale.

#### Scenario: File present in mil/
- **WHEN** checking `mil/DESIGN.md`
- **THEN** the file exists and is non-empty

### Requirement: mil/DESIGN.md covers architectural decisions
The file SHALL explain: MIL problem formulation (positive/negative bag semantics and label flip rationale), model architecture with component roles (InstanceEncoder, PositionEncoding, BiGRU, AttentionAggregator, bag_head, inst_head), the four loss terms and their weights (bag_bce, temp auxiliary, top-k instance, smoothness), why top-k uses k=n_valid//3, the pos_weight sqrt formula, DynamicTempHead integration and the Bug 1 lesson.

#### Scenario: Design rationale sections present
- **WHEN** reading mil/DESIGN.md
- **THEN** it covers problem formulation, architecture, loss function design, and temp head integration
