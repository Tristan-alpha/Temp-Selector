## ADDED Requirements

### Requirement: CLAUDE.md exists at project root
A `CLAUDE.md` file SHALL exist at the project root providing structural overview for Claude Code sessions.

#### Scenario: File is present
- **WHEN** checking the project root
- **THEN** `CLAUDE.md` exists and is non-empty

### Requirement: CLAUDE.md covers essential navigation topics
The file SHALL include sections for: project overview (one sentence), directory structure with per-file one-line descriptions, key data flow from BagSample through MIL to PPO, coding conventions (import style, type annotations, test requirements), common pitfalls (label semantics, segment_obs=None, even vote ties), and common tasks (running ablation experiments).

#### Scenario: All sections present
- **WHEN** reading CLAUDE.md
- **THEN** it contains sections on project overview, directory structure, data flow, conventions, pitfalls, and common tasks
