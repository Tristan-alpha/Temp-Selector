## Why

After the recent radical refactoring, the codebase is well-organized but lacks internal documentation: most functions and classes have no docstrings, there is no per-module design rationale, and no project-level navigation file for Claude Code. Adding targeted documentation now prevents future contributors (and future-self) from rediscovering design decisions through code archaeology.

## What Changes

- Add **docstrings** to 8 critical functions/classes across the codebase covering MIL loss mechanics, model forward flow, PPO batch construction, GAE computation, segment pooling, data structures, and feature vectorization
- Add **CLAUDE.md** at project root for Claude Code context loading — project overview, directory structure, key data flow, coding conventions, common pitfalls, and common tasks
- Add **3 module DESIGN.md** files (`mil/DESIGN.md`, `ppo/DESIGN.md`, `features/DESIGN.md`) explaining the "why" behind architectural choices specific to each module

## Capabilities

### New Capabilities

- `claude-md`: Project-level navigation file at root providing structural overview, data flow, conventions, pitfalls, and task recipes for Claude Code
- `mil-design-doc`: Module-level design document explaining MIL problem formulation, model architecture rationale, loss function design, top-k instance loss motivation, and temp head integration
- `ppo-design-doc`: Module-level design document explaining online vs offline PPO, per-segment generation loop with vLLM APC, GAE mechanics, reward shaping strategy, and overfitting diagnostic
- `features-design-doc`: Module-level design document explaining data pipeline, segmentation mode tradeoffs, vectorizer feature merging strategy, and majority voting computation
- `critical-docstrings`: Targeted docstrings on 8 key functions/classes explaining the "why" behind non-obvious design decisions

### Modified Capabilities

<!-- None -- no existing specs -->

## Impact

- **Code**: 8 functions/classes modified (docstrings only, no behavior change)
- **New files**: `CLAUDE.md`, `mil/DESIGN.md`, `ppo/DESIGN.md`, `features/DESIGN.md`
- **Tests**: No impact (documentation-only change)
- **Dependencies**: None
