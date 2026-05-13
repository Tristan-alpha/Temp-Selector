## Context

After the radical directory/config refactoring, the codebase is well-structured but lacks internal documentation. There are no per-module design docs, no project-level navigation for Claude Code, and most functions lack docstrings explaining non-obvious design decisions. This makes onboarding and future maintenance unnecessarily difficult.

## Goals / Non-Goals

**Goals:**
- Add a `CLAUDE.md` for automatic context loading in Claude Code sessions
- Add 3 module-level `DESIGN.md` files (mil, ppo, features) explaining architectural rationale
- Add targeted docstrings/comments to 8 critical functions/classes
- Zero behavior changes — documentation only

**Non-Goals:**
- Full API documentation (Sphinx, autodoc, etc.)
- Tutorials or getting-started guides (already covered by PIPELINE.md)
- Documenting every function (only the 8 most non-obvious)
- Adding type stubs or external doc tools

## Decisions

### D1: Docstring style — inline comments for logic, docstrings for APIs

| Type | Example | Rationale |
|---|---|---|
| Complex logic block | Top-k instance loss loop | Comment block above the loop explains the "why" directly |
| Public API | `MILModel.forward()`, `compute_gae()`, `segment_pooling()`, `token_to_vec()` | Docstring so IDE hover shows it |
| Data structure | `BagSample` | Comment on the `label` field explaining semantic flip |

### D2: DESIGN.md scope — explain "why", not "what"

Each DESIGN.md focuses on architectural rationale, not code walkthrough:
- **Tradeoffs**: why step segmentation over fixed_window, why k=n_valid//3, why online PPO
- **Data flow**: how tensors move through the system
- **Design lessons**: Bug 1 (dynamic head mismatch) as a cautionary tale in mil/DESIGN.md

### D3: CLAUDE.md — compact navigation, not a second README

CLAUDE.md complements PIPELINE.md and README.md. It adds:
- Per-file one-liners (PIPELINE.md has a tree but no descriptions per file)
- Coding conventions (not in PIPELINE.md)
- Common pitfalls (label flip, segment_obs=None — scattered across conversation context)
- Common task recipes (how to run ablations — partly in PIPELINE.md but scattered)

It does NOT duplicate the full pipeline documentation.

## Risks / Trade-offs

- **[Doc rot]** DESIGN.md content can drift from implementation → **Mitigation**: Keep DESIGN.md focused on architectural rationale (which changes rarely), not implementation details (which change often)
- **[No automated verification]** Doc correctness isn't testable → **Mitigation**: Review DESIGN.md against current code at time of writing; include docstring presence tests

## Migration Plan

1. Write `CLAUDE.md` at project root
2. Write `mil/DESIGN.md`, `ppo/DESIGN.md`, `features/DESIGN.md`
3. Add docstrings/comments to the 8 target functions/classes
4. Verify all files exist and are non-empty
