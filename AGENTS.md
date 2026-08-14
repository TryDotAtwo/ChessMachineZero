# ChessMachineZero Project Memory

## Standing Rules

- Read project rules and project memory before implementation work.
- Keep architecture source material in `docs/`.
- Keep prompt history and change history in `docs/`.
- Keep test outputs and test notes in `test_results/`.
- Prefer agent-centered code: explicit state, deterministic behavior, narrow modules, low hidden coupling.
- Keep inference architecture trace-based: VM trace records enumerate legal moves; no flat external policy head.
- Do not add external tree-search wrappers.
- Do not add human-game training data.
- Do not add engine labels.
- Do not add tablebase labels.
- Do not add handcrafted chess evaluation.
- Restrict `python-chess` usage to the development/test oracle wrapper.
- Do not add runtime substitutions that silently substitute weaker behavior for required behavior.
- Do not add smoke tests as acceptance evidence; tests must verify exact behavior or invariants with real assertions.
- User-visible assistant responses must be in Russian. Internal process notes, code, identifiers, and test names may remain in English.
- Percepta rule-weight target means parametric chess rules embedded as reusable frozen weights/circuits, not memorized finite board positions or finite prompt continuations.

## Current Milestone

- milestone_id=differentiable-recurrent-chess-ring-v10
- scope=Build VM3 as one immutable GPU-resident rule graph that consumes a canonical chess state and append-only move trajectory, emits the complete orthodox move LEGAL_SET and explicitly declared terminal set, routes a swappable attention-only policy through deterministic exact-hard/straight-through eligibility-weighted softmax selection, returns the same state ABI, appends exactly one MOVE token per committed ply, remains absorbing after internally computed terminal state, and preserves one autograd graph from terminal loss to first-ply policy logits without host chess semantics; exact general FIDE dead-position recognition remains a separately gated research milestone and may not be implied by material-subset tests.
- source_spec=docs/superpowers/specs/2026-08-14-differentiable-recurrent-chess-machine-design.md
- execution_plan=docs/superpowers/plans/2026-08-14-differentiable-recurrent-chess-machine.md
- acceptance=CTest plus exact pytest oracle/purity/evidence gates, full-game gradient reachability, deterministic GPU-residency artifacts, and honest Pages trace artifacts

## Update Policy

- Update `docs/project_memory.md` when durable implementation facts change.
- Update `docs/change_history.md` after significant code/test changes.
- Update `docs/prompt_history.md` after user task prompts that define scope.
- Store command output summaries or logs under `test_results/`.
