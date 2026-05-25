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
- Do not add fallbacks that silently substitute weaker behavior for required behavior.
- Do not add smoke tests as acceptance evidence; tests must verify exact behavior or invariants with real assertions.
- User-visible assistant responses must be in Russian. Internal process notes, code, identifiers, and test names may remain in English.
- Percepta rule-weight target means parametric chess rules embedded as reusable frozen weights/circuits, not memorized finite board positions or finite prompt continuations.

## Current Milestone

- milestone_id=percepta-two-transformer-dashboard-v9.1
- scope=Two independent PerceptaFrozenAttentionRuleComputer instances play white/black self-play; each committed move stores side-specific transformer id, emitted trace tokens, and trace-legal verification; dashboard auto-starts sequential transformer self-play on page load, displays the active transformer, last emitter, verification bit, and token arrays; runtime verification uses trace LEGAL_SET membership only and does not call the python-chess oracle
- source_spec=docs/chess_machine_zero_percepta_architecture.md
- acceptance=`pytest`

## Update Policy

- Update `docs/project_memory.md` when durable implementation facts change.
- Update `docs/change_history.md` after significant code/test changes.
- Update `docs/prompt_history.md` after user task prompts that define scope.
- Store command output summaries or logs under `test_results/`.
