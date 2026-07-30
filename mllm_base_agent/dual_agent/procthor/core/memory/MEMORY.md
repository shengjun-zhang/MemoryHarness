# ProcTHOR Dual-Agent Memory Index

This is the entry point of the dual-agent skill library for ProcTHOR (mirrors
the AI2-THOR dual-agent memory library at
`mllm_base_agent/dual_agent/ai2thor/core/memory/MEMORY.md`, and in turn
RPent's `resources/<env>/memory/MEMORY.md` pattern). ProcTHOR shares the same
unified action space and dual-agent collaboration protocol as AI2-THOR (same
underlying engine, procedurally generated houses instead of fixed
FloorPlans), so most lessons transfer directly — this index reuses the
AI2-THOR entries where the underlying mechanic is identical, and adds
ProcTHOR-specific notes (multi-room navigation, procedural layout) on top.

**How to use this file:**
1. Scan the one-line hooks below.
2. Read a full entry with `ReadMemory(<file_name>)` — this does NOT consume
   your action-step budget.
3. Entries are reusable patterns, not literal scripts — adapt to your current
   house layout.
4. For the full rulebook (every rule in one place), read
   `ReadMemory(AGENT_RULES.md)` in this same directory.

**When to consult memory (proactively, not just when stuck):**
- Before your first action of an episode.
- The moment an error appears that you don't immediately know how to fix.
- Before outputting `DONE`.
- After 2 consecutive failures of the same/similar action.
- When you can't find a target object in the current room (see
  `feedback_multiroom_exploration.md` before concluding it's missing).

## Index (one-line hooks — read the matching file for full detail)

- `feedback_not_in_view_distance_rule.md` — "X is not in view": distance
  decides ROTATE vs MOVE CLOSER (identical mechanic to AI2-THOR; see the
  AI2-THOR memory library for the full statistics-backed writeup — the
  underlying engine and error format are shared).
- `feedback_blocking_agents.md` — "is blocking" errors (agent-vs-agent and
  static-mesh) and how to unstick navigation without looping the same move.
- `feedback_done_verification.md` — DONE rejected by the evaluator: mandatory
  self-check before ever outputting DONE.
- `feedback_multiroom_exploration.md` — ProcTHOR houses are procedurally
  generated with multiple connected rooms; target objects are frequently in a
  DIFFERENT room than either agent's spawn point. Systematic, communicated
  room-by-room search beats repeated local scanning.
- `feedback_communication_protocol.md` — communication cadence/content that
  correlates with success; what to say and when (shared pattern with
  AI2-THOR's dual-agent protocol).
- `feedback_task_division.md` — splitting multi-subgoal / multi-room tasks
  between two bodies without duplicate work or deadlocks.
- `feedback_step_budget_management.md` — running out of the step budget is
  the most common failure mode in this family of benchmarks; pacing and
  early-stop heuristics.
- `feedback_action_format_errors.md` — action-string / parse errors
  (`ThrowObject(X)`, missing `<ACTION>` tag, bad object casing, missing move
  granularity) that burn a full retry round-trip for nothing.
- `feedback_object_interaction_gotchas.md` — hand-state, interaction-distance,
  and object-naming pitfalls specific to the unified action space.
