# AI2-THOR Dual-Agent Memory Index

This is the entry point of the dual-agent skill library (mirrors RPent's
`resources/<env>/memory/MEMORY.md` pattern). It is a reviewed, mostly-static
knowledge base of operating wisdom, magic numbers, gotchas, and reusable
collaboration patterns distilled from real benchmark runs (232 dual-agent
AI2-THOR episodes analyzed, 82 successes / 150 failures).

**How to use this file:**
1. Scan the one-line hooks below to find entries relevant to your current
   situation (a repeated error, a stuck negotiation, an upcoming DONE, ...).
2. Read the full entry with `ReadMemory(<file_name>)` (no quotes, e.g.
   `ReadMemory(feedback_blocking_agents.md)`). Reading memory does NOT consume
   your action-step budget — it is a free lookup, use it whenever unsure.
3. Entries are written as reusable patterns, not literal scripts: adapt the
   technique to your current scene/task, do not expect exact coordinates.
4. For the full rulebook (every rule in one place, more detail than the
   system prompt keeps inline), read `AGENT_RULES.md` in this same directory.

**When you should consult memory (do this proactively, not only when stuck):**
- Before your first action of an episode (cheap insurance, costs nothing).
- The moment you see an error you don't immediately know how to fix.
- Before outputting `DONE` (re-check `feedback_done_verification.md`).
- After 2 consecutive failures of the same/similar action.

## Index (one-line hooks — read the matching file for full detail)

- `feedback_not_in_view_distance_rule.md` — "X is not in view" error: distance
  decides ROTATE vs MOVE CLOSER; picking the wrong one wastes the most steps
  of any single mistake in the whole benchmark (728 occurrences observed).
- `feedback_blocking_agents.md` — "Agent N is blocking" / static-mesh blocking:
  how to unstick navigation without looping the same failed move (413
  occurrences observed; the #2 most common error).
- `feedback_done_verification.md` — DONE rejected by the evaluator: why it
  happens, and the mandatory self-check before ever outputting DONE (101
  DONE calls observed, only ~66% accepted on first try).
- `feedback_action_format_errors.md` — action-string / parse errors
  (`ThrowObject(X)`, missing `<ACTION>` tag, bad object casing) that burn a
  full retry round-trip for nothing.
- `feedback_communication_protocol.md` — communication cadence and content
  that correlates with success (successful episodes communicate ~30% more
  per step than failed ones); what to say and when.
- `feedback_task_division.md` — how to split multi-subgoal tasks between two
  bodies without duplicate work or deadlocks (who-does-what negotiation
  patterns, avoiding "I'll wait for you" loops).
- `feedback_step_budget_management.md` — the #1 failure reason is running out
  of the step budget (78/150 failures, i.e. >50% of all failures) — pacing
  and early-stop heuristics to avoid this.
- `feedback_object_interaction_gotchas.md` — hand-state, interaction-distance,
  and object-naming pitfalls specific to the AI2-THOR unified action space.
