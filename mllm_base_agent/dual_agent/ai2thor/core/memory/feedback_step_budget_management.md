# Step-budget exhaustion is the #1 cause of failure — pace accordingly

**Observed:** "Reached max steps before task completion" is the fail_reason
in 78 of 150 failed episodes (52% of ALL failures, by far the largest single
category — more than parse errors, API errors, and premature-DONE combined).
Each agent gets a fixed per-agent step budget (`per_agent_max_steps`,
commonly derived as `10 + golden_action_steps`); the global cap is
`2 * per_agent_max_steps`. Once exhausted, the episode ends regardless of
progress.

## Root causes observed in this failure bucket

1. **Unproductive exploration loops** — repeatedly rotating/moving without a
   clear search strategy, re-visiting the same area because neither agent
   is tracking (via communication) what has already been searched.
2. **Recovery churn** — spending many steps retrying a blocked/failed action
   with only minor variations instead of decisively changing strategy after
   1-2 failed attempts (see `feedback_blocking_agents.md` and
   `feedback_not_in_view_distance_rule.md` for the two largest contributors).
3. **Communication-only turns without progress** — using `Pass()` or a
   `RotateRight`-as-filler action turn after turn without new information
   exchanged, effectively burning the OTHER agent's turn allocation in
   alternating mode without task progress.
4. **Late task-division** — spending the first several steps of the episode
   without any coordination, then discovering duplicate work only when both
   agents converge on the same object (see `feedback_task_division.md`).

## Pacing heuristics

- **Budget awareness**: you know your `max_steps` from the very first system
  prompt / task context. As a rule of thumb, if you are past ~60% of your
  step budget and still in a pure-exploration phase with no task-relevant
  object located yet, treat this as urgent — prioritize systematic coverage
  (pick an unexplored direction, communicate it, do not re-scan an area
  already reported by either agent) over further speculative wandering.
- **Fail fast on recovery loops**: if the SAME category of error (e.g.
  "not in view", "is blocking") has happened 2+ times in a row for the same
  target, change strategy category entirely (e.g. stop trying to approach
  directly, ask your partner if they can see/reach it instead) rather than
  trying minor variations of the same fix a third time.
- **Do not stall waiting for your partner** — if `switch_interval` gives you
  the turn, always take a productive action; a wasted turn under alternating
  mode is a real step gone from your budget, not a free pass.
- **Prefer decisive committed moves over cautious micro-steps** once you know
  roughly where you're going (e.g. use `MoveAhead(Large)`/default granularity
  in open areas; reserve `Small` steps for close-range fine positioning near
  a target or when repeatedly overshooting).
- **Track what's already been searched via communication** rather than
  relying on memory alone — a short "kitchen fully searched, no Egg there"
  message prevents your partner (or your own later self) from re-covering
  the same ground.
