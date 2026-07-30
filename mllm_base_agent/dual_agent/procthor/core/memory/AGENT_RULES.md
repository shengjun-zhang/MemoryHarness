# ProcTHOR Dual-Agent Rulebook

This is the full rulebook for the ProcTHOR dual-agent collaboration system.
It shares almost all of its content with the AI2-THOR dual-agent rulebook at
`mllm_base_agent/dual_agent/ai2thor/core/memory/AGENT_RULES.md` (same engine,
same unified action space, same collaboration protocol) — read that file for
the complete section-by-section reference (embodiment model, turn structure,
output contract, action space, interaction constraints, DONE/FAIL
discipline, collaboration protocol, and memory-library usage). This file
covers only what's different in ProcTHOR.

## What's different in ProcTHOR

1. **Procedurally generated, multi-room houses.** AI2-THOR's dual-agent
   benchmark tasks mostly run in a single fixed `FloorPlan*` room; ProcTHOR
   houses are generated with several connected rooms (kitchen, bedroom,
   living room, bathroom, ...). A target object is often in a room neither
   agent starts in. See `feedback_multiroom_exploration.md` for the search
   strategy this implies — it is the single biggest behavioral difference
   from the AI2-THOR playbook.

2. **Mandatory move granularity.** The ProcTHOR prompt requires every
   movement action to carry an explicit granularity suffix:
   `MoveAhead(Small)` / `MoveAhead(Medium)` / `MoveAhead(Large)` (0.25m /
   0.5m / 1m). A bare `MoveAhead` or a numeric argument like `MoveAhead(0.5)`
   is a parse error here, whereas some AI2-THOR configs tolerate the bare
   form. Always include the granularity keyword in ProcTHOR.

3. **No `Pass()` turn-skip action.** Unlike the AI2-THOR dual-agent loop,
   the ProcTHOR loop does not special-case a `Pass()` no-op — every action
   you emit should be a real action from the catalogue (navigation,
   interaction, `DONE`/`FAIL`, or `ReadMemory(<file_name>)`). If you are
   waiting on your partner, prefer a low-risk exploratory action (e.g.
   `RotateLeft` to scan) over inventing an unsupported no-op.

4. **Cookable set is narrower.** In this benchmark's ProcTHOR configuration,
   `CookObject` applies to `{Egg, Potato, Tomato}` (no sliced-variant
   cooking) — check the system prompt's catalogue rather than assuming the
   AI2-THOR set (which includes `PotatoSliced`, `BreadSliced`,
   `EggCracked`) transfers unchanged.

## Everything else is shared

Embodiment model, turn structure and step budget, the `<THINK>/<ACTION>/
<COMMUNICATE>` output contract, interaction distance (1.0m to surface),
hand-state rules, abstract actions needing no tools, exact PascalCase object
naming, DONE verification discipline, and the `ReadMemory(<file_name>)`
memory-lookup pseudo-action (free, does not consume step budget) — all work
identically to the AI2-THOR dual-agent system. When in doubt, the AI2-THOR
`AGENT_RULES.md` is the authoritative long-form reference; this file only
overrides it for the four points above.
