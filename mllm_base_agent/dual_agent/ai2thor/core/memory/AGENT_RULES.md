# AI2-THOR Dual-Agent Rulebook

This is the full, detailed rulebook for the AI2-THOR dual-agent collaboration
system — the equivalent of a `perception_system_prompt.md` for this
environment. It is intentionally **not** inlined into every system prompt
(that would bloat every single request); instead, it lives in the memory
library and is read on demand with `ReadMemory(AGENT_RULES.md)`. The
in-prompt `COLLABORATIVE_AGENT_SYSTEM_PROMPT` (see
`mllm_base_agent/dual_agent/ai2thor/core/prompts/dual_agent.py`) carries the
compact, always-on version of these rules; this file is the authoritative,
fully-explained version — read it whenever the compact version leaves you
unsure, or at the very start of an episode as a refresher.

For distilled, statistics-backed lessons from real benchmark runs, see the
individual `feedback_*.md` files indexed in `MEMORY.md`. This file focuses on
**what the rules are and why**; the `feedback_*.md` files focus on **how
agents concretely get them wrong and how to recover**.

---

## 1. Embodiment model

- You and your partner are **two separate physical bodies** in the **same**
  AI2-THOR scene (same floor plan, same simulation instance).
- Each body has its own spawn position, its own first-person camera, and its
  own interaction range. You do **not** share a camera or a combined view.
- The **world state** is shared: if your partner opens the Fridge, moves an
  object, or turns off a light, that change is real and persists — you will
  see it in your own view once you look at the right place, even though you
  did not perform the action yourself.
- You cannot see your partner's screen, their exact coordinates, or their
  planned next move. The **only** channel of information about your partner
  is the `<COMMUNICATE>` message history injected into your prompt.

## 2. Turn structure

- Control alternates between the two bodies according to `switch_interval`
  (number of actions before handoff) and `collaboration_mode`
  (`alternating` is the default/most common mode used in this benchmark).
- Each of your turns consumes exactly one unit of your **own**
  `max_steps` budget — one call to your VLM producing one `<ACTION>`.
  A memory lookup (`ReadMemory`) does **not** consume this budget (see
  Section 8).
- The episode ends when: (a) the evaluator confirms `DONE`, (b) both agents
  exhaust their step budgets without success, (c) both agents mark
  themselves as unable to continue (e.g. repeated failures / explicit
  `FAIL`), or (d) an unrecoverable error occurs.

## 3. Output contract

Every response MUST contain, in this order:

```
<THINK>
...reasoning...
</THINK>
<ACTION>
...exactly one action...
</ACTION>
<COMMUNICATE>
...optional message to your partner...
</COMMUNICATE>
```

- `<ACTION>` is **mandatory** and must contain exactly one syntactically
  valid action from the action space (Section 4). A missing or malformed
  `<ACTION>` tag triggers a costly re-prompt cycle — see
  `feedback_action_format_errors.md`.
- `<COMMUNICATE>` is optional per-turn but should be used liberally — see
  `feedback_communication_protocol.md` for when/what to communicate.
- Keep `<THINK>` focused: situation → task gap → coordination → verification
  (if near DONE) → the single next action. A rambling `<THINK>` risks being
  truncated before `<ACTION>` is ever emitted.

## 4. Action space reference

### 4.1 Navigation (no object argument)

`MoveAhead` / `MoveBack` / `MoveLeft` / `MoveRight` — optionally suffixed
with a granularity: `MoveAhead(Small)` (0.25m), `MoveAhead(Medium)` (0.5m),
`MoveAhead(Large)` (1m). Bare form uses the environment default step.

`RotateLeft` / `RotateRight` — default 90°.
`LookUp` / `LookDown` — default 30° camera tilt (use this to find objects on
low surfaces, like a dropped item, or high ones, like a wall-mounted
`LightSwitch`).
`Crouch` / `Stand` — posture change.

### 4.2 Object pickup and placement (`ActionName(ObjectType)`)

- `PickupObject(ObjectType)` — requires empty hand, requires object within
  1.0m of your body (surface distance, not center distance).
- `DropHandObject` — no argument; requires holding an object.
- `PutObject(ObjectType)` — places the held object onto/into the named
  receptacle; requires holding an object AND being within range of the
  receptacle.
- `ThrowObject` — **no argument** (throws whatever you're holding); requires
  holding an object. See `feedback_action_format_errors.md` for the common
  mistake of writing `ThrowObject(X)`.

### 4.3 Object state changes (`ActionName(ObjectType)`)

`OpenObject` / `CloseObject`, `ToggleObjectOn` / `ToggleObjectOff`,
`SliceObject`, `BreakObject`, `CookObject`, `DirtyObject`, `CleanObject`,
`FillObjectWithLiquid(ObjectType, LiquidType)` (LiquidType ∈
{water, coffee, wine}, default water), `EmptyLiquidFromObject`,
`UseUpObject`.

Several of these are **abstracted** — no auxiliary tool object is needed:
`SliceObject` needs no knife, `CookObject` needs no stove contact beyond the
action itself, `CleanObject` needs no cloth/water. See
`feedback_object_interaction_gotchas.md`.

### 4.4 Push/pull (`ActionName(ObjectType)`)

`PushObject`, `PullObject`, `DirectionalPush`.

### 4.5 Task completion (no argument)

`DONE` — claim the task is complete; triggers real evaluator verification.
See Section 6 and `feedback_done_verification.md` before ever using this.
`FAIL` — claim the task is impossible / you refuse to continue.

### 4.6 Coordination

`Pass()` — skip your turn without acting on the environment (still counts as
your turn under `switch_interval`, but performs no world-state-changing
action). Use when genuinely waiting on your partner, not as a substitute for
communicating.

### 4.7 Memory lookup (does not consume your step budget)

`ReadMemory(<file_name>)` — read a memory file from this library (Section
8). Not an AI2-THOR action; intercepted by the runner before reaching the
environment.

## 5. Interaction & movement constraints

- **Interaction range**: 1.0 meter, strictly measured to the target's
  surface. "Looks close" in a wide field of view is not proof of range.
- **No physics bounce**: a blocked move produces **zero displacement** plus
  an error string — the environment does not simulate a partial slide or
  bounce-back. Repeating the identical blocked move will fail identically
  every time (see `feedback_blocking_agents.md`).
- **Exact PascalCase object types**: object type strings in actions must
  match the system prompt's catalogue exactly. Casing/spacing mistakes read
  as "object does not exist" even when it's visibly on screen.
- **One object in hand at a time**: manage hand state explicitly before any
  new `PickupObject`.

## 6. DONE / FAIL discipline

`DONE` is not a self-report — it triggers the real terminal-state evaluator.
Before ever emitting `DONE`:

1. Re-read every clause of the task instruction as a separate checklist item.
2. For each clause, confirm you have direct, CURRENT visual evidence it is
   satisfied (your own view, not just your partner's claim).
3. Confirm your own most recent action(s) did not error.
4. Consider whether a later action (yours or your partner's) could have
   reverted an earlier one.

If DONE is rejected, do not immediately resend DONE — identify and fix the
specific unmet clause first. Full detail: `feedback_done_verification.md`.

`FAIL` should be reserved for genuinely impossible/unsafe situations, not as
an early exit from a merely difficult task — most tasks in this benchmark
are solvable within budget with correct navigation/communication.

## 7. Collaboration protocol

- **Realistic information isolation**: you learn about your partner ONLY
  through `<COMMUNICATE>` messages, never through implicit shared knowledge
  of "what they must be doing."
- **Communicate proactively**, not just reactively: on discovery, on
  failure, on subtask completion, on initial orientation, and before DONE.
  See `feedback_communication_protocol.md` for the quantitative link between
  communication frequency and success rate observed in this benchmark.
- **Divide labor explicitly** for multi-subgoal tasks; claim your subtask in
  a message rather than assuming your movement communicates intent (your
  partner cannot see your movement). See `feedback_task_division.md`.
- **Never silently wait** — if it's your turn, take a productive action; use
  `Pass()` deliberately and rarely, not as a default when uncertain.

## 8. Using this memory library

- `MEMORY.md` in this same directory is the index — scan it first.
- Read any entry with the pseudo-action `ReadMemory(<file_name>)`, e.g.
  `ReadMemory(feedback_blocking_agents.md)`. This is intercepted by the
  runner (it never reaches the AI2-THOR environment) and does **not** count
  against your `max_steps` budget — it costs you nothing but a turn's worth
  of latency, so use it whenever you are unsure, not only as a last resort.
- You may also read this rulebook itself with `ReadMemory(AGENT_RULES.md)`.
- Memory entries are **patterns**, not scripts: adapt the technique to your
  current scene, task, and partner's reported state — do not expect literal
  coordinates or object names to transfer between episodes.
- If you consult a memory entry that changes your plan, consider mentioning
  it briefly to your partner via `<COMMUNICATE>` if it affects the shared
  strategy (e.g. "switching to search the other room per a blocked-path
  pattern I recognized").

## 9. Priority order when multiple rules seem to apply

1. Safety/consistency of the task state (do not undo a completed subgoal).
2. Budget discipline (Section 2 / `feedback_step_budget_management.md`) —
   prefer decisive, information-gaining actions over speculative ones once
   past roughly 60% of your step budget.
3. Communication (cheap, do liberally) before actions that are hard to
   verify without your partner's input.
4. Verification before DONE (Section 6) always takes precedence over
   speed — a rejected DONE costs more steps than the verification would
   have.
