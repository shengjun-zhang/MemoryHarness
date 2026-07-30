# Splitting multi-subgoal tasks between two bodies

Dual-agent tasks in this benchmark are typically decomposable into
independent or loosely-coupled subgoals (e.g. "open the fridge AND turn off
the desk lamp", "put two eggs in the pan", "clean the Mug and the Plate").
Because each body has its OWN position/view and the ONLY channel between
you is `<COMMUNICATE>`, poor division of labor shows up as one of these
failure patterns:

## Pattern 1 — Duplicate work

Both agents independently decide to handle the same subgoal because neither
announced a claim. This wastes half the effective step budget (both bodies
converge on the same object, often colliding — see
`feedback_blocking_agents.md`). Fix: as soon as you decide which part of the
task you will handle, say so explicitly, and wait for/read your partner's
claim before finalizing yours to avoid a symmetric collision.

## Pattern 2 — Deadlocked waiting

One agent waits ("I'll wait for you to finish the Fridge") while the other
is ALSO waiting on some unstated assumption, and neither body moves for
several steps. Fix: never wait indefinitely on an unconfirmed assumption —
if you haven't heard a concrete claim/progress update from your partner in
the last couple of exchanges, proceed on the sub-goal you can make progress
on yourself, or explicitly ask a direct question that forces a decision.

## Pattern 3 — Sequential subgoals treated as parallel (or vice versa)

Some tasks have a genuine ordering dependency (e.g. "pick up the Egg, THEN
put it in the Pan" cannot be parallelized across two different objects if
there's only one Egg) — in that case, one agent should fetch/hand off while
the other prepares the destination (e.g. opens the Microwave) rather than
both reaching for the same single object. Conversely, tasks with two
genuinely independent objects (fridge + lamp) should usually be split 1:1
rather than one agent doing both sequentially while the other stays idle.

## Practical negotiation protocol

1. On the very first 1-2 steps, both agents should describe what they see
   and propose a split ("I see the kitchen area, I'll handle the Fridge
   subtask" / "I'm near what looks like a living room, I'll handle the
   lamp").
2. If your partner's first message already claims a subtask before you've
   proposed anything, take the complementary one rather than re-negotiating
   from scratch.
3. Re-communicate progress at natural checkpoints (subtask started /
   subtask completed / subtask failed-retrying) so the other agent's mental
   model of "what's left" stays in sync without needing to ask every turn.
4. If you finish your claimed subtask early and the other agent has not
   confirmed completion of theirs, either navigate over to visually verify
   (useful right before a DONE — see `feedback_done_verification.md`) or ask
   directly, rather than assuming they're done and calling `DONE` yourself.
