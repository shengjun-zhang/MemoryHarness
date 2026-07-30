# DONE rejection — mandatory self-check before claiming completion

**Observed:** 105 `DONE` calls across 232 episodes; a large share were
rejected by the terminal-state evaluator on the first attempt
("Model claimed DONE but success conditions were not met" is the #4 most
common failure reason, 17 occurrences as the terminal fail_reason — and many
more `DONE`s were rejected mid-episode and recovered from). In 9 observed
cases, an agent called `DONE`, got rejected, and then called `DONE` again
within the next 5 steps **without changing anything about the world state in
between** — i.e. it re-asserted the same wrong belief instead of fixing the
actual gap.

## Why DONE gets rejected

`DONE` is trusted-but-verified: your call triggers the real evaluator against
the *actual* AI2-THOR object graph, not your belief about it. The most common
gaps observed between "agent thinks done" and "evaluator says no" are:

1. **Trusting the partner's message instead of your own eyes.** A partner
   saying "I put the Bowl on the CounterTop" does not make it true from the
   evaluator's perspective if the action actually failed, was misdirected
   (wrong receptacle instance), or affected a different receptacle than
   intended.
2. **Partial subgoal completion.** Multi-object / multi-state tasks (e.g.
   "put two eggs in the pan", "open the fridge AND turn off the light") are
   evaluated as a conjunction — 90% done still evaluates to failure. Recount
   every clause in the instruction, not just the one you personally worked on.
3. **State drift.** An object you or your partner changed earlier (opened,
   toggled, picked up) can be changed back by a *later* action (yours,
   partner's, or an interaction side-effect) — verify the CURRENT frame, not
   a frame from several steps ago.
4. **Wrong instance disambiguation.** When a scene has multiple objects of
   the same type (two `CounterTop`s, two `Chair`s), the interaction defaults
   to the nearest instance — confirm the STATE actually visible in front of
   you matches the task's implied target, not just "an object of the right
   type exists somewhere."

## Mandatory pre-DONE checklist

Before outputting `DONE`, answer all of the following explicitly (put this
reasoning in your `<THINK>` block):

1. List every clause/subgoal in the task instruction separately.
2. For each clause, can you see (in YOUR current image) direct visual
   evidence it is satisfied right now? If a clause was completed by your
   partner and you cannot see it yourself, ask them to confirm with a fresh
   description, or navigate to verify it visually before claiming DONE.
3. Did your OWN last action(s) return any error? If yes, that part of the
   task is not done regardless of what you intended.
4. Is there any chance a later action undid an earlier one (e.g. closed
   something that needed to stay open)?
5. If you answered "no" or "unsure" to any of the above, do NOT output
   DONE — take one more verification action (move/rotate to look, or ask
   your partner) first.

## After a DONE rejection

If your `DONE` is rejected: do not immediately call `DONE` again without
changing anything. Identify which specific clause is still failing (rotate
back to inspect it, or ask your partner), fix or re-verify it, and only then
re-attempt `DONE`. If you genuinely believe you already satisfy everything
and the environment disagrees, double-check the exact object/receptacle
instance you interacted with (see gap #4 above) before retrying.
