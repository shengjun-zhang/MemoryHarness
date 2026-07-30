# Communication cadence and content that correlates with success

**Observed:** across 232 episodes, successful episodes averaged **~1.77
communications per global step**, while failed episodes averaged only
**~1.36 per step** — roughly 30% less communication. Communication is cheap
(it does not consume an environment step; a `<COMMUNICATE>` can accompany
any action, including `Pass()`), so under-communicating is close to pure
downside.

## What to communicate, and when

1. **At episode start** (first 1-2 steps of each agent): describe what you
   see and your intended initial direction/subtask, even before you've found
   anything task-relevant. Your partner cannot see your view at all — silence
   here means they are navigating blind relative to your position.
2. **On every discovery relevant to the task**: object type + approximate
   location/landmark + current state (open/closed/on/off). E.g. "Found a
   closed Fridge against the far wall of the kitchen, right side." Vague
   reports ("found something") are close to useless — be specific.
3. **On every action failure**: state what failed and why (from the error
   message) and what you're doing about it. This prevents your partner from
   independently colliding with the same obstacle or duplicating a failed
   approach.
4. **On task/subtask completion**: state exactly what you completed and how
   you verified it, so your partner does not redundantly re-attempt it, and
   so they can factor it into their own DONE verification.
5. **Before claiming DONE**: state your own verification evidence (what you
   personally observed) rather than only stating the conclusion — this
   invites your partner to sanity-check before you commit an action budget
   step to a possibly-rejected DONE.
6. **When requesting help/coordination**: ask a specific, answerable
   question ("Are you near the LivingRoom Sofa yet?") rather than an open
   one ("What are you doing?").

## Division-of-labor communication (see also feedback_task_division.md)

State your claimed subtask explicitly ("I'll handle the Fridge, can you
get the Bowl to the CounterTop?") rather than assuming your partner infers
it from your movement alone — they cannot see your movement, only your
messages.

## Anti-patterns

- **Silent execution**: taking many actions in a row with no communication
  while your partner is equally silent is the collaboration-mode equivalent
  of two single agents ignoring each other — it forfeits the entire point of
  having two bodies with different vantage points.
- **Restating without new information**: repeating "I'm still looking" every
  turn without any new detail wastes message budget/attention; only
  communicate when there is new state, a new plan, or an explicit question.
- **Trusting a partner's claim without visual confirmation before DONE** —
  see `feedback_done_verification.md`. Communication informs your plan; it
  does not substitute for your own visual verification at the DONE moment.
