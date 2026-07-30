# Object interaction gotchas in the ProcTHOR unified action space

ProcTHOR shares its interaction engine and unified action space with the
AI2-THOR dual-agent benchmark (see the sibling entry at
`mllm_base_agent/dual_agent/ai2thor/core/memory/feedback_object_interaction_gotchas.md`
for the full statistics-backed writeup) — everything below applies
identically here.

## Interaction distance is strict and measured to the surface

Maximum interaction distance is **1.0 meter**, measured from you to the
object's **surface**, not its center. An object that "looks close" in a
wide-angle view can still be just past 1.0m — close the gap fully rather
than nudging by the smallest possible step, then verify visually before
retrying.

## Hand-state preconditions

- You can hold **exactly one** object at a time.
- `PickupObject(X)` while already holding something fails with an "already
  holding an object" style error — `DropHandObject`, `PutObject`, or
  `ThrowObject` first, then retry the pickup.
- `DropHandObject` / `PutObject(X)` / `ThrowObject` all **require** you to be
  holding an object already; calling them empty-handed fails.

## "No valid positions to place object"

Means the target receptacle has no free/valid slot from your current
position/orientation — usually because you are standing too close to or on
top of the target surface, or the receptacle is visually full/occupied.
Recovery: step back (`MoveBack`) to give the placement solver room, or
reconsider whether this is the right receptacle instance (see below).

## Abstract actions need no tools

`SliceObject`, `CookObject`, `CleanObject` are simplified in this
environment: they work directly on the target object without requiring you
to find/hold a knife, stove, or cleaning cloth first. Do not waste steps
searching for an instrumental tool object — execute the action on the
target directly once in range. (ProcTHOR's cookable set in this benchmark
is `{Egg, Potato, Tomato}` — check the system prompt's catalogue for the
current list rather than assuming AI2-THOR's exact set transfers 1:1.)

## Multi-instance disambiguation

When a house has multiple objects of the same type (two `CounterTop`s
across different rooms, two `Chair`s), the environment auto-targets the
**nearest visible** instance — you cannot address a specific instance by ID.
If an interaction "succeeds" but the wrong-looking object changed state, you
were likely closer to a different instance than intended; reposition so the
INTENDED instance is unambiguously nearest before interacting. This is more
likely in ProcTHOR than in single-room AI2-THOR tasks simply because houses
have more total furniture instances spread across rooms.

## Exact PascalCase object type names

Object type strings must match the system prompt's catalogue exactly
(`HousePlant`, not `houseplant`/`Houseplant`). A casing/spacing mismatch
produces an "object does not exist in scene" error even when the object is
clearly visible — copy the type name from the catalogue.

## Step granularity is mandatory here

Unlike some AI2-THOR configs, the ProcTHOR dual-agent prompt requires an
explicit granularity suffix on every move: `MoveAhead(Small|Medium|Large)`,
never a bare `MoveAhead` or a numeric distance like `MoveAhead(0.5)`. Using
the wrong form is a common avoidable parse error — see
`feedback_action_format_errors.md` in the AI2-THOR memory library for the
general action-format lessons (the grammar is shared).
