# Object interaction gotchas in the AI2-THOR unified action space

Collected from repeated `action_error` / `model_error` patterns across
benchmark runs. These are smaller in aggregate volume than the top-level
categories (`not in view`, `blocking`, `DONE` rejection, step budget) but
each is a clean, deterministic fix once recognized.

## Interaction distance is strict and measured to the surface

Maximum interaction distance is **1.0 meter**, measured from you to the
object's **surface**, not its center. An object that "looks close" in a
wide-angle view can still be just past 1.0m — if an interaction fails with a
distance-based error, close the gap fully rather than nudging by the
smallest possible step, then verify visually before retrying.

## Hand-state preconditions

- You can hold **exactly one** object at a time.
- `PickupObject(X)` while already holding something fails with an
  "already holding an object" style error — `DropHandObject`, `PutObject`,
  or `ThrowObject` first, then retry the pickup.
- `DropHandObject` / `PutObject(X)` / `ThrowObject` all **require** you to be
  holding an object already; calling them empty-handed fails.

## "No valid positions to place object" (17 occurrences observed)

This means the target receptacle has no free/valid slot from your current
position/orientation — usually because you are standing too close to or
directly on top of the target surface, or the receptacle is already visually
full/occupied. Recovery: step back (`MoveBack`) to give the placement solver
room, or reconsider whether this is the right receptacle instance (see the
multi-instance note below).

## Abstract actions need no tools — do not go looking for one

`SliceObject`, `CookObject`, `CleanObject`, `BreakObject`, and similar
"abstract" interactions are simplified in this environment: they work
directly on the target object without requiring you to find/hold a knife,
stove, or cleaning cloth first. Do not waste steps searching for an
instrumental tool object for these — just execute the action on the target
directly once you're in range.

## Sliced/cracked object identity changes

- `SliceObject(X)` produces `XSliced` (e.g. `SliceObject(Bread)` →
  `BreadSliced`, `SliceObject(Tomato)` → `TomatoSliced`).
- `BreakObject(Egg)` produces `EggCracked` (not `EggSliced` — Egg is the one
  exception to the naming pattern).
- If a task requires interacting with the transformed object afterward (pick
  it up, cook it, place it), you must now refer to the NEW type name
  (`BreadSliced`, `EggCracked`, etc.), not the original.

## Multi-instance disambiguation

When a scene has multiple objects of the same type (two `CounterTop`s, two
`Chair`s, two `ArmChair`s), the environment auto-targets the **nearest
visible** instance of the requested type — you cannot address a specific
instance by ID through the action string. If an interaction seems to
"succeed" but the wrong-looking object changed state, you were likely closer
to a different instance than you thought; reposition so the INTENDED
instance is unambiguously the nearest one in view before interacting.

## Exact PascalCase object type names

Object type strings must match the system prompt's catalogue exactly
(`HousePlant`, not `houseplant`/`Houseplant`/`House Plant`). A casing or
spacing mismatch produces an "object does not exist in scene" error even
when the object is clearly visible — copy the type name from the catalogue
rather than composing it from the object's plain-English appearance.
