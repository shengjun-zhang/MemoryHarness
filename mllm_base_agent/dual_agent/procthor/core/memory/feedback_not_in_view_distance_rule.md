# "X is not in view" — distance decides ROTATE vs MOVE CLOSER

**Observed:** 728 occurrences across 232 benchmark episodes — the single most
common error message in the entire dataset (Book, LightSwitch, Laptop,
Television, Pillow, SoapBottle, FloorLamp, Apple, DeskLamp, CoffeeMachine,
Tomato, Pot, CellPhone, Bed, Mirror, Bread all appear repeatedly). It is
also the error most often mishandled: agents default to "move closer" even
when the object is already close but simply off-screen.

## The rule

The error message always carries the object's distance, e.g.:

```
Book is not in view, need to approach or adjust view (distance: 0.6m)
```

Use the distance to choose your recovery, do not default to moving:

- **`distance < 1.0m`** (already within interaction range): the object is
  CLOSE but off-camera. **ROTATE** (`RotateLeft` / `RotateRight`) or tilt the
  camera (`LookUp` / `LookDown`) to bring it into frame. Moving closer here
  is wasted — you are already in range, you just aren't looking at it.
  - Empirically ~7% of "not in view" errors fall in this bucket (50/728), but
    they are the ones agents get wrong most often (repeatedly moving instead
    of rotating), burning several extra steps each time.
- **`distance >= 1.0m`**: the object is genuinely far. **MOVE** toward it
  (`MoveAhead`, or the granular variant if available) while keeping it
  roughly centered in view; re-check distance after each move rather than
  covering the whole distance in one blind step.

## Practical recovery loop

1. Read the reported `distance` in the error text.
2. `distance < 1.0` → rotate/tilt only, do not move.
3. `distance >= 1.0` → move toward the last-seen direction of the object,
   1-2 steps at a time, re-observing between moves (the object may leave
   frame again if you overshoot an angle).
4. If you rotate/tilt twice and still cannot see the object at all (not even
   a partial edge), you likely mis-estimated its direction — do a fuller
   sweep (e.g. 2-3 consecutive `RotateRight`) before assuming it moved or is
   in another room.
5. Tell your partner what happened (`<COMMUNICATE>`) if this consumes more
   than 2-3 steps — they may already know where the object is, or may be
   able to describe it from their own view.

## Anti-pattern (do NOT do this)

Repeating the *exact same* action after a "not in view" error rarely helps
(if it didn't bring the object into view once, it usually won't the second
time either, because your relative orientation to the object hasn't
meaningfully changed). Vary the recovery: alternate rotate direction, add a
`LookUp`/`LookDown` if the object might be on a low/high surface (e.g. a
`LightSwitch` on the wall, a `Pillow` on a bed), or ask your partner for a
hint.
