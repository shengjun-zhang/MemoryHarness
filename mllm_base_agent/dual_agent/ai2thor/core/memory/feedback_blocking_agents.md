# "is blocking" errors — unsticking navigation without looping

**Observed:** 413 occurrences across 232 episodes — the #2 most common error
in the dataset, split roughly evenly between:
- **Agent-vs-agent blocking**: `"[Original error] Agent 1 is blocking Agent 0
  from moving by (...)"` (85 occurrences) and the symmetric `Agent 0 is
  blocking Agent 1` (76 occurrences) — the two bodies physically get in each
  other's way.
- **Static-mesh blocking**: `"<MeshName> is blocking Agent N from moving by
  (...)"` — furniture/walls/counters (`ShelvingUnit`, `Chair`, `IslandMesh`,
  `ArmChair`, `StandardWallSize`, `StandardIslandHeight2`, ...) physically
  occupy the path.

In AI2-THOR, a blocked move does **not** bounce or push back — you simply
get **zero movement** for that action and an error string. Agents that keep
re-issuing the same blocked move burn steps for nothing (empirically rare —
only ~3/413 blocking errors were immediately followed by the exact same
action — but every occurrence is a wasted step, and near-repeats with only a
tiny variation are common).

## Recovery pattern

1. **Never repeat the identical action** that just failed with "is blocking".
   The environment state has not changed, so the identical action will fail
   identically.
2. **Static-mesh blocking** (furniture/wall name in the message): the mesh
   name tells you roughly what's in the way even if it's not visible dead
   center (e.g. `StandardIslandHeight2` = kitchen island, `ShelvingUnit` =
   shelf). Try a **lateral step** (`MoveLeft`/`MoveRight`) to sidestep around
   it, or a **smaller step size** if a large-magnitude move overshoots into
   an obstacle a normal-size move would have cleared. Rotating 45-90° and
   re-approaching from a different angle also works when a direct path is
   fully blocked.
3. **Agent-vs-agent blocking**: this means your partner's body physically
   occupies your path. Recovery:
   - `<COMMUNICATE>` your intended move/target immediately — the partner may
     not know they are in your way.
   - Step to the side (`MoveLeft`/`MoveRight`) or wait one turn (`Pass()`)
     for the partner to clear the space instead of repeatedly colliding.
   - If both agents are trying to reach the same tight spot (e.g. both
     converging on the same countertop), negotiate who goes first via
     `<COMMUNICATE>` rather than both re-attempting simultaneously.
4. After a successful detour, only return to the original path if it is
   still relevant to your goal — sometimes the detour itself reveals a
   better route.

## Anti-pattern (do NOT do this)

- Do not treat "is blocking" as a signal to retry the exact same move with
  higher magnitude (e.g. escalating from `MoveAhead(Small)` to
  `MoveAhead(Large)`) — a blocked path stays blocked regardless of step
  size; you need a *different direction*, not a *bigger* one.
- Do not silently give up and pick an unrelated action — a blocked path is
  almost always solvable with a lateral step or re-angled approach within
  1-2 tries.
