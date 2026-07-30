# Multi-room exploration in procedurally generated ProcTHOR houses

Unlike AI2-THOR's fixed, usually single-room `FloorPlan*` scenes used in the
sibling AI2-THOR benchmark, ProcTHOR houses are **procedurally generated**
and typically span multiple connected rooms (kitchen, bedroom, living room,
bathroom, ...). Target objects are frequently in a room **neither** agent
spawned in, which changes the shape of a good search strategy compared to a
single-room task.

## Why this matters

- A task instruction naming an object (e.g. "bring the Pen to the living
  room") gives no guarantee the object starts anywhere near either agent's
  spawn point — it could be in any connected room.
- Two agents blindly exploring the SAME room in parallel wastes half the
  combined search bandwidth; two agents exploring DIFFERENT rooms in
  parallel covers the house roughly twice as fast, provided they communicate
  what they've each already ruled out.

## Search strategy

1. **Split by room, not by object**, when neither agent has spotted the
   target yet: on the first 1-2 turns, describe your room and pick a
   direction to explore (through a doorway/opening you can see) rather than
   repeating the same forward move in the room you're already in.
2. **Scan before committing to a long traversal**: use `RotateLeft` /
   `RotateRight` and `LookUp` / `LookDown` to survey the current room's
   visible objects and exits before moving far — this catches the target
   early if it's in the current room, and identifies the best doorway to
   try next if it isn't.
3. **Communicate negative results, not just positive ones**: "Kitchen fully
   searched, no Egg here" is as valuable as "Found the Egg" — it prevents
   your partner (or your later self, after a room-count reset in your own
   reasoning) from re-covering the same room.
4. **Track rooms by landmark, not name**: ProcTHOR rooms don't have a
   canonical name in your observation — refer to them by their distinguishing
   furniture ("the room with the Bed and Dresser", "the room with the Sofa
   and TV") in communication so your partner can match your report to what
   they see.
5. **Do not declare FAIL for a "missing" object until every connected room
   reachable so far has been reported searched** (by either agent) — a
   target simply being in an unexplored room is the default explanation for
   "I can't find it," not "it doesn't exist" or "the task is impossible."

## Anti-pattern

Re-entering a room your partner already reported as fully searched (without
new information suggesting something changed) wastes step budget that could
go toward covering new territory. Keep a running mental note (and state it
in `<COMMUNICATE>` if it would help coordination) of which rooms are
confirmed-searched by either agent.
