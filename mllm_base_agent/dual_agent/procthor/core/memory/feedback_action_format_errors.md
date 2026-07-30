# Action-string / parse errors that burn a full retry round-trip

**Observed:** 21 episodes hit a `parse_error` failure_type; the two concrete
patterns below account for the large majority. Each parse error costs a full
extra VLM call (the runtime re-prompts up to 3 times before giving up and
handing off/failing the agent) — this is one of the cheapest categories of
mistake to eliminate entirely.

## 1. `ThrowObject` takes NO object argument

`ThrowObject` throws whatever is **already in your hand** — it is a bare
action, not a targeted one. 4 episodes failed to parse
`"Unrecognized action format: ThrowObject"`-style errors, and separately,
attempts like `ThrowObject(Bowl)` fail because the parser only recognizes the
bare form.

- Correct: `<ACTION>ThrowObject</ACTION>` (no parentheses, no object).
- Wrong: `ThrowObject(Bowl)`, `ThrowObject()`.
- Precondition: you must be holding an object already, or this action is
  meaningless (nothing to throw).

## 2. Missing `<ACTION>` tag

5+ episodes failed with `"Missing <ACTION> tag"` after retries — the model's
raw response omitted the tag entirely (e.g. wrote the action as plain text,
got cut off by a token limit, or wrapped everything in an unrelated format).

- **Always** wrap your chosen action in `<ACTION>...</ACTION>`, even for the
  simplest actions like `DONE` or a bare navigation action.
- Keep `<THINK>` concise — a very long `<THINK>` block increases the chance
  of hitting a token limit before `<ACTION>` is emitted. Prioritize actually
  closing the `<ACTION>` tag over an exhaustive thinking trace.
- Do not nest tags or add markdown code fences around the tags — emit them
  as plain text exactly as shown in the system prompt's output format.

## 3. Object type casing / spelling

The action parser requires **exact PascalCase** object type names as listed
in the system prompt's interactable-object catalogue (e.g. `HousePlant`, not
`houseplant` or `Houseplant`). A misspelled or mis-cased type causes an
"object does not exist in scene" style error even when the correct object is
clearly visible — double-check the type name against the catalogue rather
than guessing a casual English name for what you see.

## 4. Step-size / granularity suffix

Movement actions optionally take a granularity suffix: `MoveAhead(Small)`,
`MoveAhead(Medium)`, `MoveAhead(Large)` (Small=0.25m, Medium=0.5m, Large=1m).
Do not pass a numeric distance directly (e.g. `MoveAhead(0.5)`) — use the
named granularity keywords, or the bare action name for the environment's
default step size.

## General rule

When an action fails to parse, do not just resend a near-identical
malformed string hoping it works — re-read the **exact** action grammar in
the system prompt's `**Action Format Notes**` / `**Output Format**` section
and produce a syntactically exact match before retrying.
