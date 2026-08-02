# GPT-5.5 Harness Iteration Log

This ledger keeps GPT-5.5 trials separate from earlier Doubao and constrained GPT smoke runs. Each row uses an independent CSV with all task results reset.

| Iteration | Branch | Task set | Memory feedback | Launcher `k/s/w` | Status | Success | Failure / cost | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R0a | AI2-THOR single | 03044, 03054, 03064, 03075 | unexpectedly on | omitted | stopped (Hope 56099972) | invalid | task 03044: 0/1, max steps after 14 actions and 3 memory lookups | Config omission did not disable the memory harness; add explicit gate and rerun |
| R0a | ProcTHOR single | 000, 001, 002, 003 | not verified | omitted | stopped (Hope 56099973) | invalid | no completed trajectory | Stopped with the same faulty baseline definition |
| R0 | AI2-THOR dual | 03044, 03054, 03064, 03075 | off | omitted | pending | pending | pending | Raw baseline |
| R0 | ProcTHOR dual | 107, 201, 202, 203 | off | omitted | pending | pending | pending | Raw baseline |
| R0b | AI2-THOR single | 03044, 03054, 03064, 03075 | explicitly off | omitted | terminal (Hope 56100914) | 0/4 | max steps: 2; premature DONE: 1; HTTP 413: 1. 48 actions, 48 turns, 0 communications, 418600 tokens | Test only `-w 10` on the 413 task |
| R0b | ProcTHOR single | 000, 001, 002, 003 | explicitly off | omitted | submitted (Hope 56100913) | pending | pending | Corrected raw baseline |
| R0b | AI2-THOR dual | 03044, 03054, 03064, 03075 | explicitly off | omitted | submitted (Hope 56101557) | pending | pending | Corrected raw baseline |
| R0b | ProcTHOR dual | 107, 201, 202, 203 | explicitly off | omitted | submitted (Hope 56101556) | pending | pending | Corrected raw baseline |
| R1 | AI2-THOR single | 03075 | explicitly off | `w=10` only | pending | pending | pending | Isolate the raw-baseline HTTP 413 failure |

The next iteration may add only one harness intervention to an affected branch.
Memory feedback is considered before image/history restrictions; `k`, `s`, and
`w` are introduced only after trajectory evidence indicates context-size failure.
