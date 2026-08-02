# GPT-5.5 Harness Iteration Log

This ledger keeps GPT-5.5 trials separate from earlier Doubao and constrained GPT smoke runs. Each row uses an independent CSV with all task results reset.

| Iteration | Branch | Task set | Memory feedback | Launcher `k/s/w` | Status | Success | Failure / cost | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R0 | AI2-THOR single | 03044, 03054, 03064, 03075 | off | omitted | running (Hope 56099972) | pending | pending | Raw baseline |
| R0 | ProcTHOR single | 000, 001, 002, 003 | off | omitted | running (Hope 56099973) | pending | pending | Raw baseline |
| R0 | AI2-THOR dual | 03044, 03054, 03064, 03075 | off | omitted | pending | pending | pending | Raw baseline |
| R0 | ProcTHOR dual | 107, 201, 202, 203 | off | omitted | pending | pending | pending | Raw baseline |

The next iteration may add only one harness intervention to an affected branch.
Memory feedback is considered before image/history restrictions; `k`, `s`, and
`w` are introduced only after trajectory evidence indicates context-size failure.
