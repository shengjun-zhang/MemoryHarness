# AI2-THOR 双体协作 Memory 机制实验分析

> 分析对象：`mllm_base_agent/dual_agent/ai2thor/benchmark_outputs/ai2thor-memory_20260730_120603`
> 数据来源：该批次全部 16 个 shard（`shard-000-of-016` ~ `shard-015-of-016`）的 `benchmark_summary.json` 及各任务 `live.log`
> 生成日期：2026-07-30

---

## 一、实验背景

本批次是引入**技能库/经验记忆机制**（`mllm_base_agent/dual_agent/ai2thor/core/memory/`）之后的一次 16-shard 并行 benchmark。该机制的构成：

- `MEMORY.md`：索引文件，罗列 8 篇 `feedback_*.md` 经验条目的一句话摘要（每条摘要标注了历史统计依据，例如"步数耗尽是 78/150 失败的头号原因"）。
- `feedback_not_in_view_distance_rule.md` / `feedback_blocking_agents.md` / `feedback_done_verification.md` / `feedback_action_format_errors.md` / `feedback_communication_protocol.md` / `feedback_task_division.md` / `feedback_step_budget_management.md` / `feedback_object_interaction_gotchas.md`：8 篇具体经验条目。
- Agent 可通过 `ReadMemory(<file_name>)` 按需查阅任意条目，**不消耗 step 预算**，系统提示词中提示"每回合首个动作前 / 遇到不会处理的报错时 / 连续失败2次后 / 输出 DONE 前"均应主动查阅。

实验配置：`experiments/configs/ai2thor/dual/config_close_doubao-2.yaml`，`Spatial-Annotation-ai2thor-doubao-seed-2.0-lite.csv`，29 个任务按 `round_robin` 切分到 16 个 shard 并行执行。

---

## 二、总体指标

| 指标 | 数值 |
|---|---|
| 总任务数 | 29 |
| 成功 | **10（34.5%）** |
| 模型失败 failed_model | 16（55.2%） |
| 外部失败 failed_external（均为 API 413） | 3（10.3%） |
| 基础设施失败 failed_infra | 0 |
| 总耗时（16 shard 并行合计） | 19733s ≈ 5.5 小时 |
| 平均单任务耗时 | 680.5s |
| agent_1 / agent_2 总步数 | 318 / 265 |
| 通信事件 / 轮次 总和 | 589 / 443 |

### 2.1 与历史批次对比

| 批次 | 成功率 | 备注 |
|---|---|---|
| `doubao_dual_fix_llm_histroy_feedback_20260701_183820` | 13/29 = 44.8% | 早期版本，无 memory 库，单机顺序跑 |
| `doubao_dual_fix_llm_histroy_feedback_s0.5-memory_20260729_151745` | 8/29 = 27.6% | image_scale=0.5，含 memory 库，单机跑（无分片） |
| **`ai2thor-memory_20260730_120603`（本次）** | **10/29 = 34.5%** | 16-shard 并行，含 memory 库 |

本次相较昨日（07-29）同样带 memory 机制的批次成功率有所回升（27.6% → 34.5%），但仍低于 07-01 那次**不含** memory 库的历史峰值（44.8%）。样本量仅 29，任务难度分布不均，波动本身较大，尚不能据此判断 memory 机制的净效果为正或为负。

### 2.2 Shard 级别分布

29 个任务分散在 16 个 shard 中（每 shard 1~2 个任务）。其中 5 个 shard（001、009、014、015 等）任务全部成功，6 个 shard（002、003、004、005、010、012）任务全部失败，波动较大，符合各 shard 任务难度不均的特点（round_robin 分配非同质）。

---

## 三、失败原因分布

| 失败原因 | 次数 | 占比（占全部 19 个失败） |
|---|---|---|
| **Reached max steps before task completion**（步数耗尽） | 11 | 57.9% |
| Consecutive 4 action failures（连续动作失败提前终止） | 3 | 15.8% |
| API 413 Request Entity Too Large（请求体过大） | 3 | 15.8% |
| Model claimed DONE but success conditions not met | 1 | 5.3% |
| communication / pass action | 1 | 5.3% |

**步数耗尽仍是头号问题**，占比 57.9%，与 `feedback_step_budget_management.md` 中记录的历史规律（"#1 failure reason，占全部失败 50%+"）高度吻合——这条经验虽已沉淀进 memory 库，但并未在本轮实验中转化为 agent 行为层面的改进。

### 3.1 新出现的问题：413 请求体过大

本批次出现 3 起此前未在同类报告中重点提及的 **413 Request Entity Too Large**（`ai2thor04046`、`ai2thor04108`、`ai2thor05007`），推测与历史对话/图片累积导致请求体膨胀有关，建议复核 `image_scale`、`image_recent_steps` 等图像压缩参数是否对本次配置生效。

### 3.2 成功任务的动作效率

成功任务的平均 `actual_actions / golden_actions` 比例为 **5.4 倍**：即便任务最终完成，agent 也需要远超"标准答案"步数的尝试次数，这与"步数耗尽是失败首因"相互印证——效率问题比正确性问题更突出。

---

## 四、Memory 机制的实际使用情况（核心发现）

对 29 个任务的 `live.log` 逐一检索 `ReadMemory(...)` 调用记录：

| task_id | 任务结果 | ReadMemory 调用次数 | 查阅的条目 |
|---|---|---|---|
| ai2thor03038 | failed_model | 1 | feedback_object_interaction_gotchas.md |
| ai2thor05063 | **success** | 1 | feedback_action_format_errors.md |
| ai2thor05506 | failed_model | 1 | feedback_not_in_view_distance_rule.md |
| ai2thor05533 | failed_model | 1 | feedback_not_in_view_distance_rule.md |
| ai2thor05022 | failed_model | 1 | feedback_blocking_agents.md |
| ai2thor05026 | failed_model | 1 | feedback_not_in_view_distance_rule.md |
| 其余 23 个任务 | — | 0 | — |

**关键数据**：

- 仅 **6/29（20.7%）** 的任务在整个 episode 中调用过 `ReadMemory`，其余 **23/29（79.3%）** 任务全程未查阅任何经验条目。
- 调用最多的条目是 `feedback_not_in_view_distance_rule.md`（3 次），其次是 `feedback_object_interaction_gotchas.md`、`feedback_action_format_errors.md`、`feedback_blocking_agents.md`（各 1 次）。
- 每个任务**最多只调用 1 次**，与 `MEMORY.md` 中"每次遇到同类错误都应查阅""连续失败 2 次后应查阅"的建议使用频率不符，属于**低采纳率、浅层使用**。
- 调用过 memory 的 6 个任务中，**5 个仍然失败**，仅 `ai2thor05063` 成功——查阅记录并未显著扭转任务结局。
- `feedback_step_budget_management.md`（针对占比最高的"步数耗尽"问题）**全程 0 次被调用**，而步数耗尽恰恰是本批次占比 57.9% 的头号失败原因，二者形成鲜明反差。

**结论**：memory 库已经把历史 232 局的经验教训（视距判断、阻塞解除、步数管理等）沉淀成文档，但 agent 主动查阅的比例很低（约 1/5 任务，且集中在少数几类报错上），最相关的步数管理条目完全未被触达，因此本轮实验尚不能体现 memory 机制的预期收益。

---

## 五、可提升的方向

1. **提升 memory 机制的实际调用率**：当前依赖 agent 在 prompt 提示下"主动"查阅，实测采纳率仅 20.7%，且从未触达最relevant的步数管理条目。可考虑将关键条目（尤其 `feedback_step_budget_management.md`）的核心结论直接内嵌进 system prompt 常驻上下文，而非完全依赖按需查阅。
2. **优先排查新出现的 413 错误**：本批次 3 起 413 错误此前未见于同规模报告，建议检查历史图像/对话压缩逻辑在当前配置（16-shard 并行）下是否与单机批次表现一致。
3. **步数管理仍是首要瓶颈**：57.9% 的失败为步数耗尽，建议在 prompt 层面加入更明确的"已用 X/Y 步"进度提醒，配合 `feedback_step_budget_management.md` 中的 pacing 建议，而不是仅依赖 agent 自行查阅。
4. **样本量偏小，结论需谨慎**：单批次仅 29 个任务，与历史 8 个批次动辄 29 条 × 多批次的量级相比，本次 memory 机制的净效果评估建议积累更多批次数据后再下定论，并尽量采用与历史批次相同的运行方式（单机顺序 vs 16-shard 并行）以排除环境差异干扰。

---

## 六、数据口径与复核索引

- **统计范围**：`ai2thor-memory_20260730_120603/shard-000-of-016` ~ `shard-015-of-016` 下全部 16 个 `benchmark_summary.json`，合计 29 条 `task_records`。
- **成功/失败判定**：直接采用 `benchmark_summary.json` 中的 `status`/`success` 字段（`success` / `failed_model` / `failed_external`）。
- **Memory 使用统计**：对每个任务目录下的 `live.log`（非 `failed_logs/` 下的重试日志）做正则匹配 `ReadMemory\(([^)]*)\)` 统计调用次数与目标文件。
- 关键文件位置（供进一步复盘）：
  - `mllm_base_agent/dual_agent/ai2thor/benchmark_outputs/ai2thor-memory_20260730_120603/shard-000-of-016/ai2thor03038/live.log`（唯一同时含步数耗尽失败与 memory 调用的样本）
  - `mllm_base_agent/dual_agent/ai2thor/benchmark_outputs/ai2thor-memory_20260730_120603/shard-006-of-016/ai2thor04046/`、`shard-007-of-016/ai2thor04108/`、`shard-011-of-016/ai2thor05007/`（3 起 413 错误样本）
  - `mllm_base_agent/dual_agent/ai2thor/core/memory/MEMORY.md`（memory 库索引与使用说明）
  - `mllm_base_agent/dual_agent/ai2thor/benchmark_outputs/doubao_dual_fix_llm_histroy_feedback_s0.5-memory_20260729_151745/benchmark_summary.json`（对比批次：同样含 memory 库，单机跑）
