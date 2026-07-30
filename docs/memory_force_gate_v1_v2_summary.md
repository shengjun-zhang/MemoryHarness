# Memory 强制查阅机制：v1（软提示） → v2（硬门槛拦截）改动总结

> 涉及文件：
> - `mllm_base_agent/dual_agent/ai2thor/main.py`
> - `mllm_base_agent/dual_agent/procthor/main.py`
> - `mllm_base_agent/agent/runner.py`（单智能体）
>
> 记录日期：2026-07-30

---

## 一、能否用 git 检索到修改前（v1）的版本？

**结论：不能直接检索到。**

- 这三个文件在 git 仓库当前 `HEAD`（commit `f47b1e091985a7f141db8d93c6822b42663cc1da`）里**完全不包含任何 memory 库相关代码**（连最基础的 `ReadMemory` 解析、`memory_lookup` 分支、软性 nudge 都没有），即 `git show HEAD:<path>` 拿到的是"引入 memory 机制之前"的最原始版本，比本文所指的 v1 还要更早。
- 本文所指的 **v1（软提示版：已支持 `ReadMemory`、有文字 nudge，但没有硬门槛拦截）** 和 **v2（本次新增硬门槛拦截）** 都只存在于当前未提交的工作区改动中（`git status` 显示这几个文件均为 `M`，改动尚未 `git add`/`git commit`，也没有任何 stash 或中间分支记录）。
- 也就是说，v1 → v2 这一步的演进**没有被 git 记录为独立的提交点**，无法用 `git log` / `git diff <commit_a> <commit_b>` 之类命令直接调出 v1 的完整文件快照。
- 唯一能做的是：基于本次对话的编辑过程，**手动还原 v1 与 v2 之间的精确代码差异**（见下文第三节），如果需要真正找回可运行的 v1 文件，建议之后修改前先 `git commit`（哪怕是临时 commit）以便留下可检索的历史点。

---

## 二、v1 与 v2 的能力对比

| 维度 | v1（软提示版） | v2（硬门槛拦截版，本次新增） |
|---|---|---|
| 触发方式 | 在 system prompt / 错误提示 / DONE 前，用**文字建议**模型调用 `ReadMemory` | 在满足条件时，**代码层面拦截**非 `ReadMemory` 动作，不执行、不消耗 step |
| 约束力 | 软约束：模型可以读了摘要就在 `<THINK>` 里"自我脑补"，从不真正调用工具 | 硬约束：模型除了 `ReadMemory(...)` 别无选择，其余动作一律被丢弃重提示 |
| 已观测到的问题 | benchmark 显示 `ReadMemory` 实际调用次数为 0，但日志里 17.1% 的轮次在 `<THINK>` 中提到"memory" | 待新一轮 benchmark 验证是否解决"只引用摘要、不执行工具"的问题 |
| 安全阀 | 无需要，因为从不强制 | 设置最大拒绝次数（dual-agent 为 3 次），超过后放弃强制，避免卡死 episode |

---

## 三、v2 新增的核心机制说明

### 3.1 触发条件

当某个 agent（或单智能体自身）**连续 N 次真实动作执行失败**（`N = FORCE_READ_MEMORY_FAILURE_THRESHOLD = 2`）、且这一失败区间内还没有真正调用过 `ReadMemory`，则进入"强制模式"：本轮它提交的下一个动作，只要不是 `ReadMemory(...)`，就会被直接丢弃（不执行、不计入 step 预算、不 handoff 给队友），并返回明确的拒绝提示，要求其重新输出。

### 3.2 三个环境的实现方式对比

| | Dual-agent（ai2thor / procthor） | 单智能体（runner.py） |
|---|---|---|
| "连续失败"的统计方式 | 持久字段 `current_agent["consecutive_failures"]`，每次真实动作执行后 `+=1` 或清零 | 实时从 `structured_trajectory` 倒序统计 `reward < 0.05` 的连续步数（`_count_consecutive_failures`），不额外维护持久字段 |
| "本轮是否已读过 memory" 标记 | 持久字段 `current_agent["memory_consulted_for_streak"]`，失败时重置为 `False`，真正执行 `ReadMemory` 后置 `True` | 局部变量 `memory_consulted_this_turn`（仅在 `think_node` 单次调用的内层 "memory 子循环" 生命周期内有效） |
| 安全阀（防止死循环） | `forced_memory_rejections` 计数，达到 `FORCE_READ_MEMORY_MAX_REJECTIONS = 3` 后放弃强制 | 依赖已有的 `max_memory_lookups_per_turn = 5` 上限（子循环本身自带轮数上限） |
| 拦截点 | 动作解析完成之后、`memory_lookup` 分支判断之前，新增一段 `if action_type != memory_lookup and _should_force_read_memory(...): ... continue` | 在已有的 "ReadMemory 免费子循环" 中，`memory_file_name is None` 分支下新增同样的拦截判断 |

### 3.3 关键函数（三处环境命名基本对称）

- `FORCE_READ_MEMORY_FAILURE_THRESHOLD`：触发阈值，当前为 `2`。
- `_should_force_read_memory(...)`：判断当前是否处于"强制模式"。
- `_force_read_memory_rejection_text(...)`：生成拒绝提示文案，明确告知模型"本轮唯一合法动作是 `ReadMemory(<file_name>)`"。
- （仅 dual-agent）`FORCE_READ_MEMORY_MAX_REJECTIONS`：拒绝次数上限，超过后放行，避免整条 episode 卡死。

### 3.4 状态机简述（以 dual-agent 为例）

```
连续失败计数 consecutive_failures:  0 → 1 → 2（达到阈值，进入强制模式）
                                              │
                                              ▼
                        模型提交非 ReadMemory 动作 → 被拒绝，consecutive_failures 不变，
                                              forced_memory_rejections += 1
                                              │
                                              ▼
                        模型提交 ReadMemory(...) → 允许执行，
                              memory_consulted_for_streak = True
                              forced_memory_rejections 重置为 0
                                              │
                                              ▼
                        模型提交真实动作 → 允许放行尝试
                              ├─ 执行成功 → consecutive_failures 清零，memory_consulted_for_streak 重置
                              └─ 执行失败 → consecutive_failures += 1，
                                          memory_consulted_for_streak 重置为 False（下次达到阈值需再读一次，可换条目）
```

---

## 四、后续验证建议

1. 重新运行一轮 benchmark，重点观察 `memory_reads_used` 指标是否从 0 明显回升。
2. 检查日志中是否出现 `forced_memory_rejections` 达到上限后放行的 episode，评估模型"拒绝配合"硬门槛的比例。
3. 对比强制拦截前后，"blocking" 类失败和 "not in view" 类失败的占比是否下降（这两类问题在 memory 库里有对应的 `feedback_*.md` 条目）。
4. **建议**：后续再做类似的机制性改动前，先执行一次 `git add -A && git commit -m "..."`（哪怕是里程碑式的临时提交），这样未来才能用 `git diff <commit_a> <commit_b>` 精确回溯每个版本之间的差异，而不必依赖对话记录人工回想。

---

## 五、涉及代码位置索引

- `mllm_base_agent/dual_agent/ai2thor/main.py`：`FORCE_READ_MEMORY_FAILURE_THRESHOLD` 附近（约第 692-739 行定义辅助函数），拦截逻辑插入点在动作解析之后、`memory_lookup` 分支之前（约第 1710 行附近）。
- `mllm_base_agent/dual_agent/procthor/main.py`：结构与 ai2thor 对称，辅助函数约第 613-651 行，拦截逻辑约第 1480 行附近。
- `mllm_base_agent/agent/runner.py`：辅助函数约第 1258-1284 行；拦截逻辑分别插入 lookahead 版本 `think_node`（约第 828 行附近）与普通版本 `think_node`（约第 1049 行附近）的 memory 子循环中。
