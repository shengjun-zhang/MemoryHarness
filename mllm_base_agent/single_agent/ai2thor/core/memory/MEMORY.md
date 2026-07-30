# AI2-THOR Single-Agent Memory Index

这是单体（single-agent）AI2-THOR 技能记忆库的入口（结构与
`mllm_base_agent/dual_agent/ai2thor/core/memory/` 一致，但内容针对单个
身体、无搭档协作的场景重写/精简）。这是一份经过审阅、相对静态的经验
知识库，汇总了真实 benchmark 运行中发现的操作经验、常见数字、坑点，以及
可复用的失败恢复模式。

**当前基线（2026-07-28）**：`doubao_ai2thor_k3_s0.5_w10_20260728_135530`
批次共 311 个任务，成功率 **0%**（全部为 `failed_model`）。本记忆库的
前三个条目就是直接针对这次 0% 成功率的失败归因分析撰写的，是当前最
需要优先修复的问题。

**如何使用这份文件：**
1. 先浏览下面的一行摘要，找到与你当前处境相关的条目（重复的报错、
   卡住的导航、即将输出 DONE、考虑放弃任务……）。
2. 用 `ReadMemory(<file_name>)`（不带引号，例如
   `ReadMemory(feedback_action_failure_blindness.md)`）读取完整条目。
   读取记忆**不消耗**你的动作步数预算——不确定时随时可以查阅。
3. 条目是可复用的**模式**，不是可以照抄的脚本：请把技巧适配到你当前
   的场景/任务，不要期望坐标或物体名称能直接迁移。
4. 完整规则手册（比系统提示词内联版本更详细）见同目录下的
   `AGENT_RULES.md`。

**建议主动查阅记忆的时机（不要等到卡死了才想起来）：**
- 每个 episode 的第一步之前（几乎零成本的保险）。
- 看到一个你不确定怎么处理的报错时。
- 输出 `DONE` 之前（务必重新查阅
  `feedback_action_failure_blindness.md` 和 `feedback_done_verification.md`）。
- 考虑输出 `FAIL` 之前（务必先查阅 `feedback_premature_give_up.md`）。
- 同一类动作/报错连续失败 2 次以后。

## 索引（一行摘要 —— 读取对应文件获取完整细节）

- `feedback_action_failure_blindness.md` — **头号问题**：交互动作失败
  （距离超范围等）后模型未察觉，下一步凭空断言成功并输出 DONE。这是
  当前 58%（180/311）DONE 类失败的直接原因，输出 DONE 前必读。
- `feedback_premature_give_up.md` — 过早输出 FAIL：仅看 1-3 个方向、
  或对同一物体重复 1-2 次失败尝试就断言"不存在/无法完成"，实际上
  golden_actions 显示大多数任务在预算内可解。输出 FAIL 前必读。
- `feedback_task_semantics_pitfalls.md` — 任务隐含前置条件与动作语义
  误解：指令里的状态词（"clean"/"full"/"open"等）描述的是目标状态而非
  当前既定事实；动词到 API 的映射也可能有歧义（如"open laptop"到底是
  开机还是掀盖）。
- `feedback_navigation_blocking.md` — 被家具/墙体阻挡时如何脱困：33%
  的失败是触顶 step limit，很大比例源于被阻挡后只会"原地转圈"而不会
  侧移/变换步幅/改变路线。
- `feedback_not_in_view_distance_rule.md` — "X is not in view" 报错：
  用报错里的距离数字决定该旋转还是该移动，选错会浪费大量步数。
- `feedback_done_verification.md` — DONE 被拒绝的常见原因与输出 DONE
  前的强制自检清单（子目标拆解、状态漂移、多实例歧义）。
- `feedback_action_format_errors.md` — 动作字符串/解析错误
  （`ThrowObject(X)`、缺失 `<ACTION>` 标签、物体名大小写错误）——每次
  都要浪费一整轮重试。
- `feedback_step_budget_management.md` — 步预算耗尽是第二大失败原因
  （33%）：节奏把控与恢复循环止损的经验法则。
- `feedback_object_interaction_gotchas.md` — 统一动作空间里的手持状态、
  交互距离、抽象动作（无需工具）、切片/破碎后物体改名等具体坑点。
