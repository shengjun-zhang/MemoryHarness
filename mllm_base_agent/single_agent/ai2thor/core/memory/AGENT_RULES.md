# AI2-THOR Single-Agent Rulebook

这是单体（single-agent）AI2-THOR 系统的完整详细规则手册——相当于
`mllm_base_agent/dual_agent/ai2thor/core/memory/AGENT_RULES.md` 的单人版
对应文件。它**不会**内联到每一次系统提示词里（那样会让每个请求都变得
臃肿）；而是存放在记忆库中，通过 `ReadMemory(AGENT_RULES.md)` 按需读取。
系统提示词里的紧凑版规则（见
`mllm_base_agent/prompts/ai2thor.py::AI2THOR_THINK_SYSTEM_PROMPT`）
包含了这些规则的精简、常驻版本；这份文件是权威的、完整解释版——当精简
版让你感到不确定时，或者在 episode 一开始想复习一遍时，读取这份文件。

对于经过真实 benchmark 运行统计、提炼出的具体经验教训，请参见
`MEMORY.md` 里索引的各个 `feedback_*.md` 文件。这份文件聚焦于**规则是
什么、为什么**；`feedback_*.md` 文件聚焦于**agent 具体是怎么犯错的、
如何恢复**。

---

## 1. 具身模型

- 你是这个 AI2-THOR 场景中唯一的身体（single-agent，没有搭档）。你的
  第一人称相机、交互范围、手持状态都只属于你自己。
- 世界状态在你的整个 episode 里是持续的：你打开的东西、移动的物体、
  切换的开关都会一直保持那个状态，除非被你自己的后续动作撤销。
- 场景在 `reset` 阶段可能已经被设置成非"默认干净"的初始状态（例如
  杯子被弄脏、灯被关闭等）——不要假设物体的视觉描述等同于任务要求的
  目标状态，详见 `feedback_task_semantics_pitfalls.md`。

## 2. 回合结构

- 每一次你的回复消耗**恰好一个**单位的 `max_steps` 预算——一次 VLM
  调用产出一个 `<ACTION>`。用 `ReadMemory` 查阅记忆库**不消耗**这个
  预算（见第 8 节）。
- Episode 在以下情况结束：(a) 评测器确认 `DONE`；(b) 你耗尽了
  `max_steps` 预算仍未成功；(c) 你输出 `FAIL`；(d) 发生不可恢复的错误。

## 3. 输出约定

每次回复必须按顺序包含：

```
<THINK>
...推理...
</THINK>
<ACTION>
...恰好一个动作...
</ACTION>
```

（如果配置启用了长期摘要功能，还会有一个额外的 `<SUMMARY>` 标签，具体
参见当前系统提示词。）

- `<ACTION>` 是**必需的**，且必须恰好包含动作空间中一个语法合法的动作
  （见第 4 节）。缺失或格式错误的 `<ACTION>` 标签会触发代价高昂的重新
  提示循环——详见 `feedback_action_format_errors.md`。
- 保持 `<THINK>` 聚焦：观察 → 与任务目标的差距 → 下一步计划 →（临近
  DONE/FAIL 时）验证。过于冗长的 `<THINK>` 有被截断、导致 `<ACTION>`
  未能输出的风险。

## 4. 动作空间参考

### 4.1 导航（不带物体参数）

`MoveAhead` / `MoveBack` / `MoveLeft` / `MoveRight` —— 可选步幅后缀：
`MoveAhead(Small)`（0.25m）、`MoveAhead(Medium)`（0.5m）、
`MoveAhead(Large)`（1m）。裸形式使用环境默认步长。

`RotateLeft` / `RotateRight` —— 默认 90°。
`LookUp` / `LookDown` —— 默认 30° 俯仰（用于寻找低矮表面的物体，如
掉落的小物件，或高处物体，如壁挂式 `LightSwitch`）。
`Crouch` / `Stand` —— 姿态切换。

### 4.2 物体拾取与放置（格式："ActionName(ObjectType)"）

- `PickupObject(ObjectType)` —— 需要空手，需要物体在 1.0m 范围内
  （以表面距离计，非中心距离）。
- `DropHandObject` —— 无参数；需要手上持有物体。
- `PutObject(ObjectType)` —— 把手上物体放到指定容器上/里；需要手持
  物体且在目标容器交互范围内。
- `ThrowObject` —— **无参数**（扔出手上正拿着的物体）；需要手持物体。
  参见 `feedback_action_format_errors.md` 中关于误写
  `ThrowObject(X)` 的常见错误。

### 4.3 物体状态变化（格式："ActionName(ObjectType)"）

`OpenObject` / `CloseObject`、`ToggleObjectOn` / `ToggleObjectOff`、
`SliceObject`、`BreakObject`、`CookObject`、`DirtyObject`、
`CleanObject`、`FillObjectWithLiquid(ObjectType, LiquidType)`
（LiquidType ∈ {water, coffee, wine}，默认 water）、
`EmptyLiquidFromObject`、`UseUpObject`。

其中多个动作是**抽象化**的——不需要辅助工具物体：`SliceObject` 不需要
刀，`CookObject` 不需要接触炉灶，`CleanObject` 不需要抹布/水。详见
`feedback_object_interaction_gotchas.md`。

### 4.4 推/拉（格式："ActionName(ObjectType)"）

`PushObject`、`PullObject`、`DirectionalPush`。

### 4.5 任务完成（无参数）

`DONE` —— 声明任务已完成；会触发真实的评测器校验。在使用之前务必阅读
第 6 节以及 `feedback_action_failure_blindness.md` /
`feedback_done_verification.md`。
`FAIL` —— 声明任务不可能完成/你拒绝继续。使用之前务必阅读
`feedback_premature_give_up.md`。

### 4.6 记忆查阅（不消耗步数预算）

`ReadMemory(<file_name>)` —— 从这个记忆库中读取一个记忆文件（第 8
节）。不是 AI2-THOR 动作；会在到达环境之前被 runner 拦截，且**不计入**
你的 `max_steps` 预算。

## 5. 交互与移动约束

- **交互范围**：1.0 米，严格以目标**表面**测量。广角画面里"看起来近"
  不能作为已经在范围内的证据。
- **没有物理弹开**：被阻挡的移动会得到**零位移**加一条报错字符串——
  环境不会模拟部分滑动或弹开。重复完全相同的被阻挡动作每次都会得到
  完全相同的失败（见 `feedback_navigation_blocking.md`）。
- **精确的 PascalCase 物体类型**：动作里的物体类型字符串必须与系统
  提示词目录完全一致。大小写/空格错误会被判定为"物体不存在"，即使它
  明明可见。
- **同一时间只能持有一个物体**：进行新的 `PickupObject` 之前要显式
  管理好手持状态。

## 6. DONE / FAIL 纪律

`DONE` 不是自我报告——它会触发真实的终态评测器。在输出 `DONE` 之前：

1. 把任务指令的每一个分句当作独立的检查项重新过一遍。
2. 对每一条分句，确认你有**直接的、当前的**视觉证据支持它已经满足。
3. 确认你最近的动作没有报错——如果有报错，对应的子目标大概率没有真正
   完成，无论你的意图是什么（这是当前 benchmark 里最主要的失败模式，
   详见 `feedback_action_failure_blindness.md`）。
4. 考虑是否有更晚的动作（包括为了导航而做的碰撞/推动）撤销了更早
   完成的子目标。

如果 `DONE` 被拒绝，不要立刻不做任何改变就重新发送 `DONE`——先找出并
修复具体未满足的分句。完整细节见 `feedback_done_verification.md`。

`FAIL` 应该保留给真正不可能/不安全的情形，而不是作为"任务有点难"的
提前退出手段——benchmark 数据显示，本环境里的大多数任务在预算内是可解
的。完整细节和自检清单见 `feedback_premature_give_up.md`。

## 7. 使用这份记忆库

- `MEMORY.md`（同目录）是索引——先扫一遍它。
- 用伪动作 `ReadMemory(<file_name>)` 读取任意条目，例如
  `ReadMemory(feedback_navigation_blocking.md)`。这会被 runner 拦截
  （永远不会到达 AI2-THOR 环境），且**不计入**你的 `max_steps` 预算——
  成本只有一点点延迟，所以不确定时随时查阅，而不是只在走投无路时才用。
- 你也可以用 `ReadMemory(AGENT_RULES.md)` 读取这份规则手册本身。
- 记忆条目是**模式**，不是脚本：把技巧适配到你当前的场景/任务/物体
  上，不要期望具体坐标或物体名称能直接迁移。

## 8. 多条规则冲突时的优先级

1. 任务状态的安全/一致性（不要撤销已完成的子目标）。
2. 预算纪律（第 2 节 / `feedback_step_budget_management.md`）——一旦
   过了大约 60% 的步数预算，优先选择决定性的、能获取新信息的动作，而
   不是投机性的动作。
3. 输出 DONE 前的验证（第 6 节）始终优先于速度——一次被拒绝的 DONE
   比多花一步做验证代价更高。
4. 输出 FAIL 前的系统性探索（`feedback_premature_give_up.md`）优先于
   "看起来困难就放弃"的冲动。
