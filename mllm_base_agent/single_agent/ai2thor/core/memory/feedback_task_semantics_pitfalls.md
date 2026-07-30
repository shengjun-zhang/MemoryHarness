# 任务隐含前置条件与动作语义误解——看起来"做对了"但漏掉了一步

**Observed:** 2026-07-28 single-agent AI2-THOR benchmark 抽样分析中发现
多个失败案例并非动作执行报错、也不是探索不足，而是模型**从一开始就
误解了任务的隐含前置条件或某个动作的真实语义**，导致整条动作序列
"看起来很合理"却始终无法满足 `success_conditions`。

## 案例 1：忽略场景初始化埋入的隐藏状态（"clean mug" 陷阱）

`ai2thor03065`（"Please put a clean mug into the coffee machine and turn
it on so the mug is filled with coffee."）：

- `task.json` 的 `golden_actions` 明确要求：
  `MoveAhead → CleanObject(Mug) → PickupObject(Mug) → PutObject
  (CoffeeMachine) → ToggleObjectOn(CoffeeMachine) → Done`
- `success_conditions` 明确要求 `Mug.isDirty == false`。
- 但环境初始化时把这个杯子设置成了**脏的**（`DirtyObject(Mug)` 在
  reset 阶段被调用）。
- 模型的 Observation Description 从头到尾都写"clean mug on the chair"，
  完全没有识别出杯子处于脏污状态，跳过了 `CleanObject` 步骤，直接
  拾取、放入咖啡机、加热——四步操作全部"正确执行"且无报错，但因为
  杯子从未被清洗，最终 `isDirty=false` 条件不满足，判定失败。

**教训**：任务指令中出现的形容词（"clean"、"dirty"、"empty"、"full"、
"open"、"closed" 等状态词）往往描述的是**任务要求达成的目标状态**，
而不是物体的当前状态的既定事实。不要仅凭物体的外观描述就假设它已经
处于目标状态——如果任务提到"a clean X"，先假设 X 可能是脏的，并在
执行流程中主动安排一次状态检查/纠正动作（如 `CleanObject`），除非你
有明确的视觉证据（比如物体表面确实没有污渍纹理/脏污贴图）。同理："a
full battery"、"an empty trash can"、"a locked door" 等描述都可能与
任务的**当前**状态相反，需要先验证再决定是否要执行相应的状态改变
动作。

## 案例 2：动词到 API 的映射偏差

- 把 "open my laptop"（意图是开机）字面理解为掀开笔记本盖子，误用
  `OpenObject(Laptop)` 而不是应该触发 `isToggled=true` 的开机动作
  `ToggleObjectOn(Laptop)`。AI2-THOR 中 `Laptop` 同时是 `openable`
  （物理开合盖子）和 `toggleable`（电源开关）两种属性，指令的自然语言
  含义需要结合任务上下文判断到底指的是哪一个。
- 用 `DirectionalPush(Bed)` 代替 `CleanObject(Bed)` 来"整理床铺"——
  推动床铺不会改变任何"整洁度"相关的状态位，属于选错了动作类别。

**教训**：当任务指令中的动词有歧义（如"open"可能指开机也可能指打开
盖子/门）时，先在 `<THINK>` 中明确列出该物体在系统提示词"Interactable
Objects"目录中标注的全部相关属性（openable / toggleable / cleanable
等），选择与任务目标状态最直接对应的那个动作，而不是选择字面上最先
想到的动词翻译。

## 案例 3：物体类型名用错且屡次报错也不自我纠正

`ai2thor04100`（"Move pens to side table"）：连续 3 次使用
`PutObject(CounterTop)`，每次都报错 "CounterTop does not exist in
scene"（正确应为 `SideTable`，与任务描述完全一致的名字），但模型从未
怀疑自己的物体类型名用错了，只是原样重复同一个错误动作。

**教训**：当同一个动作反复报错 "object does not exist in scene" 或
类似错误时，第一时间应该怀疑**物体类型名拼写/选择错误**，并回头对照
任务指令原文和系统提示词的 Receptacles/Pickupable 目录核对拼写，而不
是假设是距离或朝向问题反复重试同一个错误名字。

## 通用检查清单

在执行任何"看起来已经完成"的多步骤任务的最后一步之前，问自己：

1. 任务指令里出现的每一个形容词/状态词，我是否**验证**过（而非假设）
   物体的当前状态符合要求？如果任务要求"clean/full/open/off"等目标
   状态，且物体初始状态未知，默认先执行相应的校正动作更安全。
2. 我选用的动作 API 是否真的对应任务动词的**功能性含义**，而不是字面
   翻译？可疑时对照系统提示词里该物体类型的属性列表（openable /
   toggleable / cleanable / fillable 等）逐一核对。
3. 如果某个物体类型名连续报错"does not exist"，先怀疑拼写/命名而不是
   环境或距离问题，回到任务描述和系统提示词的目录里重新核对准确名称。
