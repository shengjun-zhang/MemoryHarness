# Agentic Planner 分析：SpatialWorld 现状 vs RPent 实现方式

本文档整理两部分分析：
1. SpatialWorld 当前 `mllm_base_agent` 是否具备调用外部工具（tool calling）的能力；
2. 参考 `RPent` 项目，分析它是如何实现"agentic planner"（多轮自主工具调用规划）而不是简单的单步推理，并与 SpatialWorld 现状做对比，给出可落地的迁移路径。

---

## 一、SpatialWorld 当前 agent 是否具备 tool-calling 能力

结论：**不具备**。当前架构是固定的 Think → Act → Evaluate 单步循环，VLM 每次只输出一个离散动作字符串，不存在通用的“调用外部工具/模型”机制。

### 1.1 核心执行循环

`mllm_base_agent/agent/runner.py` 中 `AgentRunner` 的主循环：

```python
active_think_node = lookahead_think_node if decision_mode == 'lookahead' else think_node
iterations = 0
while True:
    if iterations >= limit:
        raise GraphRecursionError(f'Recursion limit reached: {limit}')
    iterations += 1
    state = active_think_node(state)
    yield {'think': state}
    state = act_node(state)
    yield {'act': state}
    state = evaluate_node(state)
    yield {'evaluate': state}
```

`think_node` 只做一件事：把当前帧图像 + 历史对话拼成 prompt，调用 VLM 得到一段文本，用规则/正则解析成**一个固定 schema 的离散动作**（导航、交互、结束任务等），再在环境里执行。VLM 的输出被当作纯文本解析，不是结构化的 `tool_calls` 返回。

### 1.2 LLM 封装层未启用 function calling

`mllm_base_agent/llm/provider.py` 中 `OpenAICompatibleChatModel.invoke()` 组装的请求体：

```python
payload: Dict[str, Any] = {
    "model": self.model,
    "messages": to_openai_messages(messages),
    "temperature": self.temperature,
    "max_tokens": self.max_tokens,
    "stream": False,
}
if self.top_p is not None:
    payload["top_p"] = self.top_p
if self.model_kwargs:
    payload.update(self.model_kwargs)
payload.update(kwargs)
```

没有传 `tools` / `functions` / `tool_choice`，也没有解析响应里的 `tool_calls` 字段。即使底层是标准 OpenAI 兼容接口（理论上支持 function calling），这层封装也完全没有用到这个特性。

### 1.3 `tools/` 目录是空壳

```python
# mllm_base_agent/tools/__init__.py
```

该目录只有一个空的 `__init__.py`，没有任何工具注册/执行逻辑，说明“工具调用框架”目前只是预留位置，尚未实现。

### 1.4 “深度预测”目前只是环境渲染开关，不是模型推理工具

配置里的 `render_depth`（如 `env.render_depth`）只是让 AI2-THOR/ProcTHOR 底层引擎在 `controller.step()` 时顺带渲染出深度图（`renderDepthImage=True`），属于环境模拟器自带的 ground-truth 渲染能力，不是“调用一个深度估计神经网络模型”。目前 agent 也没有把这个深度图喂给 VLM 或用它做决策——`_build_messages` 发给模型的只有 RGB 图像，代码库中也没有深度估计模型（如 MiDaS/DepthAnything）的调用。

### 1.5 “Lookahead”模式不是 tool calling

`runner.py` 里的 `lookahead_think_node` 会在决策前，先在环境里“预演”多个候选动作（拿到对应画面后再回滚），把候选画面一起发给 VLM 辅助其做选择。这本质上是“环境内的多步 rollout + 图像拼接”，仍由 Python 代码硬编码驱动，不是模型主动决定“我要调用哪个工具”。

### 1.6 小结表

| 能力 | 是否具备 |
|---|---|
| VLM 输出结构化文本动作（Move/Rotate/Pick/...） | ✅ 有（`actions/parser.py` 解析） |
| OpenAI function calling / tool_calls | ❌ 未启用 |
| Agent 自主选择调用外部工具（如深度估计、检索等） | ❌ 没有 |
| `tools/` 目录下注册的工具集 | ❌ 空 |
| 深度图 | 仅作为环境渲染选项存在，未接入决策流程，也非独立深度预测模型 |

---

## 二、RPent 如何实现真正的 Agentic Planner

`RPent`（路径：`worldmodel/RPent`）的核心区别在于：**把 LLM 变成一个"手握工具箱、自主决定下一步做什么"的智能体，而不是一个"每一步只吐出一个固定动作字符串"的单步策略**。

### 2.1 真正的 tool-calling loop（而非文本解析动作）

RPent 所有工具（无论哪个环境）都统一定义在自己的 `Toolkit` 抽象里（详见 2.2），工具的 schema、执行逻辑、结果格式完全由 RPent 自己掌控，**不依赖任何第三方框架自带的业务工具**。`rpent/planner/base.py::build_planner` 是统一工厂函数，支持三种 planner 后端，区别仅在于“谁来驱动多轮 tool-calling 循环”以及“工具以什么协议接入”：

| 后端 | 驱动方 | 工具接入方式 |
|---|---|---|
| `api` | `rpent/planner/api_loop.py`：自己实现的 `ApiAgentLoop`，基于 `pydantic-ai` 直连各家模型 API（Anthropic/OpenAI/自建兼容端点） | 把 `Toolkit.get_tools_spec()` 里的每个工具转成 `pydantic_ai.Tool`，调用时分发到 `toolkit.execute_tool()`，完全绕开 Claude Code / Codex |
| `claude_code` | `rpent/planner/claude_code.py`：官方 `claude_agent_sdk` | 把 `Toolkit` 每个工具包装成 SDK 的 **in-process MCP server**（`create_sdk_mcp_server`），命名空间 `mcp__rpent__<tool_name>`，通过 `mcp_servers` + `allowed_tools` 注入；`setting_sources=[]` 显式关闭用户/项目级 `.claude` 配置，只保留少量文件类内置工具（`Bash Read Write Glob Grep`）做辅助 |
| `codex` | `rpent/planner/codex.py`：`openai_codex` SDK | 在进程内起一个 **streamable-HTTP MCP server**（`rpent/planner/utils/http_mcp_server.py`，封装标准 MCP `Server` 的 `list_tools`/`call_tool`），再通过 `mcp_servers.rpent.url` 配置覆写把这个 HTTP 端点注册给 Codex CLI |

以 `api` 后端为例，`ApiAgentLoop` 基于 `pydantic-ai` 的 `Agent`，把每个工具注册为结构化 function-calling 定义（JSON Schema），驱动标准的 agentic loop：

```python
async with agent.iter(
    seed,
    message_history=history,
    usage_limits=UsageLimits(request_limit=max_turns + 1),
) as run:
    async for node in run:
        ...
        if Agent.is_call_tools_node(node):
            ...
            async with node.stream(run.ctx) as stream:
                async for event in stream:
                    if isinstance(event, FunctionToolCallEvent):
                        n_tool_calls += 1
                        ...
                        if event.part.tool_name == "finish":
                            finish_result = {...}
                    elif isinstance(event, FunctionToolResultEvent):
                        ...
```

模型每一轮可以自己决定调用哪个工具、调用几次、要不要先看图再决定下一步，而不是被迫在一次调用里输出唯一动作。循环直到模型显式调用 `finish` 工具，或用完 `max_turns` 预算——这是“多轮自主规划”而非“单步推理”的关键。

`claude_code` / `codex` 两种后端虽然把“何时调用工具、要不要多轮迭代”这部分规划能力交给了成熟的第三方 agent CLI，但**业务工具本身仍是 RPent 自己的**：两者都是把 `Toolkit.get_tools_spec()` 转成 MCP 协议的 `list_tools`/`call_tool` 实现，再由 `toolkit.execute_tool()` 统一执行，只是“传输通道”不同（前者是 SDK 内建的 in-process MCP server，后者是独立起的 HTTP MCP server）。

### 2.2 工具集分层：通用工具 + 环境专属工具（包含真正的“调用模型”工具）

`rpent/tools/toolkit.py` 的 `Toolkit` 基类只注册“文件读写 + finish”这类通用工具；具体环境的子类（如 `robots/libero/toolkit.py` 的 `LiberoToolkit`）在此基础上注册一整套领域工具：

- `move_to` / `rotate_wrist` / `rotate_pitch` / `move_pose`：脚本化的底层伺服控制（无需 VLM）
- `pi0_pick` / `pi0_doubled`：调用 **VLA（Vision-Language-Action）策略模型** 做闭环抓取/接触操作
- `segment`：调用**独立部署的 SAM3 分割模型服务**（`sam3_server.py`，通过 RPC）做视觉定位
- `back_project`：把分割出的像素结合深度图/相机内外参**反投影成世界坐标**
- `view_driver_state` / `view_camera_meta`：查看环境状态和相机标定

也就是说，这里的“深度/分割/VLA”确实是作为**可被 LLM 主动调用的工具**存在的（由 `Toolkit.execute_tool` 统一分发），而不是像 SpatialWorld 那样只是环境渲染时顺带产出的 ground-truth 深度图。

```python
def segment(self, prompt: str = "", camera: str = "agentview", ...) -> dict:
    """Call SAM3 on an existing image artifact without advancing the env."""
    ...
    data = self._sam3_client.segment(image_path, text_prompt=prompt if has_prompt else None, ...)
    ...
    world_result = _mask_to_world(mask, np.load(world_path))
```

### 2.3 系统提示词设计为“感知—定位—规划—执行—自我校验”工作流

`robots/libero/prompts/system.py` 的 prompt 明确要求模型：

1. 先读记忆库/历史成功案例（`READ MEMORY FIRST`）；
2. 做一次“强制感知扫描”，用 `segment`/`back_project` 把场景中所有目标物体和目的地都定位好，建成一张“实体-坐标”表（`MANDATORY PRE-TASK PERCEPTION PASS`）；
3. 逐步调用底层动作原语执行；
4. 每一步执行完都重新观察状态图像，判断是否符合预期（如 `Rule 1b`：用感知信号而非物体名字判断是否抓到了目标）；
5. 最后写审计报告 + 显式调用 `finish` 结束。

这套 prompt 把“规划”显式教给了模型：不是“看一帧图 → 吐一个动作”，而是“先建立世界模型 → 制定多步策略 → 分步验证 → 必要时用工具重新定位/纠错”。

### 2.4 `finish` 工具是显式终止信号

`rpent/tools/common.py` 中 `finish()` 返回 `{"_finish": True, ...}`；`api_loop.py` 检测到调用了名为 `finish` 的工具就结束循环——这是标准 agentic 系统的收尾方式，而不是依赖对动作字符串做正则解析来判断“DONE/FAIL”。

### 2.5 除 API 工具调用外，还支持"外壳级" agent 作为 planner 后端

`rpent/planner/codex.py` 和 `claude_code.py` 直接把整个多轮规划/工具调用循环委托给 Claude Code / Codex 这类成熟的编码 agent CLI，通过 MCP（Model Context Protocol）把 RPent 的工具暴露给它们（`rpent/planner/utils/http_mcp_server.py`），复用这些 CLI 自带的更强的长程规划能力（例如更好的错误恢复、上下文管理），但工具本身**没有使用**这两个 CLI 自带的业务工具集（详见 2.1 的对比表）。`rpent/planner/base.py::build_planner` 是统一的工厂函数，可按需切换 `api` / `claude_code` / `codex` 三种 planner 后端。

### 2.6 当前 RPent 的完整工具清单

RPent 工具分两层注册：`Toolkit` 基类自带的**通用工具**（任何环境都有）+ 各环境 `Toolkit` 子类追加注册的**环境专属工具**（当前仓库只有 LIBERO 一个环境实现）。

**通用工具**（`rpent/tools/common.py`）：

| 工具 | 作用 |
|---|---|
| `read_text_file` | 读取 UTF-8 文本文件（历史 recipe、audit 等），超长自动截断 |
| `write_text_file` | 写文本文件（自动建父目录），用于保存 recipe/audit |
| `list_dir` | 列出目录下文件（非递归），默认是当前 run 的输出目录 |
| `finish` | 终止 agent 循环的信号工具，返回 `{"_finish": True, "status", "summary"}` |

**LIBERO 环境专属工具**（`robots/libero/tools.py` + `robots/libero/toolkit.py`，共 12 个）：

感知/查看类（不推进环境状态）：

| 工具 | 作用 |
|---|---|
| `view_driver_state` | 读取某一步的机器人状态 + 场景/腕部相机图片路径（含高分辨率标定帧） |
| `view_camera_meta` | 读取相机标定元数据（内外参），用于反投影 |
| `segment` | 调用独立部署的 SAM3 分割模型服务，对已有图像做文本/点提示分割，并把 mask 反投影为世界坐标 |
| `back_project` | 把像素坐标 (row, col) 结合深度图/相机内外参反投影为世界坐标 XYZ，支持单点和区域模式 |

底层运动控制类（脚本化伺服，不调用 VLA/VLM）：

| 工具 | 作用 |
|---|---|
| `move_to` | 伺服末端执行器到指定世界坐标 XYZ（保持姿态） |
| `move_pose` | 同时伺服位置和姿态（pitch/yaw），适合狭窄空间/柜内穿入 |
| `rotate_wrist` | 绕世界 Z 轴旋转手腕（yaw） |
| `rotate_pitch` | 绕世界 X 轴旋转手腕（pitch），常用于伸入微波炉等狭窄开口前调姿 |
| `release` | 张开夹爪，直到 LIBERO 判定任务完成或步数用尽 |
| `set_gripper` | 保持当前位姿，驱动夹爪开合若干步 |

闭环策略调用类（调用 VLA 模型 Pi0.5）：

| 工具 | 作用 |
|---|---|
| `pi0_pick` | 调用 Pi0.5 做闭环抓取，依据 EEF 抬升 + 夹爪闭合信号判断成功 |
| `pi0_doubled` | 调用 Pi0.5 做非抓取类接触技能（如开关旋钮/按钮/推动） |

即通用 4 个 + LIBERO 12 个，共 **16 个**可被 LLM 调用的工具。另外 `LiberoPrimitives` 里还定义了 `place`、`get_privileged_state`、`run_full_task` 等方法，但**未被注册进 `TOOLS_SPEC`**，目前不对 LLM 暴露（推测是历史遗留 / 内部调试 / baseline 对比用）。

此外，`api` 后端在 `api_loop.py` 里额外附加了一个 `read_image` 工具（把某个本地图片路径转成多模态图像内容喂给模型），它不通过 `Toolkit` 注册，只存在于 `api` 这一种后端。

> 注：`rpent/planner/api_loop.py`、`claude_code.py`、`codex.py` 在本次分析后一度出现在工作区 `deleted_files` 列表中；以上关于三种 planner 后端及工具接入方式的描述基于分析时读取到的文件内容，如需复核请以 RPent 仓库当时的版本或 git 历史为准。

---

## 三、对比总结

| 维度 | SpatialWorld (`mllm_base_agent`) | RPent |
|---|---|---|
| 决策粒度 | 每次 VLM 调用只产出**一个**离散动作字符串（Move/Rotate/Pick/…），由规则解析 | 每轮 VLM 调用可自主选择调用**任意工具**，包括感知工具、控制工具，多轮组合完成任务 |
| 工具能力 | 无（`tools/__init__.py` 为空壳） | 通用文件工具 + 环境专属工具（含真实的 SAM3 分割模型、VLA 策略模型调用、几何反投影计算） |
| LLM 调用方式 | 纯 chat completions，`payload` 无 `tools` 字段 | 用 `pydantic-ai` 的 `Agent` + `Tool`，走标准 function calling |
| 感知与决策解耦 | 无——图像直接喂给 VLM 做端到端决策 | 有——模型可先调用 `segment`/`back_project` 主动“看清楚”目标在哪，再决定下一步物理动作 |
| 终止信号 | 靠解析文本里的 `EndTask(DONE/FAIL)` | 模型显式调用 `finish` 工具（结构化） |
| 多步规划支持 | Lookahead 模式只是硬编码地在环境里"预演"固定候选动作集合再一次性决策，不是模型自主选择工具/多轮迭代 | 模型完全自主决定要不要重新感知、重新定位、重试抓取、更换策略，直到自己判断任务完成 |

---

## 四、若要迁移到 SpatialWorld，建议的落地路径

1. **给 LLM 封装层加 tool-calling 支持**：在 `mllm_base_agent/llm/provider.py` 的 `OpenAICompatibleChatModel.invoke()` 中加入 `tools`/`tool_choice` 参数，并解析响应里的 `tool_calls` 字段（目前完全没有传，这是最基础的一步）。
2. **建立真正的工具注册表**：参考 RPent 的 `Toolkit` 抽象，在 `mllm_base_agent/tools/` 里实现一个工具容器（注册 + 分发），把 `env.step_with_action_dict` 包装成一个（或多个）工具，再加上诸如“分割定位”“反投影”“查看历史状态”等辅助感知工具。
3. **改造主循环**：把 `runner.py` 的 Think→Act→Evaluate 硬编码循环，替换/扩展为工具调用循环——模型每轮可以选择调用感知工具（不推进环境）或动作工具（推进环境），直到调用 `finish`/`EndTask` 类工具。
4. **重写 system prompt**：像 RPent 一样明确教模型“先感知定位、建表格、再规划执行、执行后自检”的工作流程，而不是要求它一步只想一个动作。

---

## 五、参考 RPent 的 Harness 思路：SpatialWorld 测试环境中哪些内容适合沉淀为 Harness

RPent 的 `robots/libero/` 并不只是"一套工具"，而是把 **环境契约 + 工具层 + 分层 Prompt + 长文档指南 + 记忆库 + 审计协议 + 编号 Rule + 只读技术参考语料** 这几层正交的资产组织在一起，统一"喂"给 agent，让它不必每次都从零摸索。核心思想是：**把每一次失败复盘 / 成功经验都变成可检索、可复用、可版本化的文本资产，而不是只让模型临场发挥**；同时用工具把"确定性子任务"（几何计算、控制伺服）从 LLM 决策中剥离出来。

RPent 的分层结构与对应文件：

| 层 | RPent 对应文件 | 作用 |
|---|---|---|
| 环境契约 | `rpent/envs/env_spec.py`（`EnvSpec`） | 静态描述一个环境需要暴露什么（prompt、CLI 参数解析、运行时初始化钩子），让 runner 保持环境无关 |
| 工具层 | `rpent/tools/toolkit.py` + `robots/libero/tools.py` | 通用工具（文件读写、`finish`）+ 环境专属工具（感知/控制/模型调用），统一 schema + 分发 |
| Prompt 分层 | `context/prompt_utils.py`（渲染引擎）+ `robots/libero/prompts/{system,user}.py` | system prompt 用 Python 数据结构（`BulletList`/`Numbered`/dict）拼装而非裸字符串，可组合、可测试 |
| 长文档指南（guides） | `robots/libero/guides/*.md` | 分层：`strict_hybrid_guide.md`（协议层，跨任务通用）→ `pro_hybrid_guide.md`（某个 benchmark 变体的特化）→ `env_calibration.md`（数值参数表） |
| 记忆库（memory） | `resources/libero/memory/MEMORY.md` + `feedback_*.md`/`project_*.md` | 每条失败教训/成功经验单独归档成文件，索引文件给一行摘要，agent 按需检索、精读 |
| 审计/终止协议 | `finish` 工具 + audit JSON schema | 显式结构化终止信号，而非正则解析文本；audit 里强制要求写"如何定位/为什么失败"，逼迫模型自我复盘 |
| Rules 体系 | system prompt 里编号的 Rule 0/1/1b/2/2b/2c/2d/4/5/6/7 | 把"踩过的坑"提炼成编号规则，每条规则配一个反例场景，而非笼统的"注意安全" |
| 技术参考语料 | `resources/libero/results_*_pert/`（seed-0 recipe） | 只读的成功轨迹语料库，供模型"抄技巧不抄坐标"，参考不复制 |

### 5.1 SpatialWorld 现状对照

结合真实失败日志（如 `scripts/procthor/benchmark_outputs/.../procthor208_attempt_1_failed.log`：agent 在同一位置被 `StandardIslandHeight2`/`StandardWallSize` 卡住后，反复 `RotateLeft→RotateRight→MoveBack` 打转 74 步未破局）和现状分析文档（`docs/single_agent_experiments_summary.md`、`docs/multi_agent_collaboration_analysis.md`），SpatialWorld 当前：

- 只有一份**扁平的 system prompt**（`mllm_base_agent/prompts/ai2thor.py` 等），列了动作清单和几条笼统规则，没有分层、没有编号 Rule、没有反例场景。
- 感知（如深度图）只是环境渲染副产品（`env.render_depth`），没有被封装成可主动调用的工具。
- **没有记忆库**：`mllm_base_agent/dual_agent/ai2thor/core/memory/__init__.py` 的 `DualAgentMemoryBuffer` 只是一个进程内滑动窗口 buffer 类，不落盘、不跨 run 复用，没有任何"失败教训/成功案例"文本沉淀——每次运行都从零摸索，相同的坑（如"卡墙后应该怎么绕"）在几百个 task 里反复踩。
- **没有结构化终止/审计协议**：靠正则解析 `<ACTION>DONE</ACTION>`，不强制模型写"我是如何验证的"。
- `docs/*.md` 目前是"事后数据分析报告"，不是"运行前喂给 agent 的指导文件"，两者尚未打通。

### 5.2 建议整理成 Harness 的内容（按优先级）

**1. 环境执行约束表（数值参数 / 物理规则），对应 `env_calibration.md`**

从已跑的失败日志和评估代码里提炼成表格常识，例如：
- 交互距离阈值（`1.0m`，已在 prompt 里但很单薄）；
- 碰撞行为目前只写了"被挡住时环境不模拟弹开、原地零位移"，但**没有配套的"检测到卡墙后怎么办"的具体处理流程**（RPent 会给出具体的恢复算法，而不是一句"请重新规划"）；
- 把 AI2-THOR/ProcTHOR 常见错误消息（如 `"X is blocking Agent 0 from moving by (...)"`）到"应对策略"整理成映射表——这正是当前 `RotateLeft→RotateRight→MoveBack` 死循环缺的东西。

**2. "失败模式 → 应对 Rule"编号清单，对应 `strict_hybrid_guide.md` 的 Rule 0-7**

把 `docs/single_agent_experiments_summary.md`、`docs/multi_agent_collaboration_analysis.md` 里已经统计出的高频失败模式，反向转成 prompt 里的强制 Rule，例如：
- Rule：连续 2 次同方向移动失败 → 禁止再次尝试同一动作，必须换方向或减小步长（现有 prompt 只说"不要盲目重试"，没有量化"几次算盲目"和"换哪个方向"的具体算法）；
- Rule：模型输出 `DONE` 前必须描述"验证证据"（对应现状总结里"DONE 但条件未过"占比很高的问题）；
- Rule：`FAIL` 前必须证明"已系统性探索过整个场景"，避免过早放弃。

**3. 记忆库（Memory / Feedback 语料），对应 `resources/*/memory/MEMORY.md`**

这是当前最大的空白。建议：
- 把每次 benchmark 跑出来的**高价值失败案例**（如"在岛台前打转 74 步未破局"）人工/半自动提炼成一条条独立的经验条目（"遇到 xxx 挡路 → 应该 yyy"），存成文件，配一个索引文件；
- 在 system prompt 或运行前，把相关经验注入 context（不需要真的做检索工具，先做"静态注入 K 条最相关经验"这种轻量版本）。

**4. 结构化"感知—定位—验证"工作流骨架，对应 `PERCEPTION_ALGORITHM` / Mandatory pre-task perception pass**

即使 SpatialWorld 暂时不引入工具调用，也可以在 prompt 里显式要求模型：先建一张"任务相关实体清单"（目标物体、容器、路径障碍），再规划，再逐步验证——而不是像现在这样"看一帧图直接吐一个动作"。这是 RPent 反复强调的"错误的目标识别不可恢复，所以要在动作前先建表"的经验，直接适用于 AI2-THOR/ProcTHOR 的物体查找/操作任务。

**5. 显式终止 + 审计 schema，对应 `finish` 工具与 audit JSON**

不必引入真正的工具调用框架，可以先在现有的文本协议里，要求 `DONE`/`FAIL` 必须搭配一段结构化"审计说明"（做了什么验证、依据是什么），并把这段说明持久化到 episode 日志里，方便后续复盘筛出"高质量失败/成功案例"反哺记忆库（与第 3 点形成数据闭环）。

**6.（更长期）真正的工具层**

如第四节已提出的路径：加 `tools/` 注册表、给 LLM 封装层加 `tools` 参数、把"预演候选动作"这种目前硬编码在 `runner.py` 里的能力，逐步暴露成可被模型自主调用的工具（例如"查看某方向是否可通行"这种轻量感知工具），而不是像 lookahead 模式那样每步都无差别全量预演。

### 5.3 小结

| Harness 内容 | RPent 对应资产 | SpatialWorld 现状 | 落地成本 |
|---|---|---|---|
| 环境约束/错误码映射表 | `env_calibration.md` | 仅几句笼统描述 | 低——纯文档整理 |
| 编号 Rule + 反例 | system prompt 里的 Rule 0-7 | 零散的"注意事项" | 低——重写 prompt 章节 |
| 记忆库（失败/成功案例） | `resources/*/memory/MEMORY.md` | 空（仅有未落盘的滑动窗口 buffer） | 中——需要人工/半自动提炼 + 索引维护 |
| 感知-定位-验证工作流 | `PERCEPTION_ALGORITHM` | 无，单帧直接决策 | 中——prompt 改造，不依赖工具调用 |
| 终止+审计协议 | `finish` 工具 + audit JSON | 仅 `DONE`/`FAIL` 文本标签 | 低——扩展现有输出格式 + 落盘逻辑 |
| 工具层 | `Toolkit` + 12 个 LIBERO 工具 | 空壳 | 高——需要改造 LLM 封装层和主循环 |
