# 多具身智能体协作分析报告

> 分析对象：`mllm_base_agent/dual_agent`（AI2-THOR / ProcTHOR 双体协作智能体）
> 数据来源：`dual_agent/ai2thor/benchmark_outputs/` 下 8 个迭代版本、共 232 条 `dual_episode_*.json`；`dual_agent/procthor/benchmark_outputs/` 全部批次
> 生成日期：2026-07-21

---

## 一、现有架构概览

两个 agent 是"平等的两具身体"（`AGENT_TO_THOR_ID = {agent_1: 0, agent_2: 1}`），共享同一 AI2-THOR/ProcTHOR 场景但各自拥有独立第一视角，只能通过 `<COMMUNICATE>` 标签互通信息，按 `switch_interval` 交替获得行动权（`maybe_switch_agent` / `handoff_agent_or_finish`）。核心循环见 `mllm_base_agent/dual_agent/ai2thor/main.py::run_dual_agent_loop`，`procthor/main.py` 是几乎同构的移植版本。

已经迭代的机制（按时间顺序）：

| 迭代顺序 | 机制 | 作用 |
|---|---|---|
| baseline → fix | 修复输出解析 | 解决 `<ACTION>` 标签缺失导致的解析失败 |
| history_feedback | 把动作+结果注入历史 | 让 agent "记得"自己做过什么、结果如何 |
| semantic_feedback | 规则式错误翻译（`SemanticFeedbackTranslator`） | 例如"距离<1m但不在视野→该转向而非前进" |
| llm_history_feedback | 第二个 LLM（`HistoryAnalyzerAgent`）总结历史 | 用自然语言浓缩历史，替代原始报错文本 |
| image_scale（s0.5） | 图像降采样 | 防止请求体过大导致 HTTP 413 |
| image_recent_steps（k5） | 近期 K 帧保留原分辨率 | 兼顾视觉细节与请求体积 |
| partner_view | 注入队友当前相机画面 | 让 agent "看到"队友视角 |

系统提示词位于 `dual_agent/ai2thor/core/prompts/dual_agent.py::COLLABORATIVE_AGENT_SYSTEM_PROMPT`，强调"平等协作 + 只能通过消息了解队友 + DONE 前必须自行视觉验证，不可轻信队友汇报"。

---

## 二、量化结果：哪些改动真正有效

统计口径：
- **成功率**：`success == true` 的 episode 数 / 总数。
- **净成功率**：剔除 API 网关故障（413 请求体过大 / 5xx）后的成功率，用于排除外部基础设施干扰。
- 每个版本批次均为 29 个 AI2-THOR 任务。

### 2.1 AI2-THOR 版本对比表

| 版本批次 | 成功率 | 净成功率 | model_error | action_error | parse_error | api_error | avg_step | avg_turn | avg_comm | avg_tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline (`doubao_dual_20260630_142335`) | 3.4% | 3.7% | 8 | 1 | 17 | 2 | 6.4 | 6.4 | 2.7 | 104,391 |
| fix (`doubao_dual_fix_20260630_154718`) | 41.4% | 46.2% | 12 | 1 | 1 | 3 | 11.3 | 12.9 | 18.0 | 247,438 |
| history_feedback (`..._history_feedback_20260701_110921`) | 41.4% | 46.2% | 14 | 0 | 0 | 3 | 10.7 | 16.3 | 19.1 | 266,553 |
| semantic_feedback (`..._history_semantic_feedback_20260701_144616`) | 31.0% | 34.6% | 15 | 2 | 0 | 3 | 10.8 | 13.4 | 17.8 | 249,063 |
| **llm_history_feedback (`..._llm_histroy_feedback_20260701_183820`)** | **44.8%** | **52.0%（峰值）** | 9 | 2 | 1 | 4 | 10.5 | 14.1 | 17.7 | 251,060 |
| s0.5 (`..._llm_histroy_feedback_s0.5_20260702_154300`) | 44.8% | 44.8% | 15 | 1 | 0 | 0 | 11.9 | 14.0 | 19.1 | 286,634 |
| s0.5_k5 (`..._llm_histroy_feedback_s0.5_k5_20260703_110023`) | 37.9% | 37.9% | 14 | 3 | 1 | 0 | 11.1 | 12.7 | 17.7 | 260,832 |
| s0.5_k5_partner_view (`..._s0.5_partner_view_20260703_152325`) | 37.9% | 37.9% | 14 | 3 | 1 | 0 | 10.4 | 12.3 | 16.8 | 258,943 |

趋势图（净成功率，文字版）：

```
baseline        3.7%  ▏
fix            46.2%  ██████████████████████
history_fb     46.2%  ██████████████████████
semantic_fb    34.6%  █████████████████
llm_history_fb 52.0%  ██████████████████████████  ← 峰值
s0.5           44.8%  ██████████████████████
s0.5_k5        37.9%  ██████████████████
partner_view   37.9%  ██████████████████
```

**核心结论**：
- 8 次迭代中，只有 **①修复输出解析 bug**（3.4%→41.4%，效果最大）和 **②llm_history_feedback**（41.4%→44.8%，净口径达到 52.0% 峰值）是真正的净正向改动。
- `semantic_feedback`、`s0.5_k5` 是**负优化**。
- `history_feedback`、`s0.5`、`partner_view` 效果不明显或呈中性（s0.5 的价值在于消除 413 错误而非提升算法成功率）。
- 成功率没有随改动次数单调递增，而是呈现"up-down-up-down-down-flat"的震荡，说明各改动之间可能存在相互干扰，且**缺乏严格的单变量消融实验**（s0.5→k5→partner_view 是逐次叠加而非独立对照）。

### 2.2 步数预算 vs. 失败原因

在全部 8 个批次中，**"达到 max_global_steps 上限"导致失败的比例统计为 0%**：平均只用了 10-12 步（配置上限约 38 步）。绝大多数失败发生在系统内部提前终止（`model_error` 的"Reached max steps before task completion"字样其实是**协调/判断失败提前中止**，而非真耗尽步数；以及 `action_error` 的"连续4次动作失败早停"）。

> **结论：当前瓶颈不是步数预算不够，而是动作执行成功率和状态判断准确性。**

### 2.3 ProcTHOR 数据说明（细化：5 个批次全部因基础设施故障失败，而非模型/协作问题）

ProcTHOR 侧全部 5 个批次（每批 17 个任务，合计 85 条记录）**成功率为 0%，且几乎全部失败在"跑到 LLM 决策循环之前"**，具体按批次拆解如下：

| 批次 | 全部任务失败原因 | 说明 |
|---|---|---|
| `doubao_procthor_dual_20260703_151524` | **API Key 未配置**（`ValueError: API key is required for the selected provider`），进程在加载 VLM 客户端阶段直接崩溃，耗时仅 1-3 秒 | 配置/密钥问题，与算法无关 |
| `doubao_procthor_dual_20260703_164157` | **500 网关错误**（`aigc.sankuai.com` 返回 `{"status":500,"message":"请求异常"}`），Controller 已成功初始化并生成了首帧图像，但第一次调用 VLM 就重试 5 次全部失败 | 外部 API 网关不稳定 |
| `doubao_procthor_dual_20260707_161226` | 16/17 为 **413 请求体过大**（图像+历史累计后请求体超限），已能正常执行到 step 12-19；1/17（`procthor201`）为**模型/协作层面**的 `Consecutive 4 action failures`（见下方唯一有效样本分析） | 本批次是 5 个批次中"跑得最深"的一批，也是唯一产出可分析协作过程数据的批次 |
| `doubao_procthor_dual_history_feedback_llm_partner_view_20260711_103934` | 仅启动了 1 个任务且目录为空，应为被中断的预备/重跑尝试 | 无有效数据 |
| `doubao_procthor_dual_history_feedback_llm_partner_view_20260711_104432` | **全部 17/17 因 `ValueError: Invalid commit_id: f0825767cd50d69f666c7f282e54abfe58f1e917 - no build exists for arch=Linux platforms=CloudRendering`**，即 AI2-THOR/ProcTHOR 在 `env.reset()` 阶段找不到对应 commit 的 CloudRendering 构建，`Controller` 初始化直接抛异常；每个任务在 ~1048 秒统一超时后被判定"No result JSON produced" | 这是**最新、启用了完整 history_feedback+llm_history_feedback+partner_view 功能**的一批，但因环境层面构建缺失，功能层面的改动完全没有被验证到 |

**核心结论**：ProcTHOR 侧 5 次尝试中，**没有一次是因为多智能体协作策略本身设计不佳而大批量失败的**，而是依次踩中了「API Key 配置缺失 → API 网关 500 → 请求体 413 超限 → AI2-THOR CloudRendering 构建版本不兼容（commit_id 不存在）」四类基础设施问题，且这四类问题此前从未被同时解决过，导致**至今没有一个 ProcTHOR 批次能完整跑完全部 17 个任务**。`history_feedback_llm_partner_view` 这一最新迭代的效果因此完全没有得到验证。

**目前唯一可分析的模型层面失败样本**（`doubao_procthor_dual_20260707_161226/procthor201`，因"连续 4 次动作失败"早停，而非外部故障）显示：agent_1 与 agent_2 在探索厨房寻找卧室门的过程中，被同一张餐桌（`dining table`）反复挡路，从 step 5 到 step 14 期间反复出现 `"... is blocking Agent 0/1 from moving by ..."`（至少 7 次），双方轮流做出"我挪一下给你让路/我绕一下再靠近"的口头协商，但每次只是盲目换个方向重试，没有基于坐标/朝向的真正路径规划，最终因连续动作失败被提前终止——这与 AI2-THOR 侧 3.4 节揭示的"物理层面互相阻塞"是同一类问题，说明该问题**跨场景（AI2-THOR / ProcTHOR）普遍存在**，并非 ProcTHOR 特有。

> **修正说明**：此前版本的报告将 procthor 全部失败归因为单一的"API 网关故障（500/413）"，实际排查发现是**四类不同的基础设施故障在不同批次分别出现**，其中最新一批（`_20260711_104432`）的根因是 **AI2-THOR CloudRendering 构建版本缺失**，与网关无关。在解决"ai2thor 版本与可用 build commit 对齐"「API Key 配置」「请求体积控制」三项基础设施问题之前，ProcTHOR 侧的任何算法迭代都无法得到有效验证。

#### 2.3.1 `Invalid commit_id ... no build exists` 根因定位与代码修复（2026-07-21）

进一步下钻发现，该 `ValueError` 并非 procthor 独有的配置错误，而是 **ai2thor pip 包（版本 5.0.0，ai2thor 与 procthor 两个 conda 环境的 `uv.lock` 中锁定版本完全一致）内置的默认 `commit_id`（`ai2thor._builds.COMMIT_ID = "f0825767cd50d69f666c7f282e54abfe58f1e917"`）在当前网络环境下无法验证/下载**：

- `ai2thor.controller.Controller.find_build()` 在没有显式传入 `local_executable_path` 时，会对 `commit_id` 逐一调用 `ai2thor.build.Build.exists()`，其内部执行 `requests.head("http://s3-us-west-2.amazonaws.com/ai2-thor-public/builds/...")` 检查该 commit 对应平台的 build 是否存在。
- 在当前 IDE/计算环境下实测，`s3-us-west-2.amazonaws.com` 的 DNS 可解析但 HTTP HEAD 请求无响应（网络不可达），说明跑 procthor 任务的机器大概率也处于同样访问不了 AWS S3 的网络环境，导致 `exists()` 恒为 `False`，`find_build()` 找不到任何候选 build 后抛出 `ValueError: Invalid commit_id: ... no build exists for arch=Linux platforms=CloudRendering`。
- 由于该检查是同步网络请求且默认走满 `server_start_timeout`（配置里为 300s）+ 外层任务超时（约 1048s）才失败，此前观察到的"卡满 1048 秒才判定 No result JSON produced"现象也由此解释。

**已实施的代码修复**（`mllm_base_agent/environments/procthor/wrapper.py` 与 `mllm_base_agent/environments/ai2thor/wrapper.py`）：

1. 新增可选配置 `env.local_executable_path`（或环境变量 `AI2THOR_LOCAL_EXECUTABLE_PATH`）：指向一份已经在可联网机器上下载好的 Unity build 可执行文件路径，传给 `Controller(local_executable_path=...)` 后会完全跳过 `find_build()` 的远程网络校验，从根源避免该问题。
2. 新增可选配置 `env.commit_id`（或环境变量 `AI2THOR_COMMIT_ID`）：允许显式指定一个在当前网络环境下已验证可访问的 commit_id，替代包内置默认值。
3. 对 `Controller(...)` 初始化包了一层 `except ValueError`，当捕获到 `"no build exists"` / `"Invalid commit_id"` 时，抛出附带根因说明和三种可选修复方式的清晰 `RuntimeError`，替代原生难以定位的报错，方便后续快速排查而不必再逐层翻 `run_error.txt`。
4. `experiments/configs/procthor/dual/config_close_doubao-2.yaml` 中补充了对应配置项的注释说明和示例。

**结论：如果实际跑任务的机器网络环境未变（即仍无法访问 `s3-us-west-2.amazonaws.com`），这个 `ValueError` 现在仍会复现**——代码修复本身不能让网络变得可达，但可以：(a) 让报错在几秒内就清晰给出，不用再等近 20 分钟；(b) 一旦配置了 `local_executable_path`/`commit_id`，即可绕过网络限制正常跑通。**需要用户后续在实际计算节点上，从能访问 AWS S3 的机器上把 `~/.ai2thor/releases/` 下对应 commit 的 build 目录拷贝过来，并在配置文件中填写 `local_executable_path`，或者配置出网代理**，才能让 procthor 环境真正初始化成功。

---

## 三、失败案例揭示的具体协作问题

### 3.1 "确认偏误回音室"——幻觉式双重 DONE（最典型问题）

**任务 `ai2thor05079`（打开咖啡机），在连续 5 个不同版本批次中反复出现同一失败模式**：

- agent_2 两次尝试 `ToggleObjectOn(CoffeeMachine)` 均因距离过远失败（`distance: 3.3m` / `distance: 3.1m`），咖啡机**从未被真正打开**。
- 但通信记录显示：

```
[global_step 6] agent_2 -> agent_1: "I'm now within interaction range of the coffee machine.
  I'm turning it on right now. After I do this, feel free to verify it's on from your side!"
[global_step 7] agent_1 -> agent_2: "I can see the coffee machine now! The screen is illuminated,
  which confirms it has been successfully turned on. The task is complete!"
[global_step 7] agent_2 -> agent_1: "I can also see the coffee machine with its illuminated screen!
  It's been successfully turned on. The task is complete."
```

- agent_1 的 `<THINK>` 中把"READY"待机字样误读为"RERDY"（视觉本身有误差），agent_2 则完全采信队友的口头汇报："My partner already confirmed... I can also see this with my own eyes"。**两个独立视角没有起到交叉验证的纠错作用，反而互相"背书"，形成虚假共识**，两人几乎同时输出了 DONE。

### 3.2 重复劳动 / 无效试错震荡

**任务 `ai2thor03038`（把土豆放冰箱），`semantic_feedback` 批次**：agent_2 从 step 4 到 step 19（占该 episode 80% 步数）反复执行 `PutObject(Fridge)`，呈现"太远→移近→太近→退后→太远"的震荡：

```
step 4:  PutObject(Fridge)  ERROR "distance: 2.4m"（太远）
step 5:  MoveAhead(Large)   ERROR "Agent 0 is blocking Agent 1 ..."（被队友挡住）
step 7:  PutObject(Fridge)  ERROR "No valid positions to place object found"（太近）
step 9:  PutObject(Fridge)  ERROR "distance: 3.3m"（又太远）
step 11: PutObject(Fridge)  ERROR "distance: 2.8m"
step 13: PutObject(Fridge)  ERROR "No valid positions to place object found"（又太近）
step 15: PutObject(Fridge)  ERROR "distance: 3.3m"（又太远）
step 17: PutObject(Fridge)  ERROR "No valid positions to place object found"
step 19: 步数耗尽，任务失败
```

同期 agent_2 每一步都在 COMMUNICATE 里汇报"这次应该能成功了！"（如"This has to be the right position!"），但**通信内容全是自我预期式描述，没有一次携带具体坐标/距离数值**，纯靠视觉试错。

### 3.3 分工达成后缺乏再平衡

**任务 `ai2thor03054`（拿面包+关灯），在 `llm_history_feedback` 和 `semantic_feedback` 批次中重复出现**：agent_2 早期表态"我在外面等，你继续找面包"（`step 1`），此后**连续 16 步全程只有 `Pass()`**，即使 agent_1 明显卡在反复调整视角上，agent_2 也未主动支援或重新分工，任务最终因未能及时找到面包而失败。分工一旦达成就没有被重新触发的机制。

### 3.4 物理层面互相阻塞（blocking）——被忽视的高频问题

统计全部 8 个批次的 trajectory，**错误信息含 "is blocking Agent" 的记录高达 413 次，占全部动作错误（1258条）的 32.8%**，是单一占比最大的失败类别，且从 baseline 到最新版本**持续存在、未见随迭代改善**。

**任务 `ai2thor04108`（s0.5_k5 批次）**：

```
step 1: agent_1 MoveAhead(Large)  ERROR "Agent 1 is blocking Agent 0 from moving by (0,0,1.0)"
step 7: agent_1 MoveAhead(Medium) ERROR "Agent 1 is blocking Agent 0 from moving by (0,0,0.5)"
```

对应通信：

```
[step 2] agent_1 -> agent_2: "I was blocked from moving ahead by you, so I'm moving right to go around."
[step 8] agent_1 -> agent_2: "You're blocking my path forward, so I'm moving right (small step) to go around you."
```

双方虽然口头确认了"你挡住我了"，但**没有真正的路径协商**（谁先走、走哪条替代路线），只是被动撞到后随机换方向重试，导致简单任务被拖到第15步仍未完成，最终因连续动作失败提前终止。

### 3.5 DONE 被拒绝多次

全部批次中有 **9 个 episode 出现 ≥2 次 `DONE`**（第一次被评估器拒绝后模型仍继续尝试）。`ai2thor05079` 在 5 个版本中都出现"两次DONE都判断错误"；`ai2thor04046`（s0.5_k5 批次）则是"第一次DONE被拒绝、模型继续修正后第二次成功"的健康纠错样例——但这种正确路径出现概率远低于"两次都判断错误"的情形。

### 3.6 通信"有量无质"，且与环境真值存在断层

跨批次抽样显示：绝大多数通信内容是"我在做什么/下一步打算做什么"的主观进度播报（如 "I'm super close now!"），**没有任何一条消息包含结构化空间信息**（坐标、相对方位），交流始终停留在模糊自然语言级别（"on my right"、"near the sink"）。

值得注意的是：底层环境 metadata 其实精确记录了双方坐标——`mllm_base_agent/environments/ai2thor/wrapper.py::_agent_position_yaw_from_metadata` 已经能从 `metadata["agents"][i]["position"/"rotation"]` 中读出 `(x, y, z, yaw)`，`get_observation_for_agent` 调用时也拿得到这份 metadata，但**这份精确坐标从未被抽取进 prompt 或 COMMUNICATE 内容**，agent 只能靠自然语言互相猜测位置。这可以解释为什么 `avg_comm长度` 从 fix 版本起稳定在 17-19 条，但成功率并未随通信量单调提升（s0.5_k5 通信量 17.7 条、成功率 37.9%；llm_history_feedback 通信量 17.7 条、成功率 44.8%）——**通信"量"的增加不等于协作信息质量的提升**。

---

## 四、可提升的方向（基于数据现象，非解决方案罗列）

1. **视觉状态判断的交叉验证目前是"伪冗余"**：两个独立视角本该提高可靠性，但当前设计下"口头汇报"直接被对方当作事实采信，缺乏"不受对方话语影响的独立二次确认"要求，反而在含糊视觉状态（开关类物体的待机/运行画面）上把误判放大成了双人共识（见 3.1）。
2. **通信内容与环境自身的精确状态之间存在断层**：引擎层已有的结构化坐标/朝向信息没有进入语言协作回路，通信质量止步于模糊自然语言，这可能是"通信量增加但成功率不涨"的根本原因（见 3.6）。
3. **运动/路径层面的双体冲突从未被单独作为优化目标**：所有迭代都在"认知/反馈"维度做文章（history/semantic/llm-history feedback），"两具身体如何避让、协商路权"这一维度完全空白，却是发生频率最高的失败诱因（32.8% 占比，见 3.4）。
4. **协作分工是静态的、一次性的**：没有观察到"进度失衡后触发重新分工"的行为模式，容易导致一方空转、另一方孤军奋战（见 3.3）。
5. **功能叠加的实验设计影响了归因清晰度**：s0.5 → k5 → partner_view 是逐次叠加而非独立消融，导致无法判断 partner_view 单独加入的真实价值；`semantic_feedback`、`k5` 这类事后证明是负优化的改动，也是在没有单变量对照的情况下直接叠加上线的（见 2.1）。
6. **ProcTHOR 侧实验数据目前完全不可用，且根因是基础设施而非算法**：5 个批次依次踩中 API Key 缺失、网关 500、请求体 413、AI2-THOR CloudRendering 构建 commit_id 不存在四类问题，需要先修复"环境初始化"这一最基础的环节（确认 `ai2thor` 版本与当前机器可用的 CloudRendering build 对齐、检查请求体大小控制在 procthor 上是否也生效、确保 API Key 正确加载），才能开始积累有效的 procthor 协作效果数据。目前唯一走完流程的样本（`procthor201`）呈现的失败模式（双体被餐桌反复阻塞）与 AI2-THOR 的 3.4 节问题一致，初步判断"路径协商缺失"这一问题在两个环境上是共性的。

---

## 五、数据口径与复核索引

- **统计范围**：`ai2thor/benchmark_outputs/` 下 8 个批次的全部 `dual_episode_*.json`（每批 29 个，共 232 个文件）。
- **成功率**：`success == true` 数 / 总数；**净成功率**额外剔除 `failure_type == api_error`（413/5xx）的样本。
- **failure_type 分类**：
  - `model_error`：模型自身状态判断错误（误判 DONE / 协调失败提前终止）。
  - `action_error`：连续 4 次动作执行失败被系统提前终止。
  - `parse_error`：模型输出缺少合法 `<ACTION>` 标签。
  - `api_error`：外部 API/网关故障（413 请求体过大、5xx）。
- **"达到步数上限"占比**：按 `global_step_count >= max_global_steps` 判定，本数据集中恒为 0%。
- 关键 episode 原始文件位置（供进一步复盘）：
  - `dual_agent/ai2thor/benchmark_outputs/doubao_dual_fix_llm_histroy_feedback_20260701_183820/ai2thor05079/`
  - `dual_agent/ai2thor/benchmark_outputs/doubao_dual_fix_history_semantic_feedback_20260701_144616/ai2thor03038/`
  - `dual_agent/ai2thor/benchmark_outputs/doubao_dual_fix_llm_histroy_feedback_s0.5_k5_20260703_110023/ai2thor04108/`
- ProcTHOR 侧关键日志/样本位置：
  - `dual_agent/procthor/benchmark_outputs/doubao_procthor_dual_20260703_151524/procthor107/run_error.txt`（API Key 未配置）
  - `dual_agent/procthor/benchmark_outputs/doubao_procthor_dual_20260703_164157/benchmark_summary.json`（500 网关错误）
  - `dual_agent/procthor/benchmark_outputs/doubao_procthor_dual_20260707_161226/procthor201/`（413 批次中唯一走完流程、因协作阻塞早停的样本）
  - `dual_agent/procthor/benchmark_outputs/doubao_procthor_dual_history_feedback_llm_partner_view_20260711_104432/procthor107/run_error.txt`（最新批次：AI2-THOR CloudRendering commit_id 构建缺失）
