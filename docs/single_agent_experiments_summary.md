# Single-Agent 实验总结（ProcTHOR / AI2-THOR）

统计时间：2026-07-28。范围：`scripts/procthor/benchmark_outputs/` 与
`scripts/ai2thor/benchmark_outputs/` 下目前已产生结果的全部 single-agent
benchmark 运行（对应 `hope/single_benchmark_procthor.hope` /
`hope/single_benchmark_ai2thor.hope` 提交的分布式任务，底层由
`hope/launch_single_benchmark.sh` → `scripts/{procthor,ai2thor}/run_benchmark.py`
→ `mllm_base_agent/agent/runner.py` 驱动）。统计口径：以每个 task 输出目录下
的 `episode_*.json` 为准，字段 `success` / `fail_reason` / `step_count` 直接来自
agent 运行时的最终判定（`evaluate_node`），未做任何二次修正。

## 一、结论先行

1. **目前所有已完成的 ProcTHOR single-agent 实验成功率均为 0%**，累计 8 次运行、
   414 条 task 记录（含重复/中断的调试批次），没有一条被判定为成功。这是跨多次
   参数调整（standard/lookahead、有无 history window 限制）持续复现的现象，而不是
   单次实验的偶然波动。
2. **AI2-THOR single-agent 实验是本仓库第一次真正跑起来**（`doubao_ai2thor_k3_s0.5_w10_20260728_103813`，
   32 shard，2026-07-28 10:38 启动），截至统计时仍在运行中，尚无一条已完成的
   episode 记录，暂时无法给出成功率。
3. 早期批次（无 `-w`/history-window-size 限制）中出现了大量 **HTTP 413（请求体过大）
   的 API 错误**（最高占比 44.6%），这是历史帧图片累积导致的 payload 超限；引入
   `-w 10`（缩短短期历史窗口）后，413 错误在最新两次运行中已降为 0%，是目前唯一
   被验证"确实带来改善"的调参方向。
4. 剩余的失败原因集中在两类：agent 主动判定任务不可行而输出 `FAIL`
   （约 3%~43%，波动较大），以及 agent 自称 `DONE` 但被 `evaluate_node` 判定
   条件未满足（约 18%~57%）。两者合计构成绝大多数失败样本，指向 **agent 的空间
   导航/物体交互能力不足**，同时也发现个别 case 存在"动作执行成功但状态未按预期
   落地"的可疑现象，尚未能与"模型判断错误"完全区分开（详见第四节）。

## 二、实验清单与逐次结果

以下按时间顺序列出目前 `benchmark_outputs` 下有 `episode_*.json` 产出的全部
single-agent 运行。前 4 个 ProcTHOR 批次任务量很小（1~56 条），是调参/联调过程
中的中间尝试，未覆盖完整 127 条 CSV；后 4 个 ProcTHOR 运行为完整或接近完整的
批次。

### ProcTHOR

| 运行目录 | 时间 | 模式 | history\_window | task 数 | 成功 | 成功率 | 平均步数 |
|---|---|---|---|---:|---:|---:|---:|
| `batch_20260724_175407_of-064` | 07-24 | standard | 未设置 | 14 | 0 | 0.0% | 5.6 |
| `batch_20260725_220114_of-032` | 07-25 | standard | 未设置 | 56 | 0 | 0.0% | 12.8 |
| `batch_20260727_003949_of-032` | 07-27 凌晨 | standard | 未设置 | 1 | 0 | 0.0% | 6.0 |
| `batch_20260727_101436_of-032` | 07-27 上午 | standard | 未设置 | 56 | 0 | 0.0% | 31.1 |
| `benchmark_doubao_procthor_k3_s0.5_lookahead_20260727_163319` | 07-27 16:33 | lookahead | 未设置 | 46 | 0 | 0.0% | 15.1 |
| `benchmark_doubao_procthor_k3_s0.5_lookahead_20260727_172528` | 07-27 17:25 | lookahead | 未设置 | 87 | 0 | 0.0% | 23.3 |
| `benchmark_doubao_procthor_k3_s0.5_lookahead_w10_20260727_190714` | 07-27 19:07 | lookahead | 10 | 127（全量） | 0 | 0.0% | 31.3 |
| `benchmark_doubao_procthor_k3_s0.5_w10_20260728_001152` | 07-28 00:11 | standard | 10 | 127（全量） | 0 | 0.0% | 28.5 |

全部使用同一模型 `Doubao-Seed-2.0-lite`、同一份任务集
`experiments/csv/procthor/Spatial-Annotation-procthor-Gpt-5p4.csv`（127 条）、
同一份环境配置 `experiments/configs/procthor/config_close_doubao-2.yaml`
（`k=3` 张近期原图 + `s=0.5` 历史图缩放）。差异仅在 `-d lookahead` 决策模式和
`-w` 历史窗口大小两个变量上。

### AI2-THOR

| 运行目录 | 时间 | 模式 | history\_window | task 数 | 成功 | 成功率 |
|---|---|---|---|---:|---:|---:|
| `doubao_ai2thor_k3_s0.5_w10_20260728_103813` | 07-28 10:38 | standard | 10 | 0（运行中） | - | - |

对应 `hope/single_benchmark_ai2thor.hope`，32 shard，任务集为
`experiments/csv/ai2thor/Spatial-Annotation-ai2thor-doubao-seed-2.0-lite.csv`。
统计时各 shard 的 `runtime_status.json` 均为 `"state": "running"`，尚无
`episode_*.json` 产出，需等运行完成后另行统计。

## 三、失败原因分布（按运行分别列出）

| 运行 | Model FAIL | Model DONE 但条件未过 | 超步数 | 连续失败提前终止 | HTTP 413 | 其他 |
|---|---:|---:|---:|---:|---:|---:|
| `batch_20260724_175407` (14) | 42.9% | 57.1% | - | - | - | - |
| `batch_20260725_220114` (56) | 30.4% | 17.9% | 1.8% | 5.4% | 44.6% | - |
| `batch_20260727_003949` (1) | - | 100% | - | - | - | - |
| `batch_20260727_101436` (56) | 42.9% | 37.5% | 19.6% | - | - | - |
| `lookahead_163319` (46) | 13.0% | 26.1% | - | 19.6% | 41.3% | - |
| `lookahead_172528` (87) | 3.4% | 40.2% | 18.4% | 33.3% | 4.6% | - |
| `lookahead_w10_190714` (127) | 8.7% | 23.6% | 33.1% | 34.6% | 0% | - |
| `w10_001152` (127，**最新 standard**) | 37.0% | 37.0% | 19.7% | 6.3% | 0% | - |

关键观察：

- **HTTP 413 随 `-w`（history\_window\_size）引入而消失**：未设置 `-w` 的三次运行
  413 占比 41.3%~44.6%；一旦设置 `-w 10`，两次全量运行（`lookahead_w10_190714`、
  `w10_001152`）413 占比均为 0%。说明历史帧图片累积是此前大量任务"未战先败"
  （尚未真正开始推理就在 API 层被拒绝）的主因，`-w 10` 是已验证有效的修复。
- **"连续 4 次动作失败提前终止"在 lookahead 模式下显著更高**（19.6%~34.6%），
  standard 模式下最新一次仅 6.3%。初步推测 lookahead 的"预演多个候选动作再选择"
  机制在候选评估阶段更容易连续撞到不可行的动作（如反复被同一家具卡住），但样本量
  有限，仍需要更多同条件对比确认。
- **最新一次全量 standard 运行（`w10_001152`）里，`Model FAIL` 与
  `Model claimed DONE 但条件未过` 两者各占 37.0%，合计 74%**，是当前最主要的
  失败来源，超步数（19.7%）和连续失败提前终止（6.3%）合计仅占约 26%。这说明在
  排除 413 这类基础设施问题之后，**真正卡住 ProcTHOR 成功率的是 agent
  自身的任务完成判断（该继续找/该放弃/该报 DONE）与实际环境状态之间的偏差**。

## 四、Case 级别复盘（最新 `w10_001152` 全量运行）

抽查该运行下的多条 trajectory，观察到两类具体模式：

### 4.1 Agent 主动放弃（Model indicated FAIL）—— 空间推理/避障能力不足

典型例子 `procthor000`（"把番茄拿到床边等我"）：agent 76 步内成功找到并拿到了
番茄，但被一扇反复尝试 `OpenObject(Door)` 却始终打不开的门挡住，最终自行判定
"任务无法完成"输出 `FAIL`。全程可见大量重复性的定位/对齐尝试（为了拾取一个
物体反复 `Rotate`/`Move` 十几步才成功），以及把冰箱认成洗碗机等物体识别错误。
这类失败更像是 Doubao-Seed-2.0-lite 在复杂室内导航+精细交互任务上的真实能力
短板，而非代码 bug。

### 4.2 Agent 自称完成但被判定失败（Model claimed DONE）—— 部分案例可疑

典型例子 `procthor112`（"打开一部手机并带到浴室"），成功条件为
`object_in_room(CellPhone, Bathroom) AND object_state(CellPhone, isToggled=True)`。
Trajectory 显示 agent 先 `PickupObject(CellPhone)` 后 `ToggleObjectOn(CellPhone)`，
两步均被环境判定为执行成功（`reward=0.1`，`error_message=null`），随后走到
浴室输出 `DONE`；但最终评估：

```
Condition 1: object_in_room(CellPhone, Bathroom)  Score: 0.00 ❌
Condition 2: object_state(CellPhone, isToggled=True)  Score: 0.00 ❌
```

`isToggled` 在动作执行时"看起来成功"、评估时却是 `False`，与"agent 走错了房间"
不属于同一类问题，更像是 **物体交互目标选择（`_find_interaction_target`，
`mllm_base_agent/environments/procthor/wrapper.py`）在拾取后物体 `visible`
状态变化、或场景中存在同类型多实例时，可能操作到了与预期不一致的物体实例，
导致状态未按预期落地**。当前 `live.log` 日志级别不记录每步的
`lastActionSuccess`/`objectId`/完整 `metadata` 快照，无法在不重跑的情况下
100% 坐实这一猜测，建议按第五节的方式针对性复现。

### 4.3 评估器本身的实现未见明显逻辑错误

审查 `evaluation/procthor/base.py::MultiConditionEvaluator` 及
`create_evaluator_from_config`，`object_state` / `object_in_hand` /
`object_in_receptacle` / `agent_in_room` / `object_in_room` / `agent_near_object`
各条件类型实现完整，`get_agent_room` 依赖的 `env.scene["rooms"]`（`roomType` +
`floorPolygon`）在 `data/procthor-10k` 数据集中确实存在且非空，评估失败时也会
打印详细的逐条件得分，不存在"评估函数恒返回失败"这类系统性 bug。因此第 4.2 节
的可疑现象更可能出在"动作执行"与"评估读取"之间的状态一致性，而不是评估逻辑
本身。

## 五、建议

1. **优先复核第 4.2 类问题**：在
   `mllm_base_agent/environments/procthor/wrapper.py::step_with_action_dict`
   中临时打印每步 `event.metadata.get("lastActionSuccess")` + 实际操作的
   `objectId`，并在 `evaluate_node` 判定失败时 dump 一次目标物体的完整状态
   字典，挑 2~3 个"claimed DONE 但失败"的 task 单独重跑复现，确认是"模型误判
   自己已完成"还是"动作未真正生效/操作了错误的物体实例"。
2. **用最简单的 `task_presets`（如 `open_fridge`）先验证 pipeline 通路**：
   由于 ProcTHOR 至今没有出现过一次成功，建议先用配置文件里内置的单条件简单
   任务跑几条，确认"交互执行 → 环境状态更新 → 评估读取"整条链路本身是通的，
   再回头判断复杂多条件任务的失败是模型能力问题还是链路问题。
3. **等待并统计 AI2-THOR 首次 single-agent 运行结果**：`doubao_ai2thor_k3_s0.5_w10_20260728_103813`
   完成后应重新统计成功率与失败原因分布，与 ProcTHOR 做交叉对比——如果
   AI2-THOR 也是 0% 成功率，则问题更可能出在 `mllm_base_agent/agent/runner.py`
   通用逻辑或模型能力上；如果 AI2-THOR 有正的成功率，则应重点排查
   ProcTHOR 环境/评估特有的部分（房间边界、语义映射等）。
4. **lookahead vs standard 的对比样本还不够干净**：目前两种模式都各自经历过
   "无 `-w`→有 `-w`"的参数变化，建议后续在完全相同的 `-w 10` 设置下，各自补跑
   一次完整 127 条任务的对照实验，才能公平判断 lookahead 决策模式是否值得
   继续投入（目前看它的"连续失败提前终止"比例明显更高，值得关注）。
5. **`-w 10` 应作为后续所有 ProcTHOR/AI2-THOR single-agent 实验的默认参数**，
   因为它已经把 413 错误从 40%+ 降到 0%，是目前唯一有确定收益的改动；后续如果
   仍出现 413，应考虑进一步降低 `-w` 或降低 `-s`（历史图片缩放比例）。
