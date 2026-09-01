# 技术设计文档：ToolRL-Lite

> 本文档是《ToolRL-Lite 项目立项文档》的技术细化，按模块给出可落地、可复现的设计细节：统一数据约定、环境接口、奖励函数、训练超参、评测指标、部署形态与一键复现流程。
>
> 阅读顺序：先读立项文档把握全局，再按本文档第 1 节的数据流逐步下钻。

---

## 1. 总体数据流与关键约定

### 1.1 阶段与产物

```
阶段           输入                                  产物
──────────────────────────────────────────────────────────────────────────
任务环境       任务模板 + 随机种子        →  任务 JSONL（含 gold 判定信息）
数据合成       任务池 + teacher 模型      →  sft_trajectories.jsonl
SFT 冷启动     sft_trajectories.jsonl    →  sft_model（checkpoint + HF 权重）
GRPO RL       任务 prompt 池 + 基座      →  grpo_model + wandb 曲线
评测          4 组模型 × benchmark       →  report.md / report.html
部署          最终 checkpoint            →  MCP Server + Agent Runtime
```

### 1.2 所有模块共用的任务 JSON schema

环境生成、数据合成、训练、评测四端共用同一任务格式，保证"环境 → 数据 → 训练 → 评测"闭环无需转换：

```json
{
  "task_id": "api_0001",
  "category": "multi_tool",
  "instruction": "帮我订明天北京到上海的机票，并添加一条日程提醒",
  "tools": [
    {
      "name": "flight.search",
      "description": "搜索航班，参数：from, to, date",
      "parameters": {"type": "object", "properties": {"from": {"type": "string"}, "to": {"type": "string"}, "date": {"type": "string"}}, "required": ["from", "to", "date"]}
    }
  ],
  "gold": {
    "calls": [
      {"api": "flight.search", "params": {"from": "北京", "to": "上海", "date": "明天"}},
      {"api": "alarm.add", "params": {"time": "09:00", "content": "去机场"}}
    ],
    "answer": "已为您查询到 3 个航班，并设置 09:00 日程提醒"
  },
  "max_steps": 5,
  "min_steps": 2
}
```

字段约定：
- `instruction`：给模型的用户指令，环境端与训练端严格一致；
- `tools`：注入系统提示词的函数定义，按 OpenAI Function Calling schema 子集生成；
- `gold.calls`：程序化判定的唯一依据，**不允许人工/LLM 裁判参与打分**；
- `max_steps` 与 `min_steps`：环境执行上限与最优步数（步数奖励依赖后者）。

### 1.3 轨迹 JSONL schema（SFT 数据）

```json
{"task_id": "api_0001", "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "<think>...</think>\\n<tool_call>{\"name\":\"flight.search\",\"arguments\":{...}}</tool_call>"}, {"role": "tool", "content": "{\"flights\": [...]}"}, {"role": "assistant", "content": "<think>...</think>\\n<answer>已为您查询到 3 个航班...</answer>"}], "success": true, "steps": 2}
```

训练端读取 `messages` 即可构造 prompt/response 对；`success` 与 `steps` 供过滤与统计。

---

## 2. 任务环境（envs/）

### 2.1 通用接口

两个环境统一实现 `BaseToolEnv`，让训练/评测端与具体环境解耦：

```python
class BaseToolEnv(abc.ABC):
    def reset(self, task: dict) -> dict: ...        # 返回初始观察（工具列表 + 指令）
    def step(self, action: dict) -> dict: ...       # action={"api":..., "params":...}，返回 {observation, done}
    def judge(self, trajectory: list[dict]) -> dict: ...  # 程序化判定，返回各奖励项与是否正确
    def task_generator(self, seed: int) -> Iterator[dict]: ...  # 参数化批量生成任务
```

环境不感知模型与训练框架：训练端通过 HTTP/进程内调用 `step` 完成 rollout，评测端在采样后批量调用 `judge`。

### 2.2 模拟 API 沙箱（envs/api_sandbox/）

**MVP 工具清单（10 个，覆盖单工具/多工具/多轮场景）**：

| 领域 | 工具 | 用途 |
|---|---|---|
| 天气 | weather.query / weather.forecast | 查询当前天气 / 未来预报 |
| 出行 | flight.search / flight.book | 航班搜索 / 预订 |
| 出行 | train.search / train.book | 火车搜索 / 预订 |
| 差旅 | hotel.search / hotel.book | 酒店搜索 / 预订 |
| 效率 | meeting.create / alarm.add | 创建会议 / 添加日程提醒 |

每个 mock API 用 FastAPI 实现，返回**固定种子的确定性数据**（同参数必同结果），保证奖励可复现。请求参数缺失或非法时返回 `{"error": ...}` 而不是崩溃。

**任务生成器**：模板 + 随机槽位。例如"订 {date} {from}→{to} 的机票，预算 {budget} 元"随机填充城市/日期/预算；同时按比例注入干扰工具（如任务本不需要 `hotel.search`），训练模型学会不调用无关工具。

**判定逻辑（分级给分）**，对轨迹中每一步工具调用：

1. 调用 API 名与 `gold.calls` 对应步一致 → 得该步的工具分；
2. 参数逐项比对（key 集合一致 + 值等价，日期做归一化）→ 每项正确得参数分；
3. 最终 `<answer>` 与 gold 答案程序化等价（数字/日期归一化后比对）→ 得答案分；
4. 未在 gold 中出现的多余调用 → 计为错误调用并扣分。

### 2.3 Text2SQL 环境（envs/text2sql/）

- 3 个 schema：电商（users/orders/order_items/products）、员工（departments/employees）、库存（warehouses/inventory）；
- 每库预置 50-200 行种子数据，问题由模板生成：单表过滤、多表 join、聚合分组、排序分页各占一定比例；
- 成功条件 = 执行生成 SQL 与 gold SQL 的结果集一致；比对前做**列序归一化、行序归一化（按全部列排序）、NULL 与类型归一化**，避免"结果对但格式不同"被误判；
- 安全约束：连接以只读模式打开，预编译黑名单拦截 `DROP/DELETE/UPDATE/INSERT/PRAGMA` 等危险语句；执行超时 5 秒。

### 2.4 奖励项与默认权重

环境端 `judge` 输出原始分项，训练端组装总奖励：

```
R_total = w1·R_format + w2·R_correct + w3·R_answer − w4·R_steps

w1 = 0.40  格式合规：轨迹全程 JSON schema 合法
w2 = 0.30  调用正确：工具选择 + 参数逐项比对（部分得分）
w3 = 0.20  最终答案正确
w4 = 0.10  步数惩罚：R_steps = max(0, steps − min_steps) / max_steps
```

权重为**初始值**，W5 奖励消融以它为中心做 ± 调整，验证 ToolRL 论文"奖励类型/粒度/尺度是关键变量"的结论。

---

## 3. 数据合成（data/）

### 3.1 流程

1. 从任务池随机采样任务（每任务可多轮采样）；
2. teacher 模型（DeepSeek / Qwen-Max API）携带工具列表以 ReAct 风格执行，环境回传真实结果；
3. 记录完整轨迹（含工具返回）；
4. 拒绝采样过滤；
5. 输出 SFT JSONL 与 RL prompt 池（RL 池只保留 `instruction + tools`，不含轨迹）。

### 3.2 过滤规则（拒绝采样）

- 轨迹必须 `success == true`；
- `steps <= max_steps` 且 `steps >= min_steps`（步数太少说明可能偷看了 gold，步数太多说明试错）；
- 无重复调用同一工具 ≥ 3 次（排除穷举）；
- 最终答案与 gold 程序化比对一致。

### 3.3 数据量与预算

| 用途 | 数量 | 说明 |
|---|---|---|
| SFT 轨迹 | 3-5k 条 | 覆盖两类环境、各维度任务 |
| RL prompt 池 | 1-2k 条 | 只含 prompt，轨迹由策略 rollout 生成 |
| API 预算 | ≤ 100 元 | 按 3-5k 条 × 往返 2-4 次调用估算 |

---

## 4. 训练（train/）

### 4.1 输出格式约定（Qwen3）

统一使用 Qwen3 的思考 + 工具调用格式：

```
<think>先规划再行动...</think>
<tool_call>{"name":"flight.search","arguments":{"from":"北京","to":"上海","date":"2026-09-01"}}</tool_call>
```

工具返回后：

```
<think>查询结果符合预期，给出最终答案</think>
<answer>已为您查询到 3 个航班...</answer>
```

格式合规判定 = 标签配对完整 + `<tool_call>` 内为合法 JSON + arguments 满足工具 schema。**SFT 阶段用同一判定函数做数据质检与 checkpoint 验证，与 RL 奖励函数共用一份代码**，避免"训练打分与 SFT 质检口径不一致"。

### 4.2 SFT 冷启动

| 配置项 | 0.6B 全参 | 1.7B LoRA |
|---|---|---|
| 优化器 | AdamW | AdamW（仅 LoRA 参数） |
| 学习率 | 1e-5 | 2e-4（LoRA），rank=16, alpha=32 |
| 训练轮数 | 3 | 3 |
| batch size | 16（grad_accum 视显存调整） | 16 |
| max_seq_len | 4096 | 4096 |
| 调度器 | cosine + warmup 10% | cosine + warmup 10% |
| 精度 | bf16 | bf16 |

验证目标：验证集格式合规率 ≥ 90%，工具选择准确率较基座明显上升。

### 4.3 GRPO 强化学习

**算法回顾**（DeepSeekMath，arXiv:2502.03393）：每个 prompt 采样 G=8 条响应，组内相对优势 `A_i = (r_i − mean(r)) / std(r)`，策略目标：

```
J(θ) = (1/G) Σ_i [ min(ρ_i·A_i, clip(ρ_i, 1−ε, 1+ε)·A_i ) − β·KL(πθ ‖ πref) ]
```

其中 `ρ_i = πθ(o_i|q) / π_old(o_i|q)`，KL 用逐 token 无偏估计。

**框架**：verl（主）/ TRL（备）。verl 侧需要自己实现的只有奖励函数（挂到 reward manager）+ 任务 prompt 数据格式；rollout 由 vLLM 引擎完成。

**初始超参**（0.6B，单卡 4090/A800，均标注为待调初始值）：

| 配置项 | 初始值 |
|---|---|
| group_size（每条 prompt 采样数） | 8 |
| 学习率 | 1e-6 |
| clip 系数 ε | 0.2 |
| KL 系数 β | 1e-3（verl 习惯）～ 0.04（TRL 默认）区间试探 |
| max_prompt_len / max_response_len | 2048 / 1024 |
| 每轮 prompt 数 × 采样数 | 128 × 8 = 1024 条轨迹 |
| 训练步数 | 至 reward 曲线平台期，预计 200-500 步 |
| 精度 / 显存优化 | bf16 + gradient checkpointing（必要时 FlashAttention） |

**训练监控（wandb 面板）**：

- reward 总曲线 + 四项分项曲线（格式/正确/答案/步数）；
- KL 散度（防策略漂移过大）；
- 响应长度分布（长度突然暴涨常伴随 reward hacking）；
- 熵（持续下降至 0 是坍缩信号）。

**奖励实现注意点**：

- 所有分项先归一化到 0-1 再加权，避免尺度不同导致某项主导；
- 组内标准化前检查 std=0 的情况（全组同分时优势置 0，防止除零）；
- 只奖励最终轨迹的判定结果，不额外奖励中间"部分正确"的试探过程（步数惩罚已覆盖）。

### 4.4 常见训练故障与对策

| 现象 | 对策 |
|---|---|
| 熵坍缩（输出重复） | 提高 KL 系数 β、降低学习率、提高采样温度、增大 group_size |
| 格式崩坏（JSON 截断/非法） | 加大 w1 格式权重；回退 SFT checkpoint 重训；max_response_len 留足余量 |
| reward hacking（如空参数骗格式分） | 空 arguments 判定为格式不合规；答案归一化比对；记录案例进技术报告 |
| 显存不足 | LoRA 降级、缩短 max_response_len、gradient checkpointing |

---

## 5. 评测（eval/）

### 5.1 Benchmark 构成（300-500 条）

| 维度 | 占比 | 说明 |
|---|---|---|
| 单工具 | 25% | 一次调用即完成 |
| 多工具 | 30% | 需按顺序调用 ≥ 2 个工具 |
| 多轮 | 20% | 依赖前一步返回结果决策 |
| 干扰项 | 15% | 混入无关工具，验证不误调用 |
| 长程 | 10% | 步数上限内完成 ≥ 3 步任务 |

### 5.2 指标定义

| 指标 | 公式 |
|---|---|
| 格式合规率 | 轨迹全程合法（标签 + JSON schema）的任务数 / 总任务数 |
| 工具选择准确率 | 首步调用与 gold 首步一致的任务数 / 总任务数 |
| 参数正确率 | 正确参数项数 / gold 参数总项数（全任务聚合） |
| 端到端成功率 | `judge` 判定 success 的任务数 / 总任务数 |
| 平均步数 | 所有任务实际步数均值 |
| token 成本 | 每任务平均生成 token 数（评估部署成本） |

### 5.3 评测流程

1. 同一 benchmark、同一随机种子、同一解码参数（temperature=0.7, top_p=0.9）对四组模型采样；
2. 环境逐任务执行并 `judge`；
3. 输出 Markdown 报告（四组对比表 + 分维度柱状图）+ 失败 case 清单；
4. 可复现性保证：评测脚本入参只有 `model_path` 与 `--seed`，两次运行结果应一致。

对比组：原始基座 / SFT 版 / GRPO 版 / Qwen3-32B（教师级参照，衡量上限）。W6 起接入 BFCL 做外部横向对标。

---

## 6. 部署（serving/）

### 6.1 MCP Server

- 从任务环境导出的工具 schema（同 1.2 节 `tools` 字段）直接注册为 MCP tools；
- 用 MCP Python SDK 提供 stdio transport，模型权重由本地 `transformers` 或 vLLM 服务加载；
- 每个工具调用加超时（默认 10s）与参数校验，错误以结构化内容返回，不抛出未处理异常。

### 6.2 最小 Agent Runtime

```
循环：
  1. 拼接 system（工具列表 + 格式说明）+ user 指令 → 模型生成
  2. 若含 <tool_call>：解析 JSON → 环境/真实 API 执行 → 结果作为 tool 消息回填 → 回到 1
  3. 若含 <answer>：校验格式后返回
  4. 步数 > max_steps 或解析失败 → 终止并返回错误
```

不依赖 LangGraph/AutoGen 等重型框架，循环与工具注册逻辑合计控制在 200 行以内，便于审阅与讲解。

---

## 7. 一键复现（scripts/）

### 7.1 `run_all.sh` 阶段划分

```
1. env_check   验证 conda 环境与依赖版本
2. gen_tasks   生成任务 JSONL（种子固定）
3. synth       蒸馏轨迹 → 拒绝采样 → sft_data
4. sft         训练 SFT checkpoint
5. grpo        训练 GRPO checkpoint（默认单卡，可 --skip 以先跑通流程）
6. eval        四组模型评测 → reports/
7. serve       启动 MCP Server（--demo 模式）
```

每个阶段幂等：产物带 hash 文件名，重跑不覆盖已有结果；`--skip` 支持跳过已完成的阶段。

### 7.2 依赖清单（conda 环境 toolrl-lite，Python 3.10）

| 包 | 用途 | 版本建议 |
|---|---|---|
| torch / transformers | 模型训练与推理 | 与 Qwen3 官方要求匹配（≥2.3） |
| verl | GRPO 训练（主） | 最新 release |
| trl | GRPO 训练（备） | ≥0.16 |
| vllm | rollout 推理引擎 | 与 verl 兼容版本 |
| fastapi / uvicorn | mock API 沙箱 | 最新 |
| pydantic | schema 校验 | ≥2 |
| mcp | MCP Server | python-sdk 最新 |
| wandb | 训练监控 | 最新 |
| pytest | 单测 | 最新 |

---

## 8. 风险与技术要求对照

| 立项文档风险 | 本文档对应技术对策 |
|---|---|
| GRPO 训练不稳定 | 4.4 故障-对策表：KL 系数、奖励归一化、降学习率 |
| 算力预算超支 | LoRA 降级、gradient checkpointing、缩短响应长度、0.6B 优先 |
| 环境开发拖期 | MVP 只做 API 沙箱（第 2 节接口先行），Text2SQL 延后 |
| Reward hacking | 4.3 奖励实现注意点 + 4.4 案例记录，作为技术报告素材 |
| 可复现性被质疑 | 固定种子、确定性 mock API、幂等脚本、评测同解码参数 |

---

## 9. 参考

- DeepSeekMath（GRPO 原始论文）：arXiv:2502.03393
- ToolRL：arXiv:2504.13958（奖励设计 8 变体与消融是 W5 的参照系）
- verl / TRL / vLLM / MCP Python SDK：见立项文档第 10 节

> 本文档中的权重、超参、工具清单均为可执行的初始值，实际以 W1-W5 实验结论为准，迭代结果回写本文档（版本号 + 变更记录见文末）。

## 变更记录

| 版本 | 日期 | 变更 |
|---|---|---|
| v0.1 | 2026-08-30 | 初稿：环境接口、奖励函数、训练超参、评测指标、部署与复现流程 |
