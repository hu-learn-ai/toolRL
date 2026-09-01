# 项目立项文档：ToolRL-Lite

> **一句话定位**：把 Qwen3 小模型训练成可靠的工具调用 Agent——用可验证环境 + SFT + GRPO 强化学习打通"环境 → 数据 → 训练 → 评测 → 部署"全链路，以工业级开源仓库标准交付。
>
> 项目代号 `ToolRL-Lite`（上线前可自行改名，避免与现有项目重名）。
> 立项日期：2026-08-30

---

## 1. 背景与动机

### 1.1 岗位需求侧

对 2026 年 Agent / 大模型算法岗真实 JD（阿里巴巴、百度、腾讯、淘天、云知声）的调研显示，高频要求高度集中于：

| 能力项 | JD 原文关键词 | 本项目覆盖模块 |
|---|---|---|
| Post-training | SFT、RL、GRPO/DAPO/PPO | 训练主线 |
| Agent 核心 | 工具调用、多步推理、Planning、Self-correction | 任务环境 + 训练目标 |
| 奖励设计 | Reward Function、Reward Hacking | 可验证奖励 + 消融实验 |
| 训练框架 | verl、ROLL、RLHF 工程经验 | 技术选型 |
| 评测 | Agent 评测体系、数据飞轮 | 自建 benchmark |
| 框架理解 | LangGraph/AutoGen、Function Calling | MCP 服务 + Agent 运行时 |
| 模型经验 | Qwen/Llama/DeepSeek 二次开发 | 基座 Qwen3 |

### 1.2 差异化判断

- "又一个 RAG Demo / Multi-Agent 聊天框架"已严重同质化；
- 纯 SFT 微调项目门槛已不足以形成区分度；
- **能在一个仓库里端到端复现 Agentic RL（含奖励设计分析与训练曲线）的个人项目稀缺**，且同时命中算法岗与 Agent 岗的双重要求。

### 1.3 学术背景

ToolRL（UIUC，arXiv:2504.13958）证明：在工具调用任务上，GRPO + 精细奖励设计比 SFT 提升约 15%，奖励的类型/粒度/尺度是关键变量。本项目以该技术路线为骨架，缩小规模、聚焦可复现性。

---

## 2. 项目目标

### 2.1 核心目标

用 **Qwen3-0.6B / 1.7B** 作为基座，通过 SFT 冷启动 + GRPO 强化学习，使其在自建工具调用环境上显著提升工具调用能力，并完整开源。

### 2.2 验收标准（量化 KPI）

| 指标 | 基线（原始模型） | 目标 |
|---|---|---|
| 工具调用格式合规率 | 跑基线获得（预计 40-60%） | ≥ 95% |
| 工具选择准确率 | 跑基线获得 | 相对提升 ≥ 15% |
| 参数生成正确率 | 跑基线获得 | 相对提升 ≥ 20% |
| 多步任务端到端成功率 | 跑基线获得 | 翻倍以上 |
| 可复现性 | — | 一条命令跑通训练/评测，CI 绿灯 |

### 2.3 交付物清单

1. GitHub 仓库（中英双语 README、一键复现脚本、CI 测试、Apache-2.0 License）
2. HuggingFace 模型权重（SFT 版与 GRPO 版）
3. 自建评测集（300-500 条）+ 自动化评测报告
4. 技术报告（训练曲线、奖励消融、reward hacking 案例分析）
5. MCP Server：训练后的模型可直接被 Claude/Cursor 等客户端调用

---

## 3. 系统架构

```
[任务环境] → [数据合成] → [SFT 冷启动] → [GRPO RL]
 可验证沙箱    蒸馏轨迹                    奖励驱动优化
     ↑                                        ↓
[评测体系] ← [训练曲线 / 检查点] ←────────────┘
     ↓
[MCP 工具服务 + Agent 运行时] → 开源交付
```

### 3.1 模块一：任务环境（envs/）

gym 风格的可验证工具调用沙箱，MVP 阶段实现 2 个环境：

- **模拟 API 环境**：天气查询、票务预订、日程管理等 8-12 个 mock API（FastAPI 实现），每个任务有程序可判的成功条件（调用了正确的 API + 参数全对 + 返回正确答案）；
- **Text2SQL 环境**：SQLite 内存数据库（电商/员工等 3 个 schema），成功条件 = 执行生成的 SQL 与 gold SQL 结果集一致。

**设计原则**：奖励完全可验证（不需要人工或 LLM 裁判）；支持部分得分（分级奖励）；环境可参数化批量生成任务（课程学习的基础）。

### 3.2 模块二：数据合成（data/）

- 用大模型（DeepSeek/Qwen-Max API）作 teacher，在环境中采样轨迹并蒸馏；
- 拒绝采样过滤：只保留成功且步数合理的轨迹作为 SFT 数据；
- 产出：约 3-5k 条 SFT 轨迹 + RL 任务的 prompt 池（RL 阶段只给 prompt，轨迹由策略自己 rollout 生成）。

### 3.3 模块三：训练（train/）

**阶段一 SFT 冷启动**：让模型学会输出合法的 `思考 + 工具调用 JSON` 格式；0.6B 可全参微调，1.7B 可用 LoRA。

**阶段二 GRPO 强化学习（核心）**：

- 框架：verl（首选，JD 点名）或 TRL（备选，上手快）；
- 每个任务采样 8 条轨迹，组内标准化计算优势，无 critic；
- 奖励函数（重点打磨对象）：

```
R_total = w1 · R_format   （JSON schema 合规，程序化判定）
        + w2 · R_correct  （工具选择 + 参数逐项比对，分级给分）
        + w3 · R_answer   （最终答案正确）
        - w4 · R_steps    （步数惩罚，抑制穷举试错）
```

- 训练监控：reward 曲线、KL 散度、响应长度分布、熵（防坍缩）。

### 3.4 模块四：评测（eval/）

- 自建 benchmark：300-500 条，维度覆盖单工具/多工具/多轮/干扰项（无关工具混入）/长程任务；
- 指标：格式合规率、工具选择准确率、参数正确率、端到端成功率、平均步数、每任务 token 成本；
- 对比组：原始基座 / SFT 版 / GRPO 版 / Qwen3-32B（教师级参照），输出 Markdown + 图表报告；
- 加分项：接入 BFCL（Berkeley Function Calling Leaderboard）或 API-Bank 做外部横向对比。

### 3.5 模块五：部署（serving/）

- 将训练后的模型包成 **MCP Server**（工具调用能力通过 MCP 协议暴露），任何支持 MCP 的客户端可直接使用；
- 附最小 Agent 运行时（ReAct 循环 + 工具注册），不依赖重型框架，展示对底层原理的理解。

---

## 4. 技术选型

| 层 | 选型 | 理由 |
|---|---|---|
| 基座模型 | Qwen3-0.6B / 1.7B | 小模型可全参训练，国内下载方便 |
| RL 训练框架 | verl（主）/ TRL（备） | JD 点名 verl；TRL 适合快速原型 |
| 推理引擎 | vLLM（训练 rollout）/ transformers（本地评测） | verl 默认 rollout 引擎 |
| 环境实现 | FastAPI + SQLite | 轻量、可控、全程序化判定 |
| 评测 | 自研 + BFCL/API-Bank | 内部纵向对比 + 外部横向对标 |
| 部署 | MCP Python SDK | 2026 年工具调用事实标准协议 |
| CI | GitHub Actions（单测 + lint + 冒烟） | 仓库质量信号 |
| 实验管理 | wandb（免费个人版） | 训练曲线是技术报告核心素材 |

## 5. 仓库结构

```
toolrl-lite/
├── envs/                # gym 风格可验证工具环境
│   ├── api_sandbox/
│   └── text2sql/
├── data/                # 数据合成管线与产物
│   ├── synthesis/
│   └── sft_data/
├── train/
│   ├── sft/
│   └── grpo/            # verl/TRL 配置与启动脚本
├── eval/
│   ├── benchmark/
│   ├── runners/
│   └── reports/
├── serving/
│   ├── mcp_server/
│   └── agent_runtime/
├── docs/
│   ├── design_zh.md     # 技术设计文档（环境接口/奖励/超参/指标细节）
│   ├── report_zh.md     # 技术报告
│   └── report_en.md
├── scripts/             # run_all.sh 一键复现
└── .github/workflows/   # CI
```

## 6. 里程碑排期（8 周）

| 周次 | 里程碑 | 验收物 |
|---|---|---|
| W1 | 环境与基线 | 本机跑通 Qwen3-0.6B 推理；仓库骨架；基线评测数据 |
| W2 | 任务环境 v1 | 模拟 API 沙箱 + 50 个种子任务 + 单测 |
| W3 | 数据 + SFT | 3k 条蒸馏轨迹；SFT v1 模型；格式合规率显著提升 |
| W4 | GRPO 首跑 | 云 GPU 跑通 verl GRPO 全流程（哪怕指标难看） |
| W5 | 奖励迭代 | 奖励消融实验 ≥ 3 组；拿到第一组提升数据 |
| W6 | 评测体系 | 自建 benchmark 全量跑分；四组对比报告 |
| W7 | 部署 + 报告 | MCP Server 可用；技术报告初稿 |
| W8 | 打磨发布 | CI 绿灯；README 中英文；HF 权重发布；写推广帖 |

**节奏要求**：每周至少 3-4 次有意义的 commit，W4 起训练曲线持续沉淀到 wandb。

## 7. 资源与预算

- **本机（无 CUDA，Windows + conda）**：环境开发、单测、本地推理评测、数据质检；
- **云 GPU（AutoDL 等）**：SFT + GRPO 训练。0.6B 全参 + 单卡 4090/A800 即可，预计总花费 200-500 元；
- **API 费用**：teacher 蒸馏约 3-5k 条轨迹，预算 100 元以内。

## 8. 风险与应对

| 风险 | 概率 | 应对 |
|---|---|---|
| GRPO 训练不稳定（熵坍缩/格式崩坏） | 高 | 调大 KL 系数、奖励归一化、降低学习率——这本身就是技术报告的好素材 |
| 算力预算超支 | 中 | 降级 LoRA；缩短 rollout 长度；0.6B 优先 |
| 环境开发拖期 | 中 | MVP 只做 1 个环境，第二个放 W6 |
| Reward hacking | 中（反而是机会） | 记录案例（如输出空参数骗格式分），修复过程写入报告 |
| 被质疑重复造轮子 | 低 | 叙事差异化：个人可复现的 AgenticRL 教学级项目 + 完整工程化交付 |

## 9. 简历表述模板（完成后填数字）

> **ToolRL-Lite：小模型工具调用能力的强化学习训练系统**（个人项目，开源）
> - 设计可验证的工具调用环境（模拟 API + Text2SQL），基于 verl 实现 Qwen3-0.6B 的 SFT + GRPO 训练；
> - 设计格式/正确性/效率多维奖励函数并完成消融实验，定位并修复 X 起 reward hacking 案例；
> - 自建 500 条多维 benchmark，工具调用成功率从 X% 提升至 Y%（+Z%），多步任务成功率翻倍；
> - 训练后模型打包为 MCP Server，支持主流客户端直接调用。

---

## 10. 参考开源项目与学习路径

### 10.1 必读必跑（直接决定项目成败）

| 项目 | 地址 | 学什么 | 优先级 |
|---|---|---|---|
| **TinyZero** | github.com/hiyouga/TinyZero | 用小模型 + verl 最小成本复现 R1 式 GRPO，是理解"GRPO 到底怎么跑"的最短路径 | ★★★★★ |
| **ToolRL** | github.com/qiancheng0/ToolRL | 与本项目路线几乎一致：verl + GRPO + 工具调用奖励设计（论文 arXiv:2504.13958），其 8 种奖励变体是奖励消融的现成参照 | ★★★★★ |
| **verl** | github.com/volcengine/verl | 生产级 RL 训练框架，JD 点名；重点读 GRPO trainer、reward manager、vLLM rollout 集成 | ★★★★★ |
| **TRL** | github.com/huggingface/trl | HuggingFace 官方，GRPOTrainer 上手快，适合 W2-W3 先出 SFT 原型 | ★★★★ |

### 10.2 环境与评测（借鉴设计）

| 项目 | 地址 | 学什么 |
|---|---|---|
| **AgentGym** | github.com/WooooDyy/AgentGym | Agent 训练环境 + 框架 + 数据完整体系，看它如何 gym 化环境接口 |
| **AgentBench** | github.com/THUDM/AgentBench | 8 大交互环境的任务设计与程序化判定方式 |
| **τ-bench** | github.com/sierra-research/tau-bench | 工具调用 + 模拟用户多轮交互评测，POMDP 建模范式 |
| **BFCL / Gorilla** | github.com/ShishirPatil/gorilla | Berkeley Function Calling Leaderboard，可直接接入做外部对标 |
| **agentbench/AgentBench** | github.com/AgentBench/AgentBench | 40 个真实任务、纯规则三层打分（结构/指标/行为）的评测体系设计 |

### 10.3 数据与生态

| 资源 | 说明 |
|---|---|
| **ToolACE / xLAM / API-Bank 数据集** | 现成工具调用数据（ToolRL 训练数据即来源于此），可作冷启动补充 |
| **MCP Python SDK** | github.com/modelcontextprotocol/python-sdk，serving 模块直接依赖 |
| **vLLM** | github.com/vllm-project/vllm，rollout 推理引擎，了解 PagedAttention 即可 |
| **ROLL** | github.com/alibaba/ROLL，阿里 RL 训练框架，JD 点名，了解架构即可 |

### 10.4 建议学习顺序（与排期对齐）

1. **第 1 周**：精读 ToolRL 论文（重点 Section 3 奖励设计 + Section 5 消融）→ 跑通 TinyZero 的最小 GRPO 示例（租 1 张卡几小时）；
2. **第 2-3 周**：读 ToolRL 的 `verl/utils/reward_score/rlla.py`（核心奖励实现，仅一个文件）+ TRL 的 GRPOTrainer 源码；
3. **第 4 周起**：读 verl 的 GRPO trainer 主流程；评测阶段参考 AgentBench / τ-bench 的任务 schema 设计；
4. **全程**：把每个参考项目与论文标注到仓库 `docs/references.md`，既方便自己也体现学术素养。

---

## 11. 下一步行动

- [ ] 注册/确认云 GPU 平台账号，验证 verl 环境可装通
- [ ] W1 任务：本地跑通 Qwen3-0.6B 推理（hf-mirror 镜像已验证可行）+ 搭建仓库骨架
- [ ] 精读 ToolRL 论文并整理一页笔记
