# ToolRL-Lite

> 把 Qwen3 小模型训练成可靠的工具调用 Agent —— 用可验证环境 + SFT + GRPO 强化学习打通「环境 → 数据 → 训练 → 评测 → 部署」全链路。

[English](README_EN.md) · [立项文档](ToolRL-Lite项目立项文档.md) · [技术设计文档](docs/design_zh.md)

## 项目定位

ToolRL-Lite 是一个面向算法/Agent 岗位的个人开源项目：以 Qwen3-0.6B/1.7B 为基座，在自建的可验证工具调用环境上，用 SFT 冷启动 + GRPO 强化学习训练小模型可靠地调用工具。核心特点：

- **可验证奖励**：奖励全部由程序化判定产生（无 LLM/人工裁判），格式/正确性/答案/步数四维加权，支持部分得分；
- **端到端闭环**：环境 → 数据合成 → SFT → GRPO → 评测 → 部署全链路在一个仓库内复现；
- **确定性可复现**：固定种子任务生成、确定性 mock API、幂等合成与评测；
- **轻量可跑**：单卡 4090/A800 即可完成 0.6B 全参训练，评测无 GPU 也能冒烟跑通。

## 管线

```
[任务环境] → [数据合成] → [SFT 冷启动] → [GRPO RL]
 可验证沙箱    蒸馏轨迹                    奖励驱动优化
     ↑                                        ↓
[评测体系] ← [训练曲线 / 检查点] ←────────────┘
     ↓
[MCP 工具服务 + Agent 运行时] → 开源交付
```

## 仓库结构

```
├── envs/                # 可验证工具环境（api_sandbox / text2sql）
├── data/                # 数据合成管线（teacher、拒绝采样、断点续跑）
├── train/
│   ├── sft/             # SFT 冷启动（0.6B 全参 / 1.7B LoRA）
│   └── grpo/            # GRPO 强化学习（奖励函数 + TRL/verl 接入）
├── eval/                # 自建 benchmark + 四组模型评测报告
├── serving/             # MCP Server + Agent Runtime（开发中）
├── scripts/             # gen_tasks.py / gen_sft.py / run_all.sh
└── .github/workflows/   # CI（pytest + ruff + 评测冒烟）
```

## 快速开始

### 0. 环境准备（Python ≥ 3.10）

```bash
conda create -n toolrl-lite python=3.11 -y
conda activate toolrl-lite
pip install -e ".[env,dev]"
```

GPU 侧依赖（torch / transformers / peft / datasets / trl）在进入训练阶段时再装。

### 1. 配置 teacher（合成数据用，可选）

```bash
cp .env.example .env   # 填入 TEACHER_API_KEY
```

### 2. 一键复现（Linux GPU 机）

```bash
bash scripts/run_all.sh                    # 全流程，产物存在时自动跳过
bash scripts/run_all.sh --only synth,sft   # 只跑指定阶段
```

### 3. 分步执行

```bash
# 生成任务池（固定种子）
python scripts/gen_tasks.py --n 400 --seed 42

# 数据合成（teacher 蒸馏 + 拒绝采样 → SFT 轨迹 + RL prompt 池）
python scripts/gen_sft.py --n-per-env 2500 --out data_out

# SFT 冷启动
python -m train.sft --data data_out/sft_trajectories.jsonl --out checkpoints/sft

# GRPO 强化学习（从 SFT checkpoint 起步）
python -m train.grpo --model checkpoints/sft --data data_out/rl_pool.jsonl

# 评测：无 GPU 冒烟 / 真实模型对比
python -m eval --mock --size 40 --out eval_out/report.md
python -m eval --base Qwen/Qwen3-0.6B --sft checkpoints/sft --grpo checkpoints/grpo

# 启动 Mock API 沙箱
uvicorn envs.api_sandbox.server:app --port 8000
```

### 4. 测试

```bash
pytest
```

## 当前进度（2026-09-01）

| 模块 | 状态 | 说明 |
|---|---|---|
| 任务环境 envs/ | ✅ 完成 | api_sandbox 10 个 mock 工具 + text2sql 3 schema，程序化分级判定，FastAPI 服务 |
| 数据合成 data/ | ✅ 完成 | GoldTeacher/LLMTeacher、拒绝采样、断点续跑、预算保护（待真实 API 跑量） |
| SFT train/sft/ | ✅ 完成 | 0.6B 全参 / 1.7B LoRA，assistant loss 掩码，训练/验证指标 |
| GRPO train/grpo/ | 🚧 进行中 | 奖励函数与 TRL 入口就绪；verl 多轮集成待做 |
| 评测 eval/ | ✅ 完成 | 400 条 benchmark、四组对比报告、无 GPU 冒烟 |
| 部署 serving/ | 🚧 未开始 | MCP Server + Agent Runtime 待实现 |
| 工程化 | ✅ 完成 | git、CI（pytest+ruff+冒烟）、中英 README、Apache-2.0 |

## 路线图

- [x] W1 仓库骨架与工程化
- [x] W2 任务环境 v1（两个环境 + 单测）
- [ ] W3 数据合成跑量 + SFT 训练（代码就绪，待真实数据）
- [ ] W4 GRPO 首跑（TRL 路径就绪；verl 集成待做）
- [ ] W5 奖励消融实验
- [ ] W6 真实四组模型评测（基座/SFT/GRPO/教师）
- [ ] W7 MCP Server + Agent Runtime + 技术报告
- [ ] W8 README 完善、权重发布、推广

## License

[Apache-2.0](LICENSE)
