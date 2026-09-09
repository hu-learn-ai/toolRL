# ToolRL-Lite

> Train a small Qwen3 model into a reliable tool-calling agent: a full pipeline of verifiable environment → data synthesis → SFT cold-start → GRPO RL → evaluation → deployment.

[中文](README.md) · [Project Charter](ToolRL-Lite项目立项文档.md) · [Technical Design](docs/design_zh.md)

## What is this?

ToolRL-Lite is an open-source personal project for agent/RL positions: starting from Qwen3-0.6B/1.7B, it uses SFT cold-start plus GRPO reinforcement learning on a self-built verifiable tool-calling environment to make small models call tools reliably.

Highlights:

- **Verifiable rewards**: all rewards come from programmatic judging (no LLM/human judge); four dimensions (format / correctness / answer / steps) with partial credit;
- **End-to-end**: environment → data synthesis → SFT → GRPO → eval → deployment, reproducible in one repo;
- **Deterministic**: fixed-seed task generation, deterministic mock APIs, idempotent synthesis and evaluation;
- **Lightweight**: full-parameter 0.6B training fits on a single 4090/A800; evaluation has a no-GPU mock mode.

## Pipeline

```
[Task Env] → [Data Synthesis] → [SFT] → [GRPO RL]
verifiable    distilled         reward-driven
sandbox       trajectories      optimization
     ↑                            ↓
[Evaluation] ← [curves / checkpoints] ←─────┘
     ↓
[MCP Server + Agent Runtime] → open-source delivery
```

## Repository layout

```
├── envs/                # Verifiable tool environments (api_sandbox / text2sql)
├── data/                # Data synthesis (teacher, rejection sampling, resume)
├── train/
│   ├── sft/             # SFT cold-start (0.6B full / 1.7B LoRA)
│   └── grpo/            # GRPO RL (reward functions + TRL/verl entry)
├── eval/                # Self-built benchmark + 4-group comparison report
├── serving/             # MCP Server + Agent Runtime + FastAPI HTTP
├── scripts/             # gen_tasks.py / gen_sft.py / run_all.sh
└── .github/workflows/   # CI (pytest + ruff + eval smoke)
```

## Quickstart

### 0. Environment (Python ≥ 3.10)

```bash
conda create -n toolrl-lite python=3.11 -y
conda activate toolrl-lite
pip install -e ".[env,dev]"
```

GPU-side deps (torch / transformers / peft / datasets / trl) are added when you reach training.

### 1. Configure the teacher LLM (optional, for synthesis)

```bash
cp .env.example .env   # fill in TEACHER_API_KEY
```

### 2. One-click reproduction (Linux GPU box)

```bash
bash scripts/run_all.sh                    # full pipeline, skips existing artifacts
bash scripts/run_all.sh --only synth,sft   # run selected stages
```

### 3. Step by step

```bash
# Generate the task pool (fixed seed)
python scripts/gen_tasks.py --n 400 --seed 42

# Data synthesis (teacher distillation + rejection sampling → SFT trajectories + RL pool)
python scripts/gen_sft.py --n-per-env 2500 --out data_out

# SFT cold-start
python -m train.sft --data data_out/sft_trajectories.jsonl --out checkpoints/sft

# GRPO RL (starts from the SFT checkpoint)
python -m train.grpo --model checkpoints/sft --data data_out/rl_pool.jsonl

# Evaluation: no-GPU smoke / real model comparison
python -m eval --mock --size 40 --out eval_out/report.md
python -m eval --base Qwen/Qwen3-0.6B --sft checkpoints/sft --grpo checkpoints/grpo

# Start the mock API sandbox (explicitly bind 127.0.0.1, local-only)
uvicorn envs.api_sandbox.server:app --host 127.0.0.1 --port 8000

# Start the MCP Server (stdio)
python -m serving --mcp

# Start the FastAPI HTTP service (main cloud-deployment entry)
python -m serving --http --host 0.0.0.0 --port 8000
# See docs/deploy_zh.md
```

> ⚠️ **Security note**: the mock API sandbox (`envs/api_sandbox/server.py`) and the MCP Server
> (`python -m serving --mcp`) are **for local training/debugging only**. They have no
> authentication or network isolation — do not bind them to a public address (default listens
> on `127.0.0.1`) or expose the port to the public internet.

### 4. Tests

```bash
pytest
```

## Status (2026-09-01)

| Module | Status | Notes |
|---|---|---|
| envs/ | ✅ Done | api_sandbox (10 mock tools) + text2sql (3 schemas), programmatic graded judging, FastAPI server |
| data/ | ✅ Done | GoldTeacher/LLMTeacher, rejection sampling, resume, call budget (needs real API data) |
| train/sft/ | ✅ Done | 0.6B full / 1.7B LoRA, assistant loss masking, train/eval metrics |
| train/grpo/ | 🚧 WIP | reward functions + TRL entry ready; verl multi-turn integration pending |
| eval/ | ✅ Done | 400-task benchmark, 4-group report, no-GPU mock mode |
| serving/ | ✅ Done (code + docs; real-model validated) | MCP stdio Server + minimal Agent Runtime + FastAPI HTTP (`/chat/{model}`) + Dockerfile + docker-compose + `docs/deploy_zh.md`; teacher-mode end-to-end smoke passed locally; real-model inference for both `/chat/sft` and `/chat/grpo_v3` was validated via curl on an Aliyun pay-as-you-go GPU instance on 2026-09-09 (each completed a 3-step run). The instance has since been released, so no live service is currently running; redeploy by following `docs/deploy_zh.md` |
| Engineering | ✅ Done | git, CI (pytest+ruff+smoke), bilingual README, Apache-2.0 |

## Roadmap

- [x] W1 repo skeleton + engineering
- [x] W2 task environments v1 (both envs + unit tests)
- [ ] W3 data synthesis at scale + SFT training (code ready, needs real data)
- [ ] W4 first GRPO run (TRL path ready; verl integration pending)
- [ ] W5 reward ablation experiments
- [ ] W6 real 4-group evaluation (base/SFT/GRPO/teacher)
- [ ] W7 MCP Server + Agent Runtime + tech report
- [ ] W8 README polish, model release, promotion

## License

[Apache-2.0](LICENSE)
