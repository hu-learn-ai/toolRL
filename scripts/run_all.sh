#!/usr/bin/env bash
# ToolRL-Lite 一键复现（design_zh.md §7.1）
#
# 用法：
#   bash scripts/run_all.sh                     # 顺序执行全部阶段，产物存在时跳过
#   bash scripts/run_all.sh --only synth,sft    # 只跑指定阶段（逗号分隔）
#   bash scripts/run_all.sh --skip synth        # 跳过指定阶段（可重复）
#   bash scripts/run_all.sh --force             # 忽略已有产物，强制重跑
#
# 环境变量：OUT_DIR / N_PER_ENV / N_TASKS / SEED / PORT / SERVE_MODE（均有默认值）
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_DIR="${OUT_DIR:-data_out}"
N_PER_ENV="${N_PER_ENV:-2500}"
N_TASKS="${N_TASKS:-400}"
SEED="${SEED:-42}"
PORT="${PORT:-8000}"

# 导出给 bash -c 子阶段（eval/serve 在子 shell 内引用 $SEED 等）
export OUT_DIR N_PER_ENV N_TASKS SEED PORT

ONLY=""
declare -a SKIP=()
FORCE=0

usage() {
    sed -n '2,10p' "$0"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --only) ONLY="$2"; shift 2 ;;
        --skip) SKIP+=("$2"); shift 2 ;;
        --force) FORCE=1; shift ;;
        -h|--help) usage ;;
        *) echo "未知参数: $1"; usage ;;
    esac
done

stage_enabled() {
    local name="$1"
    if [[ -n "$ONLY" ]]; then
        [[ ",$ONLY," == *",$name,"* ]]
        return
    fi
    for s in "${SKIP[@]}"; do
        [[ "$s" == "$name" ]] && return 1
    done
    return 0
}

have() { [[ -e "$1" ]]; }

run_stage() {
    # run_stage <名称> <产物路径> <命令...>；产物路径为空则每次执行
    local name="$1" artifact="$2"
    shift 2
    stage_enabled "$name" || { echo "[run_all] 跳过阶段 $name"; return 0; }
    if [[ -n "$artifact" ]] && [[ "$FORCE" != "1" ]] && have "$artifact"; then
        echo "[run_all] $name 产物已存在，跳过（--force 重跑）: $artifact"
        return 0
    fi
    echo "===== [run_all] 阶段: $name ====="
    "$@"
}

echo "===== ToolRL-Lite run_all ====="
echo "OUT_DIR=$OUT_DIR N_PER_ENV=$N_PER_ENV N_TASKS=$N_TASKS SEED=$SEED FORCE=$FORCE"

run_stage env_check "" bash -c '
    python -c "import sys; assert sys.version_info >= (3, 10), \"需要 Python >= 3.10\""
    python -c "import envs, data, train, eval, serving; print(\"[env_check] 核心包导入 OK\")"
    python -c "import fastapi, uvicorn" 2>/dev/null \
        && echo "[env_check] fastapi/uvicorn OK" \
        || echo "[env_check] 提示: 未装 fastapi/uvicorn（可选：pip install -e .[env]）"
    python -c "import mcp" 2>/dev/null \
        && echo "[env_check] mcp OK" \
        || echo "[env_check] 提示: 未装 mcp（可选：pip install -e .[mcp]）"
'

run_stage gen_tasks "$OUT_DIR/tasks.jsonl" \
    python scripts/gen_tasks.py --n "$N_TASKS" --seed "$SEED" --out "$OUT_DIR/tasks.jsonl"

run_stage synth "$OUT_DIR/sft_trajectories.jsonl" \
    python scripts/gen_sft.py --n-per-env "$N_PER_ENV" --seed "$SEED" --out "$OUT_DIR"

run_stage sft "checkpoints/sft" \
    python -m train.sft --data "$OUT_DIR/sft_trajectories.jsonl" --out checkpoints/sft

run_stage grpo "checkpoints/grpo" \
    python -m train.grpo --model checkpoints/sft --data "$OUT_DIR/rl_pool.jsonl" --out checkpoints/grpo

run_stage eval "eval_out/report.md" bash -c '
    if [[ -d checkpoints/sft || -d checkpoints/grpo ]]; then
        python -m eval --base "${BASE_MODEL:-Qwen/Qwen3-0.6B}" \
            --sft checkpoints/sft --grpo checkpoints/grpo \
            --seed "$SEED" --out eval_out/report.md
    else
        echo "[run_all] 未找到训练 checkpoint，改用 --mock 冒烟评测"
        python -m eval --mock --size 40 --seed "$SEED" --out eval_out/report.md
    fi
'

run_stage serve "" bash -c '
    if [[ "${SERVE_MODE:-demo}" == "mcp" ]]; then
        echo "[run_all] 启动 MCP Server（stdio transport，阻塞，Ctrl-C 退出）"
        exec python -m serving --mcp
    else
        echo "[run_all] 演示最小 Agent Runtime（GoldTeacher，§6.2）"
        python -m serving --task-seed "$SEED" --task-idx 0
    fi
'
