# ToolRL-Lite Serving 部署镜像
#
# 基础镜像：官方 python:3.11-slim（走 daemon.json 的 registry-mirrors 加速器拉取）
# 用途：把 checkpoints/sft 与 checkpoints/grpo_v3 跑成 HTTP API（FastAPI :8000）
# 启动 GPU：docker run --gpus all ...（自动挂 NVIDIA 驱动到容器）

FROM python:3.11-slim AS base

LABEL maintainer="toolrl-lite" \
      description="ToolRL-Lite 模型推理服务（SFT + GRPO_v3）"

# 阿里云镜像加速（pip 官方源在国内很慢，临时切阿里源）
ENV PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
    PIP_TRUSTED_HOST=mirrors.aliyun.com \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_ENDPOINT=https://hf-mirror.com \
    HF_HUB_DISABLE_XET=1 \
    TRANSFORMERS_OFFLINE=1

# 工具：curl 用于 HEALTHCHECK；git/wget 备用
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先 copy 依赖清单，装好再 copy 源码（利用 docker 层缓存）
COPY pyproject.toml README.md LICENSE ./
COPY envs ./envs
COPY data ./data
COPY serving ./serving

# eval/model.py 也在 copy 列表里（HuggingFaceTeacher 复用）
COPY eval ./eval

# GPU 侧依赖（torch + transformers）；不放 pyproject 里因为很多镜像不需要
# CPU 也兼容，体积大约 +1.5GB
RUN pip install --no-cache-dir \
        "torch>=2.5,<3.0" \
        "transformers>=4.45,<5" \
        "accelerate>=0.34" \
    && pip install --no-cache-dir -e ".[env,synth,mcp]"

# transformers 加载 Qwen tokenizer 依赖 protobuf（可选依赖，slim 镜像默认没有）
# 独立成层：避免改动上一行触发 torch 大层缓存失效（否则要重下 2G+ 依赖）
RUN pip install --no-cache-dir protobuf sentencepiece

# transformers 4.5x~4.57 加载 Qwen3 的 tokenizer_config.json（extra_special_tokens 为
# list 格式）时崩 "AttributeError: 'list' object has no attribute 'keys'"（内部按 dict
# 处理）。实测 4.52.4 同样崩、5.14.1 正常 → 独立成层锁 5.14.1（不动上面大层以免
# torch 层缓存失效重下 2G+）
RUN pip install --no-cache-dir "transformers==5.14.1"

# GPU 首次 forward 时 torch 原生算子（如 Qwen3 RoPE 的 bmm_outer_product）会用 Triton
# 现场 JIT 编译 CUDA 内核，宿主必须能调 cc 编译 C shim；python:3.11-slim 默认无 gcc，
# 否则报 "RuntimeError: Failed to find C compiler" → 独立层装编译工具链。
# libc6-dev 提供 stdio.h/stddef.h 等头文件（--no-install-recommends 下不自动装）。
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ make libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# checkpoints 不打进镜像，挂载更灵活（避免每次改权重都要 rebuild）
# 镜像内默认占位目录；启动时由 -v /opt/toolrl-lite/checkpoints:/app/checkpoints:ro 覆盖
RUN mkdir -p /app/checkpoints/sft /app/checkpoints/grpo_v3

EXPOSE 8000

# HEALTHCHECK：用 /health 探活（注意：仅当模型已加载完毕才会返回 200，否则也算 ok）
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

# 默认启动 FastAPI HTTP 服务
# 关键环境变量（推荐用 docker-compose / -e 覆盖）：
#   CHECKPOINT_SFT        默认 checkpoints/sft
#   CHECKPOINT_GRPO_V3    默认 checkpoints/grpo_v3
#   SERVING_TEMPERATURE   默认 0.7
#   SERVING_WARMUP        设为 1 启动时预热两个模型
#   SERVING_MAX_NEW_TOKENS 默认 1024
ENTRYPOINT ["python", "-m", "serving", "--http", "--host", "0.0.0.0", "--port", "8000"]