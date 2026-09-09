# ToolRL-Lite 部署文档（阿里云 GPU 实例）

> 把训练后的 SFT / GRPO_v3 模型部署为 HTTP API，供 curl / 前端 / MCP 客户端调用。
>
> **本文档对应 commit**：W7（serving 模块）；完整技术报告（`docs/report_zh.md`）待整理，现有评测数据见 §9。

## 1. 架构

```
┌──────────────────────────────────────────────────────────────┐
│ 阿里云 GPU 实例（按量付费，如 P100/T4/A10）                      │
│                                                              │
│   ┌────────────────────────────────────────┐                │
│   │ Docker container: toolrl-serving         │                │
│   │   ├─ uvicorn :8000                       │                │
│   │   ├─ /health  /tools  /chat/{model}      │                │
│   │   ├─ checkpoints/sft      (model.safetensors)            │
│   │   ├─ checkpoints/grpo_v3  (model.safetensors)            │
│   │   └─ ToolRegistry + run_agent (FastAPI 主循环)            │
│   └────────────────────────────────────────┘                │
│           ↑                                                  │
│           │ HTTP（curl / 前端 / Claude Desktop MCP 桥接）      │
└───────────┼──────────────────────────────────────────────────┘
            ↓
    宿主机防火墙/安全组 :8000 → 公网或内网
```

- 镜像基于官方 `python:3.11-slim`（**不要**用 `registry.cn-hangzhou.aliyuncs.com/library/python:3.11-slim`——阿里云 ACR 不透明代理 Docker Hub 公共命名空间，`library/python` 无权限会 404；改走 daemon.json 的 `registry-mirrors` 加速器拉官方镜像）。
- 启动时通过 `nvidia-container-toolkit` 挂 GPU；无需把 CUDA 编进镜像。
- `checkpoints/` 通过 bind mount 挂入，**不进镜像**（避免每次换权重都 rebuild）。
- Dockerfile 已内置两个关键修复（见 §8 对应行）：`transformers==5.14.1` pin + `gcc/g++/make/libc6-dev`（GPU 首推理 Triton JIT 编译需要）。

## 2. 前置要求（GPU 实例）

```bash
# 1. NVIDIA 驱动（按 GPU 型号装；这里假设已装好，nvidia-smi 可用）
nvidia-smi

# 2. Docker + nvidia-container-toolkit
# ⚠️ 阿里云上不要用官方模板（nvidia.github.io 被墙）：
#    官方 nvidia-container-toolkit 源在阿里云镜像站的正确前缀是 libnvidia-container
#    （https://mirrors.aliyun.com/libnvidia-container/... 返回 200），且镜像站没有
#    nvidia.github.io 那份 .list 模板，必须手写 flat repo（无 Release 文件，404 不影响安装）：
curl -fsSL https://mirrors.aliyun.com/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] \
https://mirrors.aliyun.com/libnvidia-container/stable/deb/amd64 /" | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Docker Hub 直连超时的话，配 registry-mirrors 加速器（实测存活：docker.m.daocloud.io /
# docker.1panel.live / dockerproxy.net），注意合并进 daemon.json、别覆盖 nvidia runtime 配置

# 验证：docker run --rm --gpus all nvidia/cuda:12.1.0-base nvidia-smi
```

## 3. 上传项目

GPU 实例不需要装 git；只把源码 + checkpoints 拷过去即可。

```bash
# 在本机（Windows PowerShell）执行：
# 路径换成你的 SSH 别名 / IP
SERVER=alicloud-gpu   # ~/.ssh/config 里配的别名

# 3.1 源码（不含 checkpoints 与 data_out）
rsync -avz --exclude='checkpoints' --exclude='data_out*' --exclude='.git' \
    --exclude='__pycache__' --exclude='*.pyc' \
    ./ ${SERVER}:/opt/toolrl-lite/

# 3.2 checkpoints 单独传（避免污染源码）
scp -r checkpoints/sft      ${SERVER}:/opt/toolrl-lite/checkpoints/
scp -r checkpoints/grpo_v3  ${SERVER}:/opt/toolrl-lite/checkpoints/

# 验证
ssh ${SERVER} "ls /opt/toolrl-lite/checkpoints/sft && du -sh /opt/toolrl-lite/checkpoints/*"
# 预期：config.json + model.safetensors + tokenizer.*；每个 ~1.2GB
```

> ⚠️ **scp 慢的话**：用 `rsync --partial --progress` 可续传；或者先 `tar czf checkpoints.tar.gz checkpoints/` 再传单文件，再在服务器 `tar xzf`。

## 4. 启动

```bash
ssh ${SERVER}
cd /opt/toolrl-lite

# 4.1 复制环境变量（首次）
cp .env.example .env
# 按需编辑：HOST_PORT / SERVING_WARMUP

# 4.2 构建 + 启动
docker compose up -d --build
# 首跑约 5-10 分钟（pip 装 torch + transformers + 镜像层缓存）

# 4.3 看日志（确认模型加载完毕）
docker compose logs -f serving
# 预期结尾：Uvicorn running on http://0.0.0.0:8000
# 若 SERVING_WARMUP=1：还会看到两个 "warmup sft from ..."  "warmup grpo_v3 from ..."
```

## 5. 验证

**5.1 健康检查**：

```bash
curl -s http://<GPU实例公网IP>:8000/health | jq
# 预期：
# {
#   "status": "ok",
#   "tools": ["weather.query", "weather.forecast", ... 10 个],
#   "checkpoints": {"sft": "...", "grpo_v3": "..."}
# }
```

**5.2 列出工具**：

```bash
curl -s http://<IP>:8000/tools | jq '.[0:2]'
```

**5.3 端到端对话**（teacher 模式不需要 GPU，首跑建议先用这个）：

```bash
curl -s -X POST http://<IP>:8000/chat/teacher \
    -H "Content-Type: application/json" \
    -d '{"instruction":"查下周三从北京到上海的航班，并预订第一个航班，乘客 李四"}' | jq
# 预期：answer 字段非空，error=null，steps≈3
```

**5.4 真实模型**（需要 GPU 已挂载；首次会触发权重 IO，约 10-30 秒）：

```bash
# SFT 模型
time curl -s -X POST http://<IP>:8000/chat/sft \
    -H "Content-Type: application/json" \
    -d '{"instruction":"帮我查一下深圳未来3天的天气","max_steps":4,"seed":42}' | jq

# GRPO_v3 模型
time curl -s -X POST http://<IP>:8000/chat/grpo_v3 \
    -H "Content-Type: application/json" \
    -d '{"instruction":"帮我查一下深圳未来3天的天气","max_steps":4,"seed":42}' | jq
```

> **首次慢的原因**：两层叠加 —— ① `HuggingFaceTeacher._load` 第一次调用 `generate()` 时才读权重（0.6B + bfloat16 ≈ 1.2GB 读盘，10-30 秒）；② torch 原生算子（Qwen3 RoPE 的 `bmm_outer_product`）首次 forward 时 Triton 要现场 JIT 编译 CUDA 内核（另加 30-90 秒，需要容器里有 gcc，见 §8）。两者都是一次性成本：权重由 `SERVING_WARMUP=1` 预热消除，Triton 内核缓存已用 compose 命名卷 `triton_cache:/root/.triton` 持久化（重建容器也不丢）。

## 6. 公网暴露（可选）

阿里云 GPU 实例默认安全组是**关闭所有入站**的。三种暴露方式：

| 方式 | 适用场景 | 工作量 |
|---|---|---|
| 安全组放行 8000 | 内网/测试用 | 1 分钟 |
| Caddy/Nginx 反向代理 + HTTPS + 域名 | 公开 demo、简历作品 | 30 分钟 |
| 不暴露，仅 SSH 端口转发 | 临时演示 | 0（ssh -L 本地端口:localhost:8000） |

**安全组放行**（Web 控制台 → 实例 → 安全组 → 入方向）：

```
协议：TCP    端口：8000    源：0.0.0.0/0    描述：toolrl-serving
```

**Caddy HTTPS**（最小可用）：

```caddyfile
# /etc/caddy/Caddyfile
toolrl.example.com {
    reverse_proxy 127.0.0.1:8000
}
```

## 7. 常用运维命令

```bash
# 看实时日志
docker compose logs -f serving

# 看 GPU 占用
nvidia-smi
docker exec toolrl-serving nvidia-smi

# 重启（不重建镜像）
docker compose restart serving

# 换 checkpoint 后重启
# 本机 rsync checkpoints/ → 服务器 → 服务器 docker compose restart serving
# （无需 rebuild）

# 进入容器调试
docker compose exec serving bash
python -c "from serving.api import app; print('ok')"

# 停服
docker compose down

# 清理（删容器 + 镜像）
docker compose down --rmi all
```

## 8. 故障排查

| 现象 | 原因 | 解决 |
|---|---|---|
| `docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]` | 缺 nvidia-container-toolkit | 见 §2 安装 |
| `/health` 返回 200 但 tools 列表空 | `ToolRegistry()` 初始化失败 | 查日志，多半是 `envs.api_sandbox.tools` 导入报错 |
| `/chat/sft` 报 `RuntimeError: Found no NVIDIA driver on your system` | 容器没拿到 GPU | 加 `--gpus all`（compose 已配），或重启 docker daemon |
| `/chat/sft` 首次请求 5 分钟后超时 | 0.6B 模型读盘慢 + 网络配置问题 | 设 `SERVING_WARMUP=1`；检查磁盘是否 SSD 而非 NFS |
| `/chat/grpo_v3` 报 `checkpoint not found` | 路径错或 scp 没传完 | `ls /app/checkpoints/grpo_v3/` 确认 safetensors 存在 |
| 国内服务器报 `huggingface.co ... 401` | HF 没切镜像 | `.env` 里 `HF_ENDPOINT=https://hf-mirror.com`；已默认设置 |
| 容器内存持续涨直到 OOM | 多请求并发 + 没限 batch | 单进程串行已限；如并发高把 `--workers 1` 改成 `--workers 1 --limit-concurrency 4` |
| `/chat/{sft,grpo_v3}` 500，日志 `AttributeError: 'list' object has no attribute 'keys'`（`tokenization_utils_base.py` 内） | **transformers 4.5x~4.57 全系 bug**：Qwen3 checkpoint 的 `tokenizer_config.json` 里 `extra_special_tokens` 是 list，4.5x 内部按 dict 处理。实测 4.52.4/4.57.6 均崩、**5.14.1 正常** | Dockerfile 已 pin `transformers==5.14.1`（独立层，避免 torch 大层缓存失效）。容器内应急：`pip install "transformers==5.14.1"` + restart |
| 日志报 `ImportError: requires the protobuf library`，但 protobuf 已装 | **误导性二次抛错**：transformers 的 `except import_protobuf_decode_error():` 在捕获异常时会二次执行导入检查，把真根因顶掉替换成这个假 ImportError | 不要信这行；`docker compose logs --tail 300` 往上翻真 Traceback（结尾 `xxxError:` 那行才是根因） |
| 首次请求 500，日志结尾 `RuntimeError: Failed to find C compiler` | torch 原生算子（`bmm_outer_product` 等）首次 forward 走 Triton JIT 编译，`python:3.11-slim` 无 C 编译器 | Dockerfile 已装 `gcc g++ make libc6-dev`（`libc6-dev` 提供 stdio.h，`--no-install-recommends` 下须显式加）。容器内应急：`apt-get update && apt-get install -y gcc g++ make libc6-dev` |

## 9. 简历 / 推广素材

部署成功后可直接复用以下三个端点的 curl 截图作为「线上 demo」证据：

- `GET /health`：返回 200 + tools 清单（证明服务在线）
- `POST /chat/teacher`：无 GPU 即可演示
- `POST /chat/grpo_v3`：真模型对比 SFT（成功率 52.5% → 57.5%）

现有四组评测报告见 `artifacts/eval_out/report_v3.md`（GRPO_v3 成功率 57.5%，对应上方 SFT 52.5% → GRPO_v3 57.5%）；完整技术报告待整理为 `docs/report_zh.md`。