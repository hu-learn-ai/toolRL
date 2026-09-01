"""评测 CLI（design_zh.md §7.1 eval 阶段）。

四组模型：基座（--base）/ SFT（--sft）/ GRPO（--grpo）为本地 checkpoint，教师组
（--teacher 缺省开启）用 GoldTeacher 作为程序化上限参照（§5.3）。

用法：

    python -m eval --base checkpoints/base --sft checkpoints/sft \\
                   --grpo checkpoints/grpo --size 400 --seed 42 --out eval_out/report.md

无 GPU 环境冒烟：

    python -m eval --mock --size 40 --out eval_out/report.md

入参只有 model_path / --seed / --size（§5.3 可复现性保证）：同 seed 两次运行结果一致。
"""

from __future__ import annotations

import argparse
import os
import sys

from data.teacher import GoldTeacher, Teacher
from envs.api_sandbox import ApiSandboxEnv

from .benchmark import DEFAULT_SIZE, build_benchmark
from .model import load_model_factory
from .report import render_markdown
from .runner import run_all


class _BadTeacher(Teacher):
    """无 GPU 冒烟用：输出不含任何标签的乱文本，必判失败。"""

    def generate(self, messages: list[dict]) -> str:
        return "没有 tool_call 也没有 answer 的乱输出"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="评测四组模型（§5）")
    p.add_argument("--base", default=None, help="基座模型 checkpoint 路径")
    p.add_argument("--sft", default=None, help="SFT checkpoint 路径")
    p.add_argument("--grpo", default=None, help="GRPO checkpoint 路径")
    p.add_argument("--teacher", action="store_true", default=True,
                   help="加入教师组（GoldTeacher 上限参照，缺省开启）")
    p.add_argument("--no-teacher", dest="teacher", action="store_false",
                   help="关闭教师组")
    p.add_argument("--size", type=int, default=DEFAULT_SIZE, help="benchmark 任务数（默认 400）")
    p.add_argument("--seed", type=int, default=42, help="随机种子（可复现性）")
    p.add_argument("--out", default="eval_out/report.md", help="报告输出路径")
    p.add_argument("--mock", action="store_true",
                   help="无 GPU 冒烟：用 BadTeacher 顶替三组模型")
    args = p.parse_args(argv)

    groups: dict = {}
    if args.mock:
        bad = lambda task: _BadTeacher()  # noqa: E731
        for name, path in (("基座", args.base), ("SFT", args.sft), ("GRPO", args.grpo)):
            groups[name] = bad
    else:
        if args.base:
            groups["基座"] = load_model_factory(args.base, seed=args.seed)
        if args.sft:
            groups["SFT"] = load_model_factory(args.sft, seed=args.seed)
        if args.grpo:
            groups["GRPO"] = load_model_factory(args.grpo, seed=args.seed)
    if args.teacher:
        groups["教师"] = GoldTeacher  # GoldTeacher 本身即 TeacherFactory(Task -> GoldTeacher)

    if not groups:
        raise SystemExit("未指定任何评测组：请给 --base/--sft/--grpo 或 --mock")

    tasks = build_benchmark(args.seed, args.size)
    env = ApiSandboxEnv()
    results = run_all(groups, tasks, env)

    md = render_markdown(results)
    out = args.out
    if out and out != "-":
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"[eval] 报告已写入 {out}")
    else:
        # 输出到 stdout 走 UTF-8 字节流，避免 GBK 控制台无法编码柱状图字符
        sys.stdout.buffer.write(md.encode("utf-8"))
        sys.stdout.buffer.flush()

    # 控制台摘要
    print(f"[eval] 任务数={len(tasks)} seed={args.seed} 组={list(results)}")
    for name, gr in results.items():
        m = gr.metrics
        print(f"  {name}: 格式={m.format_rate:.2f} 工具={m.tool_acc:.2f} "
              f"参数={m.param_acc:.2f} 成功={m.success_rate:.2f} 步数={m.avg_steps:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())