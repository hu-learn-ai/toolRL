"""Markdown 评测报告（design_zh.md §5.3）。

输出三部分：四组对比表（§5.2 六指标 + 加权奖励均值）、分维度端到端成功率文本柱状图、
失败 case 清单。纯文本渲染，无 matplotlib 依赖，可直接落盘为 ``report.md``。
"""

from __future__ import annotations

from .runner import GroupResult

# (显示名, EvalMetrics 字段, 是否百分率)
_METRIC_ROWS: list[tuple[str, str, bool]] = [
    ("格式合规率", "format_rate", True),
    ("工具选择准确率", "tool_acc", True),
    ("参数正确率", "param_acc", True),
    ("端到端成功率", "success_rate", True),
    ("平均步数", "avg_steps", False),
    ("平均 token", "avg_tokens", False),
    ("加权奖励均值", "reward_mean", False),
]

_BAR_WIDTH = 24
_MAX_FAILURES = 30  # 每组最多列出的失败 case 数，避免报告过长


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _fmt(x: float, is_pct: bool) -> str:
    return _pct(x) if is_pct else f"{x:.2f}"


def _bar(value: float) -> str:
    filled = round(max(0.0, min(1.0, value)) * _BAR_WIDTH)
    return "█" * filled + "░" * (_BAR_WIDTH - filled)


def _dims(results_by_group: dict[str, GroupResult]) -> list[str]:
    dims: list[str] = []
    seen: set[str] = set()
    for gr in results_by_group.values():
        for r in gr.results:
            d = r.dim or r.category
            if d not in seen:
                seen.add(d)
                dims.append(d)
    return dims


def render_markdown(
    results: dict[str, GroupResult], title: str = "ToolRL-Lite 评测报告"
) -> str:
    names = list(results.keys())
    lines: list[str] = [f"# {title}", ""]

    # ---- 四组对比表（§5.2） ----
    lines.append("## 一、四组模型指标对比（§5.2）")
    lines.append("")
    lines.append("| 指标 | " + " | ".join(names) + " |")
    lines.append("|" + "---|" * (len(names) + 1))
    for label, attr, is_pct in _METRIC_ROWS:
        cells = [_fmt(getattr(results[n].metrics, attr), is_pct) for n in names]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")

    # ---- 分维度柱状图（端到端成功率） ----
    lines.append("## 二、分维度端到端成功率")
    lines.append("")
    dims = _dims(results)
    # 表头：维度 + 每组
    lines.append("| 维度 | " + " | ".join(names) + " |")
    lines.append("|" + "---|" * (len(names) + 1))
    for dim in dims:
        cells = []
        for n in names:
            gr = results[n]
            ds = [r for r in gr.results if (r.dim or r.category) == dim]
            rate = sum(1 for r in ds if r.success) / len(ds) if ds else 0.0
            cells.append(f"{_bar(rate)} {_pct(rate)}")
        lines.append(f"| {dim} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("（柱 = 端到端成功率，█ 为成功、░ 为失败）")
    lines.append("")

    # ---- 失败 case 清单 ----
    lines.append("## 三、失败 case 清单")
    lines.append("")
    for n in names:
        fails = results[n].failures
        lines.append(f"### {n}（{len(fails)} / {len(results[n].results)} 失败）")
        lines.append("")
        if not fails:
            lines.append("无失败 case。")
            lines.append("")
            continue
        lines.append("| task_id | 维度 | 指令 | 加权奖励 |")
        lines.append("|---|---|---|---|")
        for r in fails[:_MAX_FAILURES]:
            instr = r.instruction.replace("\n", " ").replace("|", "\\|")
            if len(instr) > 40:
                instr = instr[:40] + "…"
            lines.append(f"| {r.task_id} | {r.dim or r.category} | {instr} | {r.reward:.3f} |")
        if len(fails) > _MAX_FAILURES:
            lines.append(f"（其余 {len(fails) - _MAX_FAILURES} 条省略）")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


__all__ = ["render_markdown"]