"""分级判定（design_zh.md §2.2 / §2.4 / §4.1）。

对一条 rollout 轨迹做程序化打分，输出 R_format / R_correct / R_answer / R_steps
四个原始分项（不做加权，由训练端组装 R_total）。``parse_turn`` 是 Qwen3 输出格式
（<think> + <tool_call> / <answer>）的共用解析器：SFT 阶段的数据质检（§4.1）与 RL
奖励函数（§2.4）复用同一份解析/判定代码，保证"训练打分与 SFT 质检口径一致"。

轨迹约定：``list[dict]``，每个元素是模型某一回合的原始输出 ``{"raw": str}``
（也兼容直接传 ``str``）。回合顺序 = 助手回合顺序，最后一回合应为 <answer>。

已知简化（MVP 范围内）：
- 答案程序化等价采用"事实子集"判定，事实 = ASCII 编号/数字/时间（``CA123``、
  ``FL123456``、``09:00``、``25``），中文实体（城市/酒店名）不参与比对；
- 参数比对只对 gold 逐 key 计分，多余/额外参数 key 不额外扣分。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..base_env import JudgeResult
from ..task_schema import Task, ToolParameterSchema
from .common import validate_params
from .tools import TOOLS

# ---------------------------------------------------------------------------
# 输出格式解析（Qwen3 约定，§4.1）
# ---------------------------------------------------------------------------

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


@dataclass
class ParsedTurn:
    kind: str  # "tool_call" | "answer" | "invalid"
    tool_name: str | None = None
    arguments: dict | None = None
    answer: str | None = None
    error: str | None = None


def parse_turn(raw: str) -> ParsedTurn:
    """解析一回合原始输出；合法 tool_call / answer 之外一律 ``invalid``。"""
    if not isinstance(raw, str):
        return ParsedTurn("invalid", error="raw 必须是字符串")

    tc_open = raw.count("<tool_call>")
    tc_close = raw.count("</tool_call>")
    ans_open = raw.count("<answer>")
    ans_close = raw.count("</answer>")

    # 一个回合只允许一种动作块
    if (tc_open or tc_close) and (ans_open or ans_close):
        return ParsedTurn("invalid", error="同一回合不能同时含 tool_call 与 answer")

    if tc_open or tc_close:
        if tc_open != 1 or tc_close != 1:
            return ParsedTurn("invalid", error="<tool_call> 标签未配对")
        m = _TOOL_CALL_RE.search(raw)
        if m is None:
            return ParsedTurn("invalid", error="<tool_call> 内容缺失")
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError as e:
            return ParsedTurn("invalid", error=f"tool_call JSON 非法: {e}")
        if not isinstance(obj, dict):
            return ParsedTurn("invalid", error="tool_call 必须是 JSON 对象")
        name = obj.get("name")
        arguments = obj.get("arguments", {})
        if not isinstance(name, str) or not name:
            return ParsedTurn("invalid", error="tool_call 缺 name")
        if not isinstance(arguments, dict):
            return ParsedTurn("invalid", error="arguments 必须是对象")
        return ParsedTurn("tool_call", tool_name=name, arguments=arguments)

    if ans_open or ans_close:
        if ans_open != 1 or ans_close != 1:
            return ParsedTurn("invalid", error="<answer> 标签未配对")
        m = _ANSWER_RE.search(raw)
        if m is None:
            return ParsedTurn("invalid", error="<answer> 内容缺失")
        return ParsedTurn("answer", answer=m.group(1).strip())

    return ParsedTurn("invalid", error="缺少 tool_call 或 answer 块")


# ---------------------------------------------------------------------------
# 值等价与答案事实比对
# ---------------------------------------------------------------------------

_FULLWIDTH = str.maketrans(
    "０１２３４５６７８９：ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ",
    "0123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
)


def _norm_text(x) -> str:
    if isinstance(x, float) and x.is_integer():
        x = int(x)
    return str(x).translate(_FULLWIDTH).strip().lower()


def _value_equal(a, b) -> bool:
    """参数值等价：列表顺序无关、数值 int/float 等价、其余字符串归一化。"""
    if isinstance(a, list) and isinstance(b, list):
        return sorted(_norm_text(v) for v in a) == sorted(_norm_text(v) for v in b)
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    return _norm_text(a) == _norm_text(b)


_FACT_RE = re.compile(r"[A-Za-z]{1,}\d{3,}|\d{1,2}:\d{2}|\d+(?:\.\d+)?")


def _norm_fact(f: str) -> str:
    if re.fullmatch(r"\d{1,2}:\d{2}", f):  # 时间去前导零：09:00 → 9:00
        hh, mm = f.split(":")
        return f"{int(hh)}:{mm}"
    if re.fullmatch(r"\d+(?:\.\d+)?", f):  # 数字规整：3.0 → 3
        n = float(f)
        return str(int(n)) if n.is_integer() else str(n)
    return f.upper()  # 编号大写：ca123 → CA123


def _extract_facts(text: str) -> set[str]:
    return {_norm_fact(m) for m in _FACT_RE.findall(text or "")}


# ---------------------------------------------------------------------------
# 判定入口
# ---------------------------------------------------------------------------


def _raw_of(item) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return item.get("raw", "")
    return ""


def _schema_of(name: str) -> ToolParameterSchema | None:
    tool = TOOLS.get(name)
    return tool.parameters if tool is not None else None


def judge(task: Task, trajectory: list) -> JudgeResult:
    """程序化分级判定，返回四个原始分项 + success（不做加权）。"""
    turns = [_raw_of(x) for x in trajectory]
    parsed = [parse_turn(t) for t in turns]
    n = len(turns)

    # R_format：全程合法（标签配对 + JSON + arguments 满足 schema），且恰好一个
    # answer 位于最后一回合（§4.1）。空 arguments 骗格式分 → 不合规（§4.4）。
    format_ok = True
    for p in parsed:
        if p.kind == "invalid":
            format_ok = False
        elif p.kind == "tool_call":
            schema = _schema_of(p.tool_name)
            if schema is None or validate_params(p.arguments, schema) is not None:
                format_ok = False
    answer_idx = [i for i, p in enumerate(parsed) if p.kind == "answer"]
    if len(answer_idx) != 1 or answer_idx[0] != n - 1:
        format_ok = False
    r_format = 1.0 if format_ok else 0.0

    # R_correct = 0.5·R_tool + 0.5·R_param（§2.2 分级给分）：
    #   R_tool  按位置比对 API 名，多余调用经分母 max(G,M) 扣分；
    #   R_param 只对位置匹配的步逐 key 比对参数值。
    gold_calls = task.gold.calls
    model_calls = [p for p in parsed if p.kind == "tool_call"]
    G, M = len(gold_calls), len(model_calls)
    matched_steps = 0
    matched_params = 0
    param_total = 0
    for i in range(min(G, M)):
        if model_calls[i].tool_name != gold_calls[i].api:
            continue
        matched_steps += 1
        model_params = model_calls[i].arguments or {}
        for k, gv in gold_calls[i].params.items():
            param_total += 1
            if k in model_params and _value_equal(model_params[k], gv):
                matched_params += 1
    r_tool = matched_steps / max(G, M) if max(G, M) > 0 else 0.0
    r_param = (
        matched_params / param_total
        if param_total > 0
        else (1.0 if G > 0 and matched_steps == G else 0.0)
    )
    r_correct = 0.5 * r_tool + 0.5 * r_param

    # R_answer：gold 事实集 ⊆ 模型答案事实集（数字/编号/时间归一化后比对）。
    if len(answer_idx) == 1:
        gold_facts = _extract_facts(task.gold.answer)
        model_facts = _extract_facts(parsed[answer_idx[0]].answer)
        r_answer = 1.0 if gold_facts <= model_facts else 0.0
    else:
        r_answer = 0.0

    # R_steps：步数惩罚（§2.4），步数 = 助手回合数 = 工具调用 + 最终回答。
    r_steps = max(0, n - task.min_steps) / task.max_steps

    success = r_format == 1.0 and r_correct == 1.0 and r_answer == 1.0
    return JudgeResult(
        format=r_format,
        correct=r_correct,
        answer=r_answer,
        steps=r_steps,
        success=success,
        matched_params=matched_params,
        param_total=param_total,
    )


__all__ = ["ParsedTurn", "parse_turn", "judge"]