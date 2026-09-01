"""评测测试（design_zh.md §5）。

覆盖：JudgeResult 参数计数字段（向后兼容默认 0）、两个 judge 的填充、§5.2 指标聚合
（param_acc / reward_mean）、benchmark 构建（占比/确定性/干扰项标签）、四组评测执行
（复用 train.sft.eval + train.grpo.RewardManager 加权核）与 Markdown 报告渲染。
无 torch 依赖，用 GoldTeacher（全对）与 BadTeacher（全错）做确定性断言。
"""

from collections import Counter

import pytest

from data import GoldTeacher
from data.synthesis import synthesize_one
from data.teacher import Teacher
from envs.api_sandbox import ApiSandboxEnv
from envs.api_sandbox.task_generator import generate_tasks as api_tasks
from envs.base_env import JudgeResult
from envs.text2sql import Text2SQLEnv
from envs.text2sql import generate_tasks as sql_tasks
from eval import GroupResult, build_benchmark, render_markdown, run_all, run_group
from train.sft.eval import (
    estimate_tokens,
    evaluate_tasks,
    evaluate_tasks_detailed,
)


class BadTeacher(Teacher):
    def generate(self, messages):
        return "没有 tool_call 也没有 answer 的乱输出"


# ---------------------------------------------------------------------------
# JudgeResult 参数计数字段（§5.2 参数正确率）
# ---------------------------------------------------------------------------


def test_judge_result_param_fields_default_zero():
    jr = JudgeResult(format=1.0, correct=1.0, answer=1.0, steps=0.0)
    assert jr.matched_params == 0
    assert jr.param_total == 0


def test_api_judge_populates_params_gold_full():
    task = api_tasks(0, 1)[0]
    env = ApiSandboxEnv()
    r = synthesize_one(task, env, GoldTeacher(task))
    assert r.judge.param_total > 0
    assert r.judge.matched_params == r.judge.param_total


def test_text2sql_judge_param_counter():
    task = sql_tasks(0, 1)[0]
    env = Text2SQLEnv()
    r = synthesize_one(task, env, GoldTeacher(task))
    assert r.judge.param_total == 1
    assert r.judge.matched_params == 1
    assert r.judge.success


# ---------------------------------------------------------------------------
# §5.2 指标聚合（evaluate_tasks / evaluate_tasks_detailed）
# ---------------------------------------------------------------------------


def test_evaluate_tasks_gold_all_pass():
    tasks = build_benchmark(seed=42, size=20)
    env = ApiSandboxEnv()
    m = evaluate_tasks(GoldTeacher, tasks, env)
    assert m.n == 20
    assert m.format_rate == 1.0
    assert m.tool_acc == 1.0
    assert m.param_acc == 1.0
    assert m.success_rate == 1.0
    assert m.reward_mean == pytest.approx(0.90)
    assert m.avg_tokens > 0


def test_evaluate_tasks_bad_all_fail():
    tasks = build_benchmark(seed=42, size=10)
    env = ApiSandboxEnv()
    m = evaluate_tasks(lambda t: BadTeacher(), tasks, env)
    assert m.n == 10
    assert m.format_rate == 0.0
    assert m.tool_acc == 0.0
    assert m.param_acc == 0.0
    assert m.success_rate == 0.0
    assert m.reward_mean == pytest.approx(0.0)


def test_evaluate_tasks_detailed_returns_results():
    tasks = build_benchmark(seed=42, size=5)
    env = ApiSandboxEnv()
    m, results = evaluate_tasks_detailed(GoldTeacher, tasks, env)
    assert m.n == 5
    assert len(results) == 5
    assert all(r.success for r in results)
    assert all(r.dim for r in results)
    assert results[0].task_id == tasks[0].task_id


def test_estimate_tokens_basic():
    assert estimate_tokens("") == 0
    assert estimate_tokens("hello") == pytest.approx(2)  # 5 ASCII → ceil(5/4)=2
    assert estimate_tokens("天气") == 2              # 2 CJK → 2


# ---------------------------------------------------------------------------
# benchmark 构建（§5.1）
# ---------------------------------------------------------------------------


def test_benchmark_size_and_proportions():
    tasks = build_benchmark(seed=42, size=20)
    assert len(tasks) == 20
    c = Counter(t.meta["bench_dim"] for t in tasks)
    assert c["single_tool"] == 5
    assert c["multi_tool"] == 6
    assert c["multi_turn"] == 4
    assert c["long_horizon"] == 2
    assert c["distractor"] == 3


def test_benchmark_deterministic():
    a = build_benchmark(seed=7, size=50)
    b = build_benchmark(seed=7, size=50)
    assert [t.task_id for t in a] == [t.task_id for t in b]
    assert [t.instruction for t in a] == [t.instruction for t in b]


def test_benchmark_distractor_injects_extra_tools():
    tasks = build_benchmark(seed=0, size=100)
    dists = [t for t in tasks if t.meta["bench_dim"] == "distractor"]
    assert dists
    for t in dists:
        assert t.meta["distractor"] is True
        assert len(t.tools) > len(t.gold.calls)


# ---------------------------------------------------------------------------
# 四组评测执行（run_group / run_all）+ 报告
# ---------------------------------------------------------------------------


def test_run_all_groups_and_failures():
    tasks = build_benchmark(seed=42, size=20)
    env = ApiSandboxEnv()
    results = run_all({"教师": GoldTeacher, "基座": lambda t: BadTeacher()}, tasks, env)
    assert list(results) == ["教师", "基座"]
    assert isinstance(results["教师"], GroupResult)
    assert results["教师"].metrics.success_rate == 1.0
    assert results["基座"].metrics.success_rate == 0.0
    assert results["教师"].failures == []
    assert len(results["基座"].failures) == 20


def test_run_group_uses_reward_weights():
    tasks = build_benchmark(seed=42, size=5)
    env = ApiSandboxEnv()
    gr = run_group(GoldTeacher, tasks, env)
    # 与 GRPO 同权：黄金轨迹 reward_mean = 0.90
    assert gr.metrics.reward_mean == pytest.approx(0.90)


def test_render_markdown_sections():
    tasks = build_benchmark(seed=42, size=20)
    results = run_all({"教师": GoldTeacher, "基座": lambda t: BadTeacher()}, tasks, ApiSandboxEnv())
    md = render_markdown(results)
    assert "一、四组模型指标对比" in md
    assert "二、分维度端到端成功率" in md
    assert "三、失败 case 清单" in md
    assert "格式合规率" in md
    assert "参数正确率" in md
    assert "教师" in md and "基座" in md