"""GRPO 奖励函数测试（design_zh.md §2.4 / §4.3）。

核心是奖励函数的纯逻辑：加权组装（复用 judge + DEFAULT_REWARD_WEIGHTS）、组内优势
（std=0 防除零）、以及 rollout 奖励（GoldTeacher 全对 / BadTeacher 全错）。
"""

import json

import pytest

from data import GoldTeacher
from data.teacher import Teacher
from envs.api_sandbox import ApiSandboxEnv
from envs.api_sandbox.task_generator import generate_tasks as api_tasks
from envs.base_env import JudgeResult
from envs.text2sql import generate_tasks as sql_tasks
from train.grpo import (
    GRPOConfig,
    RewardManager,
    build_prompt,
    compute_trajectory_reward,
    extract_task_id,
    group_advantages,
    judge_for_task,
    make_reward_func,
    total_reward,
)


class BadTeacher(Teacher):
    def generate(self, messages):
        return "没有标签的乱输出"


# ---------------------------------------------------------------------------
# total_reward（§2.4：R = w1·F + w2·C + w3·A − w4·S）
# ---------------------------------------------------------------------------


def test_total_reward_weighted():
    jr = JudgeResult(format=1.0, correct=0.5, answer=0.0, steps=0.25)
    # 0.40*1 + 0.30*0.5 + 0.20*0 − 0.10*0.25 = 0.40 + 0.15 − 0.025 = 0.525
    assert total_reward(jr) == pytest.approx(0.525)


def test_total_reward_gold_is_09():
    jr = JudgeResult(format=1.0, correct=1.0, answer=1.0, steps=0.0)
    assert total_reward(jr) == pytest.approx(0.90)


def test_total_reward_steps_penalty_negative():
    # 格式正确但多试错、未答对：steps 惩罚把奖励拉低
    jr = JudgeResult(format=1.0, correct=0.0, answer=0.0, steps=0.5)
    assert total_reward(jr) == pytest.approx(0.40 - 0.10 * 0.5)


# ---------------------------------------------------------------------------
# group_advantages（DeepSeekMath 组内标准化，std=0 防除零）
# ---------------------------------------------------------------------------


def test_group_advantages_normalized():
    adv = group_advantages([1.0, 2.0, 3.0, 4.0])
    assert sum(adv) == pytest.approx(0.0)
    pop_std = (sum(a * a for a in adv) / len(adv)) ** 0.5
    assert pop_std == pytest.approx(1.0)


def test_group_advantages_std_zero():
    assert group_advantages([5.0, 5.0, 5.0]) == [0.0, 0.0, 0.0]
    assert group_advantages([]) == []


# ---------------------------------------------------------------------------
# compute_trajectory_reward（rollout 复用 data.synthesize_one）
# ---------------------------------------------------------------------------


def test_compute_trajectory_reward_gold():
    env = ApiSandboxEnv()
    task = api_tasks(0, 1)[0]
    r = compute_trajectory_reward(env, task, GoldTeacher(task))
    assert r == pytest.approx(0.90)


def test_compute_trajectory_reward_bad():
    env = ApiSandboxEnv()
    task = api_tasks(0, 1)[0]
    r = compute_trajectory_reward(env, task, BadTeacher())
    assert r == pytest.approx(0.0)


def test_reward_manager_advantages_gold_identical():
    env = ApiSandboxEnv()
    task = api_tasks(0, 1)[0]
    rm = RewardManager(env)
    adv = rm.advantages(task, GoldTeacher, group_size=8)
    # 黄金轨迹 8 条全同 → std=0 → 优势全 0
    assert len(adv) == 8
    assert all(x == 0.0 for x in adv), f"expected all-zero advantages, got {adv}"
    # 防 list 渲染混淆：组内标准化定义上 sum(A_i) 必 = 0；任何 sum 不为 0 都是 bug
    assert sum(adv) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# GRPOConfig / prompt
# ---------------------------------------------------------------------------


def test_grpo_config_defaults():
    c = GRPOConfig()
    assert c.group_size == 8
    assert c.lr == 1e-6
    assert c.epsilon == 0.2
    assert c.beta == 0.04
    assert (c.max_prompt_len, c.max_response_len) == (2048, 1024)
    assert c.num_prompts == 128


def test_build_prompt_roundtrip():
    task = api_tasks(0, 1)[0]
    prompt = build_prompt(task)
    assert extract_task_id(prompt) == task.task_id
    assert task.instruction in prompt
    assert "<tool_call>" in prompt


# ---------------------------------------------------------------------------
# make_reward_func / judge_for_task（TRL 单轮简化路径）
# ---------------------------------------------------------------------------


def test_make_reward_func_single_turn():
    # 固定取单 gold 调用任务（weather.forecast），方便对第一回合 tool_call 精确断言
    task = api_tasks(2, 1)[0]
    assert len(task.gold.calls) == 1
    rf = make_reward_func({task.task_id: task})
    prompt = build_prompt(task)

    # 单轮 answer：第一回合就该调工具（防 reward hacking），不给工具调用一律 0 分
    rewards = rf([prompt], ["<answer>ok</answer>"])
    assert rewards[0] == pytest.approx(0.0)

    # 单轮正确 tool_call：format=1 + first-turn correct=1 → 0.40 + 0.30 = 0.70
    c = task.gold.calls[0]
    tool_raw = (
        "<tool_call>"
        + json.dumps({"name": c.api, "arguments": c.params}, ensure_ascii=False)
        + "</tool_call>"
    )
    rewards = rf([prompt], [tool_raw])
    assert rewards[0] == pytest.approx(0.70)


def test_judge_for_task_dispatches_text2sql():
    task = sql_tasks(0, 1)[0]
    traj = [{"raw": '<tool_call>{"name":"sql.execute","arguments":{"query":"SELECT 1"}}</tool_call>'}]
    jr = judge_for_task(task, traj)
    assert isinstance(jr, JudgeResult)
    assert jr.format == 0.0  # sql judge 要求 answer 结尾，单轮 SQL 无 answer → format=0