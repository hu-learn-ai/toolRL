"""base_env 的接口契约测试：抽象性、JudgeResult 约束、默认权重。"""

import pytest
from pydantic import ValidationError

from envs import DEFAULT_REWARD_WEIGHTS, BaseToolEnv, JudgeResult


class _DummyEnv(BaseToolEnv):
    def reset(self, task):
        return {"observation": [], "instruction": task.instruction}

    def step(self, action):
        return {"observation": {}, "done": True}

    def judge(self, trajectory):
        return JudgeResult(format=1.0, correct=1.0, answer=1.0, steps=0.0, success=True)

    def task_generator(self, seed):
        return iter(())


def test_base_env_is_abstract():
    with pytest.raises(TypeError):
        BaseToolEnv()  # type: ignore[abstract]


def test_concrete_subclass_is_instantiable():
    env = _DummyEnv()
    assert isinstance(env, BaseToolEnv)


def test_judge_result_requires_all_components():
    with pytest.raises(ValidationError):
        JudgeResult()


def test_judge_result_success_defaults_false():
    r = JudgeResult(format=1.0, correct=0.5, answer=0.0, steps=0.2)
    assert r.success is False


def test_judge_result_rejects_out_of_range_format():
    with pytest.raises(ValidationError):
        JudgeResult(format=1.5, correct=0.0, answer=0.0, steps=0.0)


def test_judge_result_rejects_negative_steps():
    with pytest.raises(ValidationError):
        JudgeResult(format=1.0, correct=0.0, answer=0.0, steps=-0.1)


def test_reward_weights_match_design():
    assert DEFAULT_REWARD_WEIGHTS == {
        "format": 0.40,
        "correct": 0.30,
        "answer": 0.20,
        "steps": 0.10,
    }
