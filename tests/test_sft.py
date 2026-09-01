"""SFT 冷启动测试（design_zh.md §4.2）。

无 torch 依赖：loss 掩码用假 tokenizer 验证，切分/chat 转换/配置预设为纯函数测试，
验证指标用 GoldTeacher（全对）与 BadTeacher（全错）做确定性断言。
"""

from data import GoldTeacher
from data.teacher import Teacher
from envs.api_sandbox import ApiSandboxEnv
from envs.api_sandbox.task_generator import generate_tasks as api_tasks
from envs.task_schema import Message, Trajectory
from train.sft import (
    SFTConfig,
    compute_labels,
    evaluate,
    first_tool,
    to_chat_messages,
    train_val_split,
)

# ---------------------------------------------------------------------------
# 假 tokenizer：role 标记 + 每词一 token，支持前缀一致的 apply_chat_template
# ---------------------------------------------------------------------------


class FakeTokenizer:
    ROLE_IDS = {"system": 0, "user": 1, "assistant": 2, "tool": 3, "assistant-start": 4}

    def __init__(self):
        self._words: dict[str, int] = {}
        self._next = 10

    def _word_id(self, w: str) -> int:
        if w not in self._words:
            self._words[w] = self._next
            self._next += 1
        return self._words[w]

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        ids = []
        for m in messages:
            ids.append(self.ROLE_IDS[m["role"]])
            for w in m["content"].split():
                ids.append(self._word_id(w))
        if add_generation_prompt:
            ids.append(self.ROLE_IDS["assistant-start"])
        return ids


# ---------------------------------------------------------------------------
# compute_labels
# ---------------------------------------------------------------------------


def _assistant_positions(tok, msgs):
    pos = set()
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        start = len(tok.apply_chat_template(msgs[:i], tokenize=True, add_generation_prompt=True))
        end = len(tok.apply_chat_template(msgs[: i + 1], tokenize=True, add_generation_prompt=False))
        pos.update(range(start, end))
    return pos


def test_compute_labels_masks_non_assistant():
    tok = FakeTokenizer()
    msgs = [
        {"role": "system", "content": "sys prompt"},
        {"role": "user", "content": "do thing"},
        {"role": "assistant", "content": "call tool"},
        {"role": "tool", "content": "result x"},
        {"role": "assistant", "content": "final answer"},
    ]
    input_ids = tok.apply_chat_template(msgs, tokenize=True)
    labels = compute_labels(input_ids, msgs, tok)

    assert len(labels) == len(input_ids)
    assistant = _assistant_positions(tok, msgs)
    assert assistant  # 确实有 assistant 内容被标注
    for j in range(len(input_ids)):
        if j in assistant:
            assert labels[j] == input_ids[j]
        else:
            assert labels[j] == -100
    # 恰好 4 个 assistant 内容词被标注
    assert sum(1 for lab in labels if lab != -100) == 4


def test_compute_labels_no_assistant_all_masked():
    tok = FakeTokenizer()
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    input_ids = tok.apply_chat_template(msgs, tokenize=True)
    labels = compute_labels(input_ids, msgs, tok)
    assert all(lab == -100 for lab in labels)


# ---------------------------------------------------------------------------
# to_chat_messages / train_val_split
# ---------------------------------------------------------------------------


def test_to_chat_messages():
    traj = Trajectory(
        task_id="t",
        messages=[Message(role="system", content="s"), Message(role="user", content="u")],
        success=True,
        steps=1,
    )
    assert to_chat_messages(traj) == [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
    ]


def test_train_val_split_deterministic_and_ratio():
    trajs = [Trajectory(task_id=f"t{i}", messages=[], success=True, steps=1) for i in range(20)]
    a = train_val_split(trajs, val_ratio=0.2, seed=42)
    b = train_val_split(trajs, val_ratio=0.2, seed=42)
    assert [t.task_id for t in a[0]] == [t.task_id for t in b[0]]
    assert [t.task_id for t in a[1]] == [t.task_id for t in b[1]]
    assert len(a[1]) == 4  # 20 * 0.2
    assert len(a[0]) == 16
    # 无交集
    assert set(t.task_id for t in a[0]) & set(t.task_id for t in a[1]) == set()


# ---------------------------------------------------------------------------
# SFTConfig 预设
# ---------------------------------------------------------------------------


def test_config_presets():
    full = SFTConfig.full_06b()
    assert full.model_name == "Qwen/Qwen3-0.6B"
    assert full.lr == 1e-5
    assert full.use_lora is False

    lora = SFTConfig.lora_17b()
    assert lora.model_name == "Qwen/Qwen3-1.7B"
    assert lora.lr == 2e-4
    assert lora.use_lora is True
    assert (lora.lora_rank, lora.lora_alpha) == (16, 32)


# ---------------------------------------------------------------------------
# evaluate（复用 data.synthesize_one）
# ---------------------------------------------------------------------------


class BadTeacher(Teacher):
    def generate(self, messages):
        return "没有标签的乱输出"


def test_evaluate_gold_all_pass():
    env = ApiSandboxEnv()
    n = 10
    tasks = api_tasks(0, n)
    expected_steps = sum(t.min_steps for t in tasks) / n

    m = evaluate(GoldTeacher, [env], n_per_env=n, seed=0)

    assert m.n == n
    assert m.format_rate == 1.0
    assert m.tool_acc == 1.0
    assert m.success_rate == 1.0
    assert m.avg_steps == expected_steps


def test_evaluate_bad_all_fail():
    env = ApiSandboxEnv()
    m = evaluate(lambda task: BadTeacher(), [env], n_per_env=5, seed=0)
    assert m.n == 5
    assert m.format_rate == 0.0
    assert m.tool_acc == 0.0
    assert m.success_rate == 0.0


def test_first_tool_none_on_no_tool():
    traj = Trajectory(
        task_id="t",
        messages=[Message(role="assistant", content="没有 tool_call")],
        success=False,
        steps=1,
    )
    assert first_tool(traj) is None
