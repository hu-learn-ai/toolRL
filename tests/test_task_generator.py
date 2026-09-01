"""task_generator 的测试：确定性、结构一致性、多轮依赖、干扰注入。"""

import itertools

from envs.api_sandbox import call, generate_tasks, task_generator


def test_generator_is_deterministic():
    assert generate_tasks(0, 50) == generate_tasks(0, 50)
    assert generate_tasks(1, 50) != generate_tasks(2, 50)


def test_task_generator_is_infinite():
    first = list(itertools.islice(task_generator(0), 5))
    assert len(first) == 5


def test_gold_calls_are_valid():
    for task in generate_tasks(0, 300):
        for c in task.gold.calls:
            result = call(c.api, c.params)
            assert "error" not in result, f"{task.task_id}: {c.api} 参数非法"


def test_min_max_steps_consistent():
    for task in generate_tasks(0, 300):
        assert task.min_steps == len(task.gold.calls) + 1  # 步数含最终回答那一轮
        assert task.max_steps >= task.min_steps


def test_tools_contain_all_gold_apis():
    for task in generate_tasks(0, 300):
        tool_names = {t.name for t in task.tools}
        for c in task.gold.calls:
            assert c.api in tool_names


def test_no_distractors_when_ratio_zero():
    for task in generate_tasks(0, 50, distractor_ratio=0.0):
        assert {t.name for t in task.tools} == {c.api for c in task.gold.calls}


def test_distractors_injected_when_ratio_one():
    for task in generate_tasks(0, 50, distractor_ratio=1.0):
        gold_apis = {c.api for c in task.gold.calls}
        tool_names = {t.name for t in task.tools}
        assert gold_apis <= tool_names  # 必需工具仍在
        assert tool_names - gold_apis  # 确有干扰工具


def test_categories_cover_all_structural_types():
    cats = {t.category for t in generate_tasks(0, 400)}
    assert cats == {"single_tool", "multi_tool", "multi_turn", "long_horizon"}


def test_multiturn_booking_references_search_result():
    checked = 0
    for task in generate_tasks(0, 500):
        calls = task.gold.calls
        if len(calls) < 2:
            continue
        search, book = calls[0], calls[1]
        if not (search.api.endswith(".search") and book.api.endswith(".book")):
            continue
        result = call(search.api, search.params)
        if "flights" in result:
            assert any(f["flight_no"] == book.params["flight_no"] for f in result["flights"])
        elif "trains" in result:
            assert any(t["train_no"] == book.params["train_no"] for t in result["trains"])
        elif "hotels" in result:
            assert any(h["name"] == book.params["hotel"] for h in result["hotels"])
        checked += 1
    assert checked > 0


def test_gold_answer_nonempty():
    for task in generate_tasks(0, 100):
        assert task.gold.answer.strip()
