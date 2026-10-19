"""Rule unit tests; these are not evidence of general natural-language ability."""

import pytest

from app.agent.memory import SessionMemory
from app.agent.state import SessionState


def test_switch_and_correction_take_precedence_over_mentions() -> None:
    state = SessionState()
    for text, expected in [
        ("S001 的价格", "S001"), ("改看 S002", "S002"),
        ("不是 S002，是 S001", "S001"),
    ]:
        state.update(text)
        assert state.current_sku == expected
    assert state.sources["current_sku"]["basis"] == "user_explicit"


def test_budget_update_and_retraction() -> None:
    state = SessionState()
    for text, expected in [
        ("预算 100 元以内", 100), ("改成 200 元以内", 200), ("预算不限", None),
    ]:
        state.update(text)
        assert state.budget_max == expected
    assert state.sources["budget_max"]["basis"] == "user_retraction"
    state.update("预算100元以内，改成200元以内")
    assert state.budget_max == 200


@pytest.mark.parametrize("text", ["预算不限", "取消预算限制", "不限预算"])
def test_budget_retraction_does_not_revive_old_limit(text: str) -> None:
    state = SessionState()
    state.update("预算 100 元以内")
    state.update(text)
    state.update("S001 的规格")
    assert state.budget_max is None


def test_comparison_order_and_explicit_reversal() -> None:
    state = SessionState()
    for first, second in [("S001", "S002"), ("S002", "S001"), ("S010", "S011")]:
        state.update(f"比较 {first} 和 {second}")
        assert state.comparison_skus == (first, second)
        assert state.current_sku is None
        state.update("前者的规格")
        assert state.current_sku == first
        assert state.query_field == "spec"
        state.update("后者多少钱")
        assert state.current_sku == second
        assert state.comparison_skus == (first, second)
        assert state.sources["current_sku"]["basis"] == "comparison_reference"


def test_reverse_order_without_repeating_ids_requires_known_comparison() -> None:
    state = SessionState()
    state.update("比较顺序反过来")
    assert state.clarification
    state.update("比较S001和S002")
    state.update("顺序反过来")
    assert state.comparison_skus == ("S002", "S001")
    assert state.current_sku is None
    state.update("前者规格")
    assert state.current_sku == "S002"


def test_unresolved_new_product_name_never_retains_old_sku() -> None:
    state = SessionState()
    state.update("S001多少钱")
    state.update("改看维生素")
    assert state.current_sku is None
    assert state.clarification
    state.update("它多少钱")
    assert state.current_sku is None
    assert state.clarification


def test_named_price_query_cannot_inherit_an_unrelated_old_product() -> None:
    state = SessionState()
    state.update("S001多少钱")
    state.update("演示 岚溪外用护理膏多少钱")
    assert state.current_sku is None
    assert state.clarification


def test_first_shopping_goal_retains_budget_from_same_task() -> None:
    state = SessionState()
    state.update("预算100元以内")
    state.update("想买维生素")
    assert state.budget_max == 100
    assert state.goal == "维生素"
    state.update("它多少钱")
    assert state.current_sku is None
    assert state.clarification


@pytest.mark.parametrize("reference", ["它多少钱", "这个多少钱", "另一个多少钱"])
def test_ambiguous_reference_after_comparison_never_picks_old_selection(reference: str) -> None:
    state = SessionState()
    state.update("S001多少钱")
    state.update("比较 S001 和 S002")
    state.update(reference)
    assert state.current_sku is None
    assert state.clarification


def test_another_product_requires_two_candidates_and_a_selected_one() -> None:
    state = SessionState()
    state.update("比较 S001 和 S002")
    state.update("前者规格")
    state.update("另一个多少钱")
    assert state.current_sku == "S002"
    state.update("改看 S003")
    state.update("另一个多少钱")
    assert state.current_sku is None
    assert state.clarification


@pytest.mark.parametrize("new_topic", [
    "换个话题，想买维生素", "我想买维生素", "Python 是什么？",
    "换个需求，S002多少钱？",
])
def test_new_topic_clears_irrelevant_context(new_topic: str) -> None:
    state = SessionState()
    for text in ("S001多少钱", "预算100元以内", "目标是日常养护", "不要推荐护理膏"):
        state.update(text)
    state.update(new_topic)
    assert state.current_sku == ("S002" if "S002" in new_topic else None)
    assert state.comparison_skus == ()
    assert state.budget_max is None
    assert state.goal != "日常养护"
    assert state.excluded_directions == ()
    if "S002" not in new_topic:
        state.update("它多少钱")
        assert state.current_sku is None
        assert state.clarification


def test_explicit_preferences_and_field_evidence() -> None:
    state = SessionState()
    state.update("目标是日常养护，不要推荐护理膏")
    assert state.goal == "日常养护"
    assert state.excluded_directions == ("护理膏",)
    state.update("目标是日常营养补充")
    assert state.goal == "日常营养补充"
    state.update("取消推荐排除限制")
    assert state.excluded_directions == ()
    state.update("S001 的价格和规格")
    assert state.query_field is None  # Do not invent a primary field.
    state.update("S001多少钱")
    state.update("S002呢？")
    assert state.query_field == "price"
    assert state.sources["query_field"]["basis"] == "previous_field_followup"
    state.update("谢谢")
    assert state.query_field is None


def test_multiple_unconfirmed_mentions_or_references_do_not_choose_one() -> None:
    state = SessionState()
    state.update("S001和S002")
    assert state.current_sku is None
    assert state.comparison_skus == ()
    assert state.clarification
    state.update("比较S001和S002")
    state.update("前者和后者多少钱")
    assert state.current_sku is None
    assert state.clarification


def test_sessions_snapshots_and_reset_are_isolated() -> None:
    first, second = SessionMemory(), SessionMemory()
    first.update_state("S001多少钱，预算100元以内")
    second.update_state("S002的规格，预算200元以内")
    snapshot = first.get_state()
    snapshot.current_sku = "S999"
    snapshot.sources.clear()
    assert first.get_state().current_sku == "S001"
    assert first.get_state().sources
    first.add_message("user", "S001多少钱")
    first.clear()
    assert first.get_history() == []
    assert first.get_last_turn() is None
    assert first.get_state() == SessionState()
    assert second.get_state().current_sku == "S002"
    assert second.get_state().budget_max == 200


def test_pending_name_is_not_a_bound_identity_and_resolution_is_query_scoped():
    memory = SessionMemory()
    memory.update_state("S001多少钱，预算200元以内")
    state = memory.update_state("演示 岚溪外用护理膏多少钱")
    assert state.current_sku is None
    assert state.name_query == "演示 岚溪外用护理膏"
    assert state.budget_max == 200
    with pytest.raises(ValueError, match="pending query"):
        memory.resolve_name("其他查询", "S001", None)
    state = memory.resolve_name(state.name_query, "S002", None)
    assert state.current_sku == "S002" and state.clarification is None
    assert state.sources["current_sku"]["basis"] == "unique_name_search"
    state = memory.update_state("它的规格")
    assert state.current_sku == "S002" and state.name_query is None
    memory.clear()
    assert memory.get_state() == SessionState()
