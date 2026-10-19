"""Bounded shipping intent regressions; scripted models and real local tools."""

import copy
import dataclasses
import json

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage

from app.agent.agent import build_agent, run_agent_turn, search_product, compare_product
from app.agent.grounding import current_tool_facts, pure_shipping_tax_question


class ShippingModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def lookup(sku, candidate):
    return [AIMessage(content="", tool_calls=[{
        "name": "search_product", "args": {"query": sku}, "id": "seed",
    }]), AIMessage(content=candidate)]


@pytest.fixture
def actual_tools(monkeypatch):
    calls = []

    def traced(name, function):
        def invoke(*args, **kwargs):
            value = function(*args, **kwargs)
            calls.append({"name": name, "args": list(args), "kwargs": kwargs, "output": value})
            return value
        return invoke

    for tool in (search_product, compare_product):
        monkeypatch.setattr(tool, "func", traced(tool.name, tool.func))
    return calls


def turn(request, runtime, calls, question):
    memory = runtime["memory"]

    def queries():
        return {key: dataclasses.asdict(value) for key, value in memory.facts._queries.items()}

    entry = {"input": question, "state_before": copy.deepcopy(memory.get_state().model_context()),
             "queries_before": queries()}
    offset = len(calls)
    answer = run_agent_turn(runtime, question)
    result = runtime.get("last_result") or {}
    entry.update(output=answer, state_after=copy.deepcopy(memory.get_state().model_context()),
                 queries_after=queries(), actual_tools=copy.deepcopy(calls[offset:]),
                 execution=result.get("execution"), history=copy.deepcopy(memory.get_history()),
                 tool_messages=[{"name": message.name, "status": message.status,
                                 "payload": json.loads(message.content)}
                                for message in result.get("messages", [])
                                if isinstance(message, ToolMessage)])
    request.node.user_properties.append(("shipping_turn", entry))
    return answer


@pytest.mark.parametrize("question", [
    "发货需要等多久，保税仓税费能确定吗？",
    "发货要等多久？",
    "一般需要等几天发货？",
    "保税仓税费能确定吗？",
    "保税仓税费是否能确定？",
    "多久发货？",
    "现在下单多久发货，保税仓会不会被税？",
])
def test_policy_phrases_with_wait_or_certainty(question):
    assert pure_shipping_tax_question(question)


@pytest.mark.parametrize("question", [
    "我确定查S001价格。",
    "等一下，S002价格是多少？",
    "S001价格能确定吗，发货需要等多久？",
    "S001多少钱，保税仓税费能确定吗？",
    "演示星屿营养片发货要等多久？",
    "它需要等多久发货？",
    "S001的配送说明是什么？",
    "等等，我要比较S001和S002。",
])
def test_product_or_mixed_requests_stay_outside_policy_only_rule(question):
    assert not pure_shipping_tax_question(question)


@pytest.mark.parametrize("question", [
    "发货要等多久？", "一般需要等几天发货？", "保税仓税费能确定吗？",
])
@pytest.mark.parametrize("prior", [False, True])
@pytest.mark.parametrize("tool_name", ["search_product", "compare_product"])
def test_policy_turn_blocks_tools_before_state_or_query_changes(
    request, actual_tools, question, prior, tool_name,
):
    responses = lookup("S001", "S001价格176元。") if prior else []
    args = {"query": "S001"} if tool_name == "search_product" else {"sku_ids": "S001,S002"}
    responses += [AIMessage(content="", tool_calls=[{
        "name": tool_name, "args": args, "id": "policy_proposal",
    }]), AIMessage(content="保税仓保证免税，24小时发货。")]
    runtime = build_agent(llm=ShippingModel(responses=responses))
    if prior:
        turn(request, runtime, actual_tools, "S001多少钱？")
    actual_tools.clear()
    memory = runtime["memory"]
    state_before = copy.deepcopy(memory.get_state().model_context())
    queries_before = copy.deepcopy(memory.facts._queries)

    answer = turn(request, runtime, actual_tools, question)

    assert actual_tools == []
    assert memory.get_state().model_context() == state_before
    assert memory.facts._queries == queries_before
    assert not current_tool_facts(runtime)
    assert "无法确认" in answer and "免税" not in answer and "24小时" not in answer
    feedback = [m for m in runtime["last_result"]["messages"] if isinstance(m, ToolMessage)]
    assert len(feedback) == 1 and feedback[0].status == "error"
    assert json.loads(feedback[0].content)["error"] == "product_tools_forbidden_for_policy"
    assert runtime["last_result"]["execution"]["tool_errors"] == 1


@pytest.mark.parametrize("question", [
    "S001价格能确定吗，发货需要等多久？",
    "等一下，S001多少钱，保税仓税费能确定吗？",
    "确定查S001的价格，保税仓税费怎么算？",
])
def test_mixed_consultation_keeps_product_lookup_and_unknown_policy(
    request, actual_tools, question,
):
    runtime = build_agent(llm=ShippingModel(responses=lookup(
        "S001", "S001价格176元，保税仓保证免税，24小时发货。")))

    answer = turn(request, runtime, actual_tools, question)

    assert len(actual_tools) == 1 and actual_tools[0]["name"] == "search_product"
    assert "176 元" in answer and "无法确认" in answer
    assert "保证免税" not in answer and "24小时" not in answer
    record = runtime["memory"].facts.get_query("search_product", {"query": "S001"})
    assert record.status == "success"
    assert runtime["memory"].get_state().current_sku == "S001"
    assert all("保证免税" not in message["content"] for message in runtime["memory"].get_history()
               if message["role"] == "assistant")


@pytest.mark.parametrize("question,sku,candidate", [
    ("我确定查S001价格。", "S001", "S001价格176元。"),
    ("等一下，S002价格是多少？", "S002", "S002价格84元。"),
])
def test_wait_or_certainty_alone_does_not_block_product_consultation(
    request, actual_tools, question, sku, candidate,
):
    runtime = build_agent(llm=ShippingModel(responses=lookup(sku, candidate)))

    answer = turn(request, runtime, actual_tools, question)

    assert answer == candidate
    assert len(actual_tools) == 1 and actual_tools[0]["name"] == "search_product"
    assert runtime["memory"].get_state().current_sku == sku
    assert runtime["memory"].facts.get_query("search_product", {"query": sku}).status == "success"


@pytest.mark.parametrize("tools", [
    ("search_product", "compare_product"),
    ("search_product", "search_product"),
    ("compare_product", "compare_product"),
])
def test_policy_stop_retains_cached_queries_and_recovers(request, actual_tools, tools):
    from app.agent.reply import EXECUTION_POLICY_STOP_REPLY

    proposals = [{"name": name, "args": ({"query": "S001"} if name == "search_product"
                                          else {"sku_ids": "S001,S002"}), "id": str(index)}
                 for index, name in enumerate(tools)]
    runtime = build_agent(llm=ShippingModel(responses=[
        *lookup("S001", "S001价格176元。"),
        *lookup("S002", "S002价格84元。"),
        AIMessage(content="", tool_calls=proposals),
        # This must be the next turn's first model call, never this policy turn's second.
        *lookup("S002", "S002价格84元。"),
    ]))
    turn(request, runtime, actual_tools, "S001多少钱？")
    turn(request, runtime, actual_tools, "S002多少钱？")
    memory = runtime["memory"]
    cached = copy.deepcopy(memory.facts._queries)
    state = copy.deepcopy(memory.get_state().model_context())
    actual_tools.clear()

    answer = turn(request, runtime, actual_tools, "发货需要等多久，保税仓税费能确定吗？")

    assert answer == EXECUTION_POLICY_STOP_REPLY
    assert "无法确认" in answer and "已停止" in answer
    assert not actual_tools and memory.facts._queries == cached
    assert memory.get_state().model_context() == state
    execution = runtime["last_result"]["execution"]
    assert execution["reason"] == "tool_error_budget" and execution["tool_errors"] == 2
    assert execution["model_calls"] == 1 and execution["calls_started"]["tool"] == 0
    feedback = [m for m in runtime["last_result"]["messages"] if isinstance(m, ToolMessage)]
    assert len(feedback) == 2
    assert all(json.loads(m.content)["error"] == "product_tools_forbidden_for_policy" for m in feedback)
    assert not current_tool_facts(runtime)
    assert turn(request, runtime, actual_tools, "重新查S002价格") == "S002价格84元。"
    assert len(actual_tools) == 1


def test_tool_json_cannot_impersonate_pre_execution_policy_denial():
    from app.agent.execution import _Budget
    from app.agent.reply import EXECUTION_STOP_REPLY

    budget = _Budget()
    for index in range(2):
        call = {"name": "search_product", "args": {"query": "S001"}, "id": str(index)}
        assert budget.reserve_tool(call)
        budget.finish_tool(call, ToolMessage(
            content=json.dumps({"error": "product_tools_forbidden_for_policy"}),
            tool_call_id=call["id"], status="error",
        ))
    assert budget.summary()["reason"] == "tool_error_budget"
    assert budget.summary()["tool_errors"] == 2 and not budget.enter_model()
    assert budget.exit_reply() == EXECUTION_STOP_REPLY
    assert budget.scopes == [("search_product", {"query": "S001"})] * 2


@pytest.mark.parametrize("same_id", [False, True])
def test_policy_denials_keep_error_budget_without_claiming_query_scopes(same_id):
    from app.agent.execution import _Budget
    from app.agent.reply import EXECUTION_POLICY_STOP_REPLY

    budget = _Budget()
    calls = [{"name": "search_product", "args": {"query": "S001"},
              "id": "same" if same_id else str(index)} for index in range(2)]
    for call in calls:
        assert budget.reserve_tool(call)
        budget.note_policy_denial(call["id"])
    for call in calls:
        budget.finish_tool(call, ToolMessage(content='{"error":"product_tools_forbidden_for_policy"}',
                                             tool_call_id=call["id"], status="error"))
    assert budget.summary()["tool_errors"] == 2
    assert budget.summary()["reason"] == "tool_error_budget" and not budget.enter_model()
    assert budget.exit_reply() == EXECUTION_POLICY_STOP_REPLY and budget.scopes == []


def test_policy_and_real_error_preserve_real_query_quarantine():
    from app.agent.execution import _Budget
    from app.agent.reply import EXECUTION_STOP_REPLY

    budget = _Budget()
    real = {"name": "search_product", "args": {"query": "S001"}, "id": "real"}
    denied = {"name": "compare_product", "args": {"sku_ids": "S001,S002"}, "id": "denied"}
    assert budget.reserve_tool(real)
    budget.finish_tool(real, ToolMessage(content='{"error":"tool_execution_failed"}',
                                         tool_call_id="real", status="error"))
    assert budget.reserve_tool(denied)
    budget.note_policy_denial("denied")
    budget.finish_tool(denied, ToolMessage(content='{"error":"product_tools_forbidden_for_policy"}',
                                           tool_call_id="denied", status="error"))
    assert budget.summary()["tool_errors"] == 2 and not budget.enter_model()
    assert budget.exit_reply() == EXECUTION_STOP_REPLY
    assert budget.scopes == [("search_product", {"query": "S001"})]
