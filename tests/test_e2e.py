"""End-to-end tests for the OTC Agent prototype.

These tests validate the current enterprise flow with mocked LLM responses:
input guard -> Agent -> Tool -> output guard. They do not call a real model API
or modify Tool, Safety, or data modules.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from pydantic import Field

from app.agent.agent import build_agent, run_agent_turn
from app.safety.guard import check_ai_output, check_user_input


class E2EFakeModel(FakeMessagesListChatModel):
    """Mock LLM that supports tool binding and records high-level flow trace."""

    trace: list[str] = Field(default_factory=list)

    def bind_tools(
        self,
        tools: list[Any],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> "E2EFakeModel":
        """Return self so LangChain Agent can bind tools during E2E tests."""
        _ = tools, tool_choice, kwargs
        return self

    def _generate(self, messages: list[BaseMessage], stop: Any = None, **kwargs: Any) -> Any:
        """Record Agent and Tool execution points visible to the mock model."""
        tool_messages = [message for message in messages if isinstance(message, ToolMessage)]
        if tool_messages and tool_messages[-1].name not in self.trace:
            self.trace.append(str(tool_messages[-1].name))
        elif "agent" not in self.trace:
            self.trace.append("agent")

        return super()._generate(messages, stop=stop, **kwargs)


def _input_guard_with_trace(model: E2EFakeModel, text: str) -> dict[str, Any]:
    """Run input guard while recording E2E trace."""
    model.trace.append("input_guard")
    return check_user_input(text)


def _output_guard_with_trace(model: E2EFakeModel, text: str) -> dict[str, Any]:
    """Run output guard while recording E2E trace."""
    model.trace.append("output_guard")
    return check_ai_output(text)


def _tool_messages(runtime: dict[str, Any]) -> list[ToolMessage]:
    """Return ToolMessages from the latest Agent result."""
    return [
        message
        for message in runtime.get("last_result", {}).get("messages", [])
        if isinstance(message, ToolMessage)
    ]


def test_e2e_normal_flow_searches_product_and_returns_price() -> None:
    """Normal flow should pass guards, call search_product, and return answer."""
    model = E2EFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_product",
                        "args": {"query": "S001"},
                        "id": "e2e_search_call",
                    }
                ],
            ),
            AIMessage(content="演示星屿营养片的价格是 176 元。"),
        ]
    )
    runtime = build_agent(llm=model)

    answer = run_agent_turn(
        runtime,
        "S001多少钱？",
        input_guard=lambda text: _input_guard_with_trace(model, text),
        output_guard=lambda text: _output_guard_with_trace(model, text),
    )
    tool_messages = _tool_messages(runtime)
    tool_output = json.loads(tool_messages[0].content)

    assert model.trace == ["input_guard", "agent", "search_product", "output_guard"]
    assert tool_messages[0].name == "search_product"
    assert tool_output["results"][0]["name"] == "演示星屿营养片"
    assert tool_output["results"][0]["price"] == "176 元"
    assert "演示星屿营养片" in answer
    assert "176 元" in answer


def test_e2e_risky_input_stops_before_agent_and_tool() -> None:
    """Risky input should stop at input guard before Agent and Tool execution."""
    model = E2EFakeModel(responses=[AIMessage(content="不应该执行")])
    runtime = build_agent(llm=model)

    answer = run_agent_turn(
        runtime,
        "这个药能治好我的皮炎吗？",
        input_guard=lambda text: _input_guard_with_trace(model, text),
        output_guard=lambda text: _output_guard_with_trace(model, text),
    )

    assert model.trace == ["input_guard"]
    assert "请咨询医生或药师" in answer
    assert runtime.get("last_result") is None


def test_e2e_tool_error_response_does_not_crash_agent() -> None:
    """Tool error responses should not crash the Agent turn."""
    model = E2EFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "compare_product",
                        "args": {"sku_ids": ""},
                        "id": "e2e_compare_error_call",
                    }
                ],
            ),
            AIMessage(content="工具返回错误，请检查 SKU 输入。"),
        ]
    )
    runtime = build_agent(llm=model)

    answer = run_agent_turn(
        runtime,
        "帮我比较商品",
        input_guard=lambda text: _input_guard_with_trace(model, text),
        output_guard=lambda text: _output_guard_with_trace(model, text),
    )
    tool_messages = _tool_messages(runtime)
    tool_output = json.loads(tool_messages[0].content)

    assert model.trace == ["input_guard", "agent", "compare_product", "output_guard"]
    assert "error" in tool_output
    assert "工具返回错误" in answer
