"""End-to-end Agent flow tests with mocked function-calling LLM responses.

These tests verify the Phase 5.2 Agent -> LLM decision -> Tool execution path
without calling a real model API, adding Memory, or integrating Safety Guard.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from app.agent.agent import build_agent


class ToolCallingFakeModel(FakeMessagesListChatModel):
    """Fake chat model that supports LangChain Agent tool binding."""

    def bind_tools(
        self,
        tools: list[Any],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> "ToolCallingFakeModel":
        """Return self so create_agent can proceed with mocked tool calls."""
        _ = tools, tool_choice, kwargs
        return self


def _run_agent_flow(user_input: str, tool_name: str, tool_args: dict[str, Any]) -> dict[str, Any]:
    """Run one mocked Agent flow and return parsed tool output."""
    llm = ToolCallingFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": tool_name,
                        "args": tool_args,
                        "id": "mock_tool_call",
                    }
                ],
            ),
            AIMessage(content="mock final answer"),
        ]
    )
    runtime = build_agent(llm=llm)
    result = runtime["agent"].invoke(
        {"messages": [{"role": "user", "content": user_input}]}
    )
    tool_message = _find_tool_message(result["messages"], tool_name)

    return json.loads(tool_message.content)


def _find_tool_message(messages: list[BaseMessage], tool_name: str) -> ToolMessage:
    """Find the ToolMessage produced by the requested tool."""
    for message in messages:
        if isinstance(message, ToolMessage) and message.name == tool_name:
            return message
    raise AssertionError(f"ToolMessage not found for tool: {tool_name}")


def test_agent_flow_calls_search_product_for_price_question() -> None:
    """Agent flow should call search_product and return SKU price data."""
    tool_output = _run_agent_flow(
        user_input="S001多少钱？",
        tool_name="search_product",
        tool_args={"query": "S001"},
    )

    assert tool_output["results"]
    product = tool_output["results"][0]
    assert product["sku_id"] == "S001"
    assert product["name"] == "演示星屿营养片"
    assert product["price"] == "176 元"


def test_agent_flow_calls_compare_product_for_compare_question() -> None:
    """Agent flow should call compare_product and return comparison blocks."""
    tool_output = _run_agent_flow(
        user_input="帮我比较S001和S002",
        tool_name="compare_product",
        tool_args={"sku_ids": "S001,S002"},
    )

    assert "basic_difference" in tool_output
    assert "suitable_crowd_difference" in tool_output
    assert "taboo_difference" in tool_output
    assert "product_position_difference" in tool_output
