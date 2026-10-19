"""Regression tests for independently phrased portfolio scenarios.

The legacy file/function names do not imply official endorsement or permission.

These tests focus on input -> Agent decision -> Tool selection behavior with a
mock LLM. They do not evaluate final copywriting quality.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from pydantic import Field

from app.agent.agent import build_agent, run_agent_turn
from app.safety.guard import check_ai_output, check_user_input


class OfficialCaseFakeModel(FakeMessagesListChatModel):
    """Mock LLM that supports LangChain tool binding and trace recording."""

    trace: list[str] = Field(default_factory=list)

    def bind_tools(
        self,
        tools: list[Any],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> "OfficialCaseFakeModel":
        """Return self so create_agent can bind tools in tests."""
        _ = tools, tool_choice, kwargs
        return self

    def _generate(self, messages: list[BaseMessage], stop: Any = None, **kwargs: Any) -> Any:
        """Record agent and tool execution while returning mocked responses."""
        tool_messages = [message for message in messages if isinstance(message, ToolMessage)]
        if tool_messages and tool_messages[-1].name not in self.trace:
            self.trace.append(str(tool_messages[-1].name))
        elif "agent" not in self.trace:
            self.trace.append("agent")

        return super()._generate(messages, stop=stop, **kwargs)


def _input_guard(model: OfficialCaseFakeModel, text: str) -> dict[str, Any]:
    """Run input guard and record trace."""
    model.trace.append("input_guard")
    return check_user_input(text)


def _output_guard(model: OfficialCaseFakeModel, text: str) -> dict[str, Any]:
    """Run output guard and record trace."""
    model.trace.append("output_guard")
    return check_ai_output(text)


def _tool_calls(runtime: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect AI tool calls from the latest agent result."""
    result = runtime.get("last_result") or {}
    calls: list[dict[str, Any]] = []
    for message in result.get("messages", []):
        if isinstance(message, AIMessage):
            calls.extend(message.tool_calls)
    return calls


def _tool_messages(runtime: dict[str, Any]) -> list[ToolMessage]:
    """Collect ToolMessages from the latest agent result."""
    result = runtime.get("last_result") or {}
    return [
        message
        for message in result.get("messages", [])
        if isinstance(message, ToolMessage)
    ]


def test_official_case_otc_red_line_is_blocked_before_agent() -> None:
    """Risky OTC treatment question should stop at input guard."""
    model = OfficialCaseFakeModel(responses=[AIMessage(content="不应该执行")])
    runtime = build_agent(llm=model)

    answer = run_agent_turn(
        runtime,
        "这款演示外用商品能治好我的皮炎吗？",
        input_guard=lambda text: _input_guard(model, text),
        output_guard=lambda text: _output_guard(model, text),
    )

    assert model.trace == ["input_guard"]
    assert "请咨询医生或药师" in answer
    assert runtime.get("last_result") is None


def test_official_case_compare_recommendation_calls_compare_product() -> None:
    """Two-SKU shopping consultation should call compare_product correctly."""
    model = OfficialCaseFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "compare_product",
                        "args": {"sku_ids": "S001,S002"},
                        "id": "official_compare_call",
                    }
                ],
            ),
            AIMessage(content="这是基于商品差异的导购回答。"),
        ]
    )
    runtime = build_agent(llm=model)

    run_agent_turn(
        runtime,
        "比较S001和S002：经常熬夜这一场景与哪些资料有关？",
        input_guard=lambda text: _input_guard(model, text),
        output_guard=lambda text: _output_guard(model, text),
    )
    calls = _tool_calls(runtime)
    tool_output = json.loads(_tool_messages(runtime)[0].content)

    assert calls
    assert calls[0]["name"] == "compare_product"
    assert calls[0]["args"] == {"sku_ids": "S001,S002"}
    assert "basic_difference" in tool_output
    assert "suitable_crowd_difference" in tool_output
    assert "taboo_difference" in tool_output
    assert "product_position_difference" in tool_output


def test_official_case_platform_rule_does_not_call_product_tools() -> None:
    """Platform logistics and tax question should not call product tools."""
    model = OfficialCaseFakeModel(
        responses=[
            AIMessage(
                content=(
                    "保税仓时效和税费请以商品页、结算页和平台规则为准。"
                )
            )
        ]
    )
    runtime = build_agent(llm=model)

    answer = run_agent_turn(
        runtime,
        "发货需要等多久，保税仓税费能确定吗？",
        input_guard=lambda text: _input_guard(model, text),
        output_guard=lambda text: _output_guard(model, text),
    )
    calls = _tool_calls(runtime)

    assert calls == []
    assert "保税仓" in answer
    assert "平台规则" in answer
