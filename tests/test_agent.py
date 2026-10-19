"""Basic tests for Phase 5.1 Agent assembly.

The tests verify Agent initialization and tool registration with a fake chat
model, so no real LLM request or API key is required.
"""

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.agent.agent import build_agent, get_agent_tools


def test_agent_can_initialize_with_mock_llm() -> None:
    """Agent runtime should initialize with an injected fake LLM."""
    runtime = build_agent(llm=FakeListChatModel(responses=["ok"]))

    assert runtime["status"] == "ready"
    assert runtime["agent"] is not None
    assert runtime["llm"] is not None


def test_agent_tool_registry_contains_product_tools() -> None:
    """Agent tool registry should include search and comparison tools."""
    tool_names = {tool.name for tool in get_agent_tools()}

    assert "search_product" in tool_names
    assert "compare_product" in tool_names


def test_built_agent_runtime_exposes_registered_tools() -> None:
    """Built Agent runtime should expose the registered tool objects."""
    runtime = build_agent(llm=FakeListChatModel(responses=["ok"]))
    tool_names = {tool.name for tool in runtime["tools"]}

    assert tool_names == {"search_product", "compare_product"}
