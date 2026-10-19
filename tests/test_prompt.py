"""Tests for Agent system prompt configuration.

These tests verify that Phase 6.1 business behavior constraints are loaded from
the dedicated prompt module.
"""

from app.agent.prompt import SYSTEM_PROMPT


def test_system_prompt_loads() -> None:
    """System prompt should be available as non-empty text."""
    assert isinstance(SYSTEM_PROMPT, str)
    assert SYSTEM_PROMPT.strip()


def test_system_prompt_contains_tool_guidance() -> None:
    """Prompt should mention core OTC guide role and registered tools."""
    assert "OTC" in SYSTEM_PROMPT
    assert "导购" in SYSTEM_PROMPT
    assert "search_product" in SYSTEM_PROMPT
    assert "compare_product" in SYSTEM_PROMPT


def test_system_prompt_contains_compliance_keywords() -> None:
    """Prompt should explicitly list key compliance terms."""
    assert "治疗" in SYSTEM_PROMPT
    assert "治愈" in SYSTEM_PROMPT
    assert "根治" in SYSTEM_PROMPT
