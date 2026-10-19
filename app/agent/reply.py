"""Ephemeral reply intent, derived from user expressions and existing state.

This is neither another session state nor a fact store. Unknown intentions stay
general; only whole, explicit closing utterances bypass language generation.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from app.agent.facts import ToolFact
from app.agent.state import FIELD_TERMS, SessionState, normalize_condition
from app.safety.guard import PRODUCT_DISCLAIMER


REPLY_CONTEXT_PREFIX = "CURRENT_REPLY_POLICY\n"
CLOSING_REPLY = "好的，先聊到这里。"
EXECUTION_STOP_REPLY = "本轮查询未能完成，已停止继续尝试。请核对商品 SKU 或稍后重新查询；需要帮助可联系人工客服。"
EXECUTION_POLICY_STOP_REPLY = "无法确认发货时效或税费规则，已停止无关商品查询。请联系平台客服核实。"
EXECUTION_TIMEOUT_REPLY = "本轮查询超时，已停止继续尝试，未完成的结果不会用于回答。请稍后重新查询；需要帮助可联系人工客服。"
EXECUTION_BUSY_REPLY = "此前的调用仍未结束，本轮暂不能继续查询。请稍后重试；需要帮助可联系人工客服。"
_CLOSING = re.compile(
    r"(?:(?:好的|好|嗯|知道了|明白了|了解了|不用了|不需要了|先这样|先到这里|暂时不用|不买了|谢谢)(?:了)?)+"
)
_CONCERN = re.compile(r"有点贵|太贵|超(?:出)?预算|超出我的预算|买不起|价格偏高|预算不够")
_HESITATION = re.compile(r"再想想|考虑一下|还没决定|有点犹豫")
_CORRECTION = re.compile(r"不是\s*S\d{3,}|我说的是|你(?:理解|看|听)错|搞错了|更正", re.I)
_EXPLICIT_QUERY = re.compile(r"多少钱|多少元|价格(?:是|为)?多少|规格(?:是|为)?什么|什么规格|比较|对比|区别|差别|哪个|查询|查一下")
_COMPARISON_INTENT = re.compile(r"比较|对比|区别|差别|哪个|顺序|怎么选|如何选")
COMPARISON_FIELDS = {
    "name": "名称", "category": "类目", "price": "价格", "spec": "规格",
    "product_desc": "商品描述", "suitable_crowd": "适用人群", "taboo": "禁忌",
}
COMPARISON_USAGE_NOTICE = "当前资料未提供具体用法及剂量，无法确认；请核对商品说明书或咨询医生或药师。"
COMPARISON_NEXT_STEP = "需要继续核对哪款的规格或注意事项？个人使用问题请联系人工客服或咨询医生或药师。"


@dataclass(frozen=True)
class ReplyPlan:
    intent: str
    basis: str
    skus: tuple[str, ...]
    fields: tuple[str, ...]
    correction: bool = False
    repeated: bool = False
    budget_concern: bool = False
    requires_tool_evidence: bool = False
    comparison_guidance: bool = False

    def model_context(self) -> dict[str, Any]:
        return asdict(self)


def comparison_guidance_requested(text: str, state: SessionState) -> bool:
    """Complete comparisons only; focused field/business questions stay focused."""
    return (state.current_sku is None and len(state.comparison_skus) == 2
            and bool(_COMPARISON_INTENT.search(text) or _comparison_edit(text, state))
            and not any(t in text for terms in FIELD_TERMS.values() for t in terms)
            and not re.search(r"成分|含有|添加|不含|产地|保质期|订单|物流|税|优惠|促销", text))


def _comparison_edit(text: str, state: SessionState) -> bool:
    """Only explicit edits to an existing ordered comparison trigger replanning."""
    return (not state.clarification and (
        any(state.condition_changes.get(k) for k in ("added", "removed", "focused"))
        or bool(re.search(r"我说错了|更正|改为|改成|换成", text)
                and re.search(r"S\d{3,}.*S\d{3,}", text, re.I))
    ))


def _matching_crowd_phrases(text: str, crowd: str) -> tuple[str, ...]:
    """Literal positive phrase overlap, not inferred health/semantic suitability."""
    compact = re.sub(r"\s+", "", text)
    matches = []
    for phrase in re.split(r"[、，,；;。]", crowd):
        phrase = re.sub(r"人群$", "", re.sub(r"\s+", "", phrase))
        if len(phrase) < 2:
            continue
        for match in re.finditer(re.escape(phrase), compact):
            if not re.search(r"(?:不|没有|并非|不是|不再)(?:会|再)?$", compact[:match.start()]):
                matches.append(phrase)
                break
    return tuple(matches)


def render_comparison_guidance(text: str, state: SessionState, facts: tuple[ToolFact, ...]) -> str:
    """Render verified fields in user order; no SKU-specific facts or model prose.

    Callers supply only fresh, validated current-turn facts. A source crowd
    phrase shared with the request is a label association, never an individual
    medical recommendation. Missing sources are not negative product facts.
    """
    skus = state.comparison_skus
    values = {sku: {f.field: str(f.value) for f in facts if f.sku == sku} for sku in skus}
    # A removed scene may remain in the current utterance and chat history.
    # Once explicit conditions exist, neither is a source of active preferences.
    if "conditions" in state.sources:
        focus = state.primary_condition
        matches = {sku: tuple(phrase for phrase in re.split(r"[、，,；;。]", values[sku].get("suitable_crowd", ""))
                              if focus and normalize_condition(re.sub(r"人群$", "", phrase)) == focus)
                   for sku in skus}
    else:
        matches = {sku: _matching_crowd_phrases(text, values[sku].get("suitable_crowd", "")) for sku in skus}
    associated = [sku for sku in skus if matches[sku]]
    if not all(values.values()):
        opening = f"关于{'与'.join(skus)}，当前缺少部分商品的有效资料，无法完成可靠比较，也无法判断个人适用性。"
    elif associated:
        clauses = [f"{sku}与“{'、'.join(matches[sku])}”场景存在直接文字关联" for sku in associated]
        for sku in skus:
            if not matches[sku]:
                clauses.append(f"{sku}的人群描述未明确提及这一场景" if values[sku].get("suitable_crowd")
                               else f"{sku}的人群描述缺失，无法判断场景关联")
        opening = "仅按资料中的人群描述，" + "；".join(clauses) + "；无法据此判断个人适用性。"
    else:
        opening = f"仅按现有资料，{'与'.join(skus)}无法判断场景关联；可先核对下列定位差异，不能据此判断个人适用性。"
    sections = [opening]
    for sku in skus:
        lines = []
        for field, label in COMPARISON_FIELDS.items():
            value = values[sku].get(field)
            lines.append(f"{sku}本地资料中的{label}：{value}。" if value
                         else f"{sku}资料未提供{label}，无法确认。")
        sections.append("\n".join(lines))
    sections.extend((COMPARISON_USAGE_NOTICE, PRODUCT_DISCLAIMER, COMPARISON_NEXT_STEP))
    return "\n\n".join(sections)


def plan_reply(
    text: str, state: SessionState, previous: SessionState,
    last_turn: dict[str, Any] | None,
) -> ReplyPlan:
    """Use explicit signals, never infer a new budget, SKU or product benefit."""
    compact = re.sub(r"[\s，,。！？!?；;、]+", "", text)
    closing = _CLOSING.fullmatch(compact) is not None
    concern = _CONCERN.search(text) is not None
    correction = _CORRECTION.search(text) is not None
    fields = tuple(name for name, terms in FIELD_TERMS.items() if any(t in text for t in terms))
    if not fields and state.query_field:
        fields = (state.query_field,)
    comparison_context = state.current_sku is None and len(state.comparison_skus) == 2
    comparing = comparison_context and bool(fields or _COMPARISON_INTENT.search(text) or _comparison_edit(text, state))
    skus = state.comparison_skus if comparison_context else ((state.current_sku,) if state.current_sku else ())
    previous_skus = (previous.comparison_skus if previous.current_sku is None
                     else (previous.current_sku,))
    previous_fields = tuple(name for name, terms in FIELD_TERMS.items()
                            if last_turn and any(t in last_turn["user_input"] for t in terms))
    if not previous_fields and previous.query_field:
        previous_fields = (previous.query_field,)
    previous_safe = bool(last_turn and not last_turn["input_check"].get("risk")
                         and last_turn.get("output_check")
                         and last_turn["output_check"].get("safe"))
    repeated = bool(previous_safe and fields and skus and skus == previous_skus
                    and fields == previous_fields
                    and not (_CONCERN.search(last_turn["user_input"])
                             and not _EXPLICIT_QUERY.search(last_turn["user_input"])))

    if closing:
        intent, basis = "closing", "explicit_whole_utterance"
    elif state.clarification:
        intent, basis = "clarification", "unresolved_session_state"
    elif correction and not (comparing and not fields):
        intent, basis = "correction", "explicit_correction"
    elif concern and not _EXPLICIT_QUERY.search(text):
        intent, basis = "budget_concern", "explicit_price_concern"
    elif comparing:
        intent, basis = "comparison", "ordered_session_comparison"
    elif fields:
        intent, basis = "field_query", "explicit_field_or_state_followup"
    elif concern:
        intent, basis = "budget_concern", "explicit_price_concern"
    elif _HESITATION.search(text):
        intent, basis = "hesitation", "explicit_hesitation_not_closing"
    else:
        intent, basis = "general", "no_deterministic_intent"
    return ReplyPlan(
        intent=intent, basis=basis, skus=skus, fields=fields,
        correction=correction, repeated=repeated, budget_concern=concern,
        requires_tool_evidence=(intent in ("comparison", "field_query")
                                or (intent == "correction" and bool(fields))),
        comparison_guidance=(intent == "comparison" and comparison_guidance_requested(text, state)),
    )
