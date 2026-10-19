"""Conservative final boundaries for the currently available local-only tools.

This is a bounded rule check, not a semantic entailment model. Volatile business
topics use evidence-only rendering; recognizable unsupported product assertions
fall back to verified requested fields. Arbitrary paraphrases remain a limit.
"""

from __future__ import annotations

import json
import re
import unicodedata
from decimal import Decimal, InvalidOperation
from typing import Any

from langchain_core.messages import ToolMessage

from app.agent.facts import ToolFact
from app.agent.reply import comparison_guidance_requested, render_comparison_guidance
from app.agent.state import FIELD_TERMS, SKU_PATTERN
from app.safety.guard import DISCLAIMER_MESSAGE, SAFETY_FALLBACK_MESSAGE


_NUMERAL = r"(?:[-负]?(?:\d+(?:\.\d+)?|[零〇一二两三四五六七八九十百千万壹贰叁肆伍陆柒捌玖拾佰仟点]+))"
_MONEY = re.compile(rf"(?:[¥￥]|人民币){_NUMERAL}(?:元|块钱?)?|{_NUMERAL}(?:元|块钱?)")
_BARE_PRICE = re.compile(rf"(?:售价|价格|单价|标价)(?:是|为|只要|仅需|[:：])?({_NUMERAL})")
_BUDGET_AMOUNT = re.compile(rf"^(?:你的|您的|按|根据|目前|当前)?预算(?:上限)?(?:仍按|改成|改为|是|为|最多|不超过)?{_NUMERAL}(?:元|块钱?)(?:以内|以下)?")

# Recognize quantities separately from validating their SKU/field provenance.
# Keep numeral spelling literal: unsupported conversions use the source quote.
_SPEC_UNIT = r"(?:毫升|毫克|千克|公斤|克|升|kg|mg|ml|g|l|粒|片|丸|贴|支|瓶|盒|袋|包)"
_SPEC_PACKAGE = r"(?:瓶|支|盒|袋|包|板|罐|箱|套)"
_SPEC_QUANTITY = re.compile(
    rf"(?<![\d.零〇一二两三四五六七八九十百千万壹贰叁肆伍陆柒捌玖拾佰仟点负-])"
    rf"(?:每(?P<prefix_package>{_SPEC_PACKAGE}))?"
    rf"(?P<number>{_NUMERAL})(?P<unit>{_SPEC_UNIT})"
    rf"(?:/(?P<suffix_package>{_SPEC_PACKAGE}))?(?![a-zA-Z])",
    re.I,
)
_SPEC_LABEL = re.compile(r"规格|容量")
_UNPARSED_SPEC = re.compile(
    rf"(?:规格|容量)(?:是|为|[:：])?(?:大约|约)?(?:{_NUMERAL}|半|若干|数)"
)
_CONTAINER_UNITS = {"支", "瓶", "盒", "袋", "包"}

# Field semantics are independent of product identity or ingredient vocabulary.
# The current tool schema has no ingredients field. Descriptions/names and
# other SKUs cannot license either a positive or a negative composition claim.
_INGREDIENT_VERB = r"(?:不含(?:有)?|未添加|富含|蕴含|包含|含(?:有)?|添加(?:了)?|由.+?(?:组成|构成))"
_INGREDIENT_REQUEST = re.compile(r"成分|配方|" + _INGREDIENT_VERB)
_PRODUCT_REFERENCE = re.compile(
    SKU_PATTERN + r"|本品|本产品|该商品|该产品|这款|这件商品|它|保健品|软膏|药膏", re.I,
)
MISSING_INGREDIENT_NOTICE = "当前商品资料未提供成分字段，无法确认具体成分。"
_PRESENCE = re.compile(r"(?:里面|其中|内)?(?:没)?有")
_OTHER_PRODUCT_FIELDS = re.compile(
    r"^(?:现货|库存|货|优惠|折扣|促销|税|关税|运费|价格|规格|容量|配送|发货|"
    r"禁忌|注意事项|说明|介绍|描述|标签|说明书|适用人群|副作用|效果|哪些|什么|几|多少|\d)"
)
_INGREDIENT_CONFIRMATION = re.compile(
    r"(?:它|这款|该商品|这个)?(?:也|确实|真的)?(?:有|含有|含|是这样)(?:吗|么)?|"
    r"(?:你|您)?(?:能)?(?:确定|确认)(?:吗|么)?|(?:是|对)(?:吗|么)|"
    rf"(?:那|那么)?(?:{SKU_PATTERN}|它|这款|前者|后者)(?:也)?(?:呢|吗)",
    re.I,
)
_INGREDIENT_AFFIRMATION = re.compile(
    r"(?:(?:是的|对的|没错|是|对)[，,]?)?(?:它|这款|该商品|这个)?"
    r"(?:也|确实|真的)?(?:有|含有|含)(?:的)?[。.!！]?|(?:是的|对的|没错|是|对)[。.!！]?"
)

BUSINESS_TOPICS = {
    "order": r"订单|物流|运单|快递|签收",
    "stock": (rf"库存|现货|有货|无货|缺货|售罄|售完|卖完|补货|备货|"
              rf"(?:还剩|仅剩|剩余|只剩|剩下|剩){_NUMERAL}(?:件|支|瓶|盒|箱|包|套)|"
              r"(?:可以|可|能|支持|放心)(?:直接)?(?:拍下|拍|下单|购买)"),
    "promotion": r"促销|优惠|折扣|打折|满减|降价|保价|价格保证|最低价|保证.*价|价格.*不变|赠品|优惠券|[一二三四五六七八九\d]折",
    "delivery": r"发货|配送|出库|送达|到货|多久到|包邮|小时内|工作日|[明后]天.*(?:到|收)",
    "platform": r"平台规则|税费|被税|免税|关税|通关|保税仓|退换|退货|售后",
}
FIELD_LABELS = {"price": "价格", "spec": "规格", "suitable_crowd": "适用人群",
                "taboo": "禁忌", "product_desc": "商品描述", "name": "名称", "category": "类目"}
BUSINESS_NOTICES = {
    "order": "未接入订单或物流查询，无法确认具体订单的出库、物流进度或送达时间。",
    "stock": "没有实时库存数据，无法确认当前有货或无货。",
    "promotion": "没有可靠的促销或保价来源，无法确认当前优惠、折扣或价格保证。",
    "delivery": "无法确认具体出库或送达时效；本地配送说明不是履约保证。",
    "platform": "没有可靠的平台规则来源，无法确认保税仓、通关、税费或售后当前适用规则，请向平台客服核实。",
}
LOCAL_QUALIFICATION = "以上仅是本地资料记录；数据版本和缓存有效期不代表实时库存、实时促销或具体订单履约状态。"


def _normal(text: str) -> str:
    return re.sub(r"[\s\u200b-\u200f\u2060\ufeff]+", "", unicodedata.normalize("NFKC", text))


def pure_shipping_tax_question(text: str) -> bool:
    """Full-input policy intent only; product names/pronouns/mixed tasks stay open.

    This boundary uses the current human request, never a previously selected
    SKU. Unknown wording remains subject to the existing task and fact checks.
    """
    text = re.sub(r"[，,。！？!?；;：:]", "", _normal(text))
    return bool(re.search(r"发货|出库|送达|到货|多久到|配送|税|保税仓", text)
                and re.fullmatch(
                    r"(?:请问|麻烦|请|想问|咨询|你们|现在|今天|一般|通常|正常|"
                    r"下单|后|等|多久|多长时间|几天|什么时候|发货|出库|送达|到货|配送|到|"
                    r"保税仓|税费|关税|被税|收税|免税|税|会不会|是否|会|不会|能|要|需要|确定|"
                    r"怎么|如何|算|计算|收|收取|多少|钱|呢|吗|啊|呀|的|了|和|以及|及|从|由)+",
                    text))


def _ingredient_claim(text: str, facts: tuple[ToolFact, ...]) -> bool:
    """Recognize a field assertion before stripping any source field values.

    No substance-specific whitelist and no substring-based evidence matching.
    Omitted subjects and verified product names are recognized; unrelated prose
    such as '报告含有三个章节' does not become a product composition claim.
    """
    names = [_normal(str(f.value)) for f in facts if f.field == "name"]
    for clause in re.split(r"[。；;\n]+", _normal(text)):
        if re.search(r"(?:成分|配方)(?:是|为|[:：])", clause):
            return True
        for match in re.finditer(_INGREDIENT_VERB + "|" + _PRESENCE.pattern, clause):
            subject = clause[:match.start()]
            if _PRESENCE.fullmatch(match.group()):
                # '有现货/优惠/规格' is a separate field, not composition.
                obj = clause[match.end():].lstrip("了")
                if not obj or _OTHER_PRODUCT_FIELDS.match(obj):
                    continue
                # Generic '有' requires a complete product subject, not merely
                # a product word somewhere in '该商品缺少有效工具依据'.
                plain_subject = re.sub(r"^(?:是的|对的|没错)[，,]?", "", subject)
                plain_subject = re.sub(r"(?:也|确实|真的)$", "", plain_subject)
                if not plain_subject and not match.group().startswith(("里面", "其中", "内")):
                    continue
                if (plain_subject and not _PRODUCT_REFERENCE.fullmatch(plain_subject)
                        and plain_subject not in names):
                    continue
            if (not subject or _PRODUCT_REFERENCE.search(subject)
                    or any(name and name in subject for name in names)):
                return True
    return False


def _ingredient_request(runtime: dict[str, Any], text: str, facts: tuple[ToolFact, ...]) -> bool:
    if _INGREDIENT_REQUEST.search(_normal(text)) or _ingredient_claim(text, facts):
        return True
    # The exact program notice is only a topic signal. It is never ingredient
    # evidence, and a new price/spec/product request does not inherit this topic.
    last = runtime["memory"].get_last_turn()
    return bool(last and MISSING_INGREDIENT_NOTICE in last["displayed_reply"]
                and _INGREDIENT_CONFIRMATION.fullmatch(_normal(text).strip("，,。！？!?；;")))


def current_tool_facts(runtime: dict[str, Any]) -> tuple[ToolFact, ...]:
    """Resolve current execution scopes through the store, never trust prose.

    Stale/failed scopes and other SKUs are excluded even if an old price exists.
    A fuzzy result stays a query result; it never becomes a complete stock list.
    """
    memory = runtime["memory"]
    state = memory.get_state()
    selected = {state.current_sku} if state.current_sku else set(state.comparison_skus)
    facts = []
    for message in runtime.get("last_result", {}).get("messages", []):
        if not isinstance(message, ToolMessage) or message.status != "success":
            continue
        try:
            evidence = json.loads(message.content).get("_tool_evidence", {})
            args = evidence.get("query_args")
            if evidence.get("status") not in ("success", "partial") or not isinstance(args, dict):
                continue
            for fact in memory.facts.valid_query_facts(message.name, args):
                if fact.sku in selected and fact.source_kind == "local_json_snapshot":
                    facts.append(fact)
        except (ValueError, TypeError, AttributeError):
            continue
    return tuple(facts)


def _number_value(text: str) -> Decimal | None:
    """Parse bounded common Chinese/Arabic monetary notation, not prose."""
    text = text.translate(str.maketrans("〇两壹贰叁肆伍陆柒捌玖拾佰仟", "零二一二三四五六七八九十百千"))
    if text.startswith(("负", "-")):
        value = _number_value(text[1:])
        return -value if value is not None else None
    try:
        return Decimal(text)
    except InvalidOperation:
        pass
    digits = {c: i for i, c in enumerate("零一二三四五六七八九")}
    units = {"十": 10, "百": 100, "千": 1000}
    whole, dot, fraction = text.partition("点")
    if not whole or (dot and (not fraction or any(c not in digits for c in fraction))):
        return None
    if all(c in digits for c in whole):
        value = int("".join(str(digits[c]) for c in whole))
    else:
        total = section = number = 0
        for c in whole:
            if c in digits:
                number = digits[c]
            elif c in units:
                section += (number or 1) * units[c]
                number = 0
            elif c == "万":
                total += (section + number) * 10000
                section = number = 0
            else:
                return None
        value = total + section + number
    return Decimal(str(value) + ("." + "".join(str(digits[c]) for c in fraction) if dot else ""))


def _amount_value(text: str) -> Decimal | None:
    match = re.search(_NUMERAL, text)
    return _number_value(match.group()) if match else None


def _checked_price_residue(clause: str, active: str | None, facts: tuple[ToolFact, ...]) -> tuple[bool, str]:
    """Each amount requires the price field of its own selected SKU.

    Only a literal budget amount is exempt, never the rest of that sentence.
    Ambiguous shared/multiple-SKU clauses fail conservatively.
    """
    residue = _BUDGET_AMOUNT.sub("", clause)
    amounts = list(_MONEY.finditer(residue))
    bare = list(_BARE_PRICE.finditer(residue))
    labelled_fields = {f for f, terms in FIELD_TERMS.items() if any(t in residue for t in terms)}
    prices = [f for f in facts if f.sku == active and f.field == "price"]
    for match in [*amounts, *bare]:
        # A spec carrying a price-shaped number is still a field mismatch.
        if labelled_fields and "price" not in labelled_fields:
            return False, residue
        amount = _amount_value(match.group(1) if match.re is _BARE_PRICE else match.group())
        if amount is None or not any(
            _MONEY.fullmatch(_normal(str(f.value)))
            and _amount_value(_normal(str(f.value))) == amount for f in prices
        ):
            return False, residue
    # Already checked as prices; do not let generic numeric matching reinterpret
    # them as another field. Keep all other claims for subsequent inspection.
    return True, _BARE_PRICE.sub("", _MONEY.sub("", residue))


def _spec_key(match: re.Match[str]) -> tuple[str, str, str | None, str | None]:
    """Compare complete quantity/unit/package structure, not source substrings."""
    return (match["number"], match["unit"].casefold(),
            match["prefix_package"], match["suffix_package"])


def _checked_spec_residue(clause: str, active: str | None,
                          facts: tuple[ToolFact, ...]) -> tuple[bool, str]:
    """Only the active SKU's spec field licenses a recognized specification.

    Exact simple source expressions are supported; no numeral/unit conversion or
    partial extraction from compound specifications is assumed. Other forms can
    be rendered by the existing attributed-source replacement. A bare purchase
    count such as '每月购买一瓶' is not a specification without a label/package.
    """
    supported = set()
    for fact in facts:
        if fact.sku == active and fact.field == "spec":
            source = _SPEC_QUANTITY.fullmatch(_normal(str(fact.value)))
            if source:
                supported.add(_spec_key(source))
    labelled = _SPEC_LABEL.search(clause) is not None
    matches = []
    for match in _SPEC_QUANTITY.finditer(clause):
        packaged = bool(match["prefix_package"] or match["suffix_package"])
        # '每次1片' is a dose, not package content. The existing medical review
        # and exact product_desc check still apply; this is no label whitelist.
        dosing = re.search(r"(?:每次|一次|每日|每天|每周|每晚)$", clause[:match.start()])
        if not labelled and not packaged and dosing:
            continue
        if labelled or packaged or match["unit"].casefold() not in _CONTAINER_UNITS:
            matches.append(match)
    if any(_spec_key(m) not in supported for m in matches):
        return False, clause
    residue = clause
    for match in reversed(matches):
        residue = residue[:match.start()] + residue[match.end():]
    # A labelled numeric-looking value outside the supported grammar is not
    # verified merely because the quantity recognizer could not parse it.
    return not _UNPARSED_SPEC.search(residue), residue


def _boundary_residue(text: str, facts: tuple[ToolFact, ...]) -> str:
    """Remove only whole reviewed notices and exact, attributed local quotes.

    Not a negative-word whitelist: attached promises remain reviewable. A local
    stock/delivery quote also requires the explicit non-realtime qualification.
    """
    residue = _normal(text)
    statements = list(BUSINESS_NOTICES.values()) + [LOCAL_QUALIFICATION]
    qualified = _normal(LOCAL_QUALIFICATION) in residue
    for fact in facts:
        if _ingredient_claim(str(fact.value), facts):
            # An attributed description is still not the missing field.
            continue
        if fact.field not in ("stock", "delivery") or qualified:
            label = FIELD_LABELS.get(fact.field, {"stock": "库存", "delivery": "配送"}.get(fact.field, fact.field))
            statements.append(f"{fact.sku}本地资料中的{label}：{fact.value}。")
    for statement in sorted(statements, key=len, reverse=True):
        residue = re.sub(r"(?:^|(?<=[。！？!?\n]))" + re.escape(_normal(statement)), "", residue)
    return residue


def _unsupported_product_claim(text: str, facts: tuple[ToolFact, ...], skus: tuple[str, ...]) -> bool:
    """Check numeric/labelled/common property assertions per SKU, conservatively.

    Exact source wording is allowed; unknown paraphrases of medical/property
    claims can require verification. This does not claim all-language coverage.
    """
    if _ingredient_claim(text, facts):
        return True
    active = skus[0] if len(skus) == 1 else None
    for clause in re.split(r"[。；;\n]+", _normal(text)):
        pieces = []
        for part in clause.split(","):
            mentioned = re.findall(SKU_PATTERN, part, re.I)
            if mentioned:
                active = mentioned[-1].upper() if len(set(mentioned)) == 1 else None
            price_ok, residue = _checked_price_residue(part, active, facts)
            if not price_ok:
                return True
            spec_ok, residue = _checked_spec_residue(residue, active, facts)
            if not spec_ok:
                return True
            pieces.append(residue)
        residue = ",".join(pieces)
        mentioned = re.findall(SKU_PATTERN, clause, re.I)
        owner = active if len(set(mentioned)) <= 1 else None
        labelled_fields = {field for field, terms in FIELD_TERMS.items()
                           if any(term in clause for term in terms)}
        if owner:
            for fact in sorted((f for f in facts if f.sku == owner
                                and (not labelled_fields or f.field in labelled_fields)),
                               key=lambda f: len(str(f.value)), reverse=True):
                value = _normal(str(fact.value)).rstrip("。；;")
                # A quoted 198元 must not license a hallucinated 1198元.
                start = r"(?<![\d.])" if value[:1].isdigit() else ""
                end = r"(?!\d)" if value[-1:].isdigit() else ""
                residue = re.sub(start + re.escape(value) + end, "", residue)
        if re.search(r"\d+(?:\.\d+)?(?:元|[gG克粒片毫升]|ml|mL)|无副作用|零副作用|"
                     r"(?:名称|功效|作用|产地|保质期|成分|材质)(?:是|为|[:：])[^，,]+|"
                     r"(?:适用人群|适合人群|适合)[:：]?[^，,:：]+|无禁忌|"
                     r"(?:能够|可以|能)(?:用于|改善|增强|提升|缓解)|"
                     r"提升|增强|改善|缓解|促进|美白|减肥|防脱|抗衰", residue):
            return True
    return False


def apply_fact_boundary(runtime: dict[str, Any], user_input: str, text: str, *, verify_only: bool = False,
                        reply_kind: str = "answer", clarification_query: str | None = None) -> dict[str, Any]:
    """Return the final candidate plus a diagnostic decision, without writing facts."""
    if reply_kind == "execution_failure" and runtime.get("last_result", {}).get("execution", {}).get("stopped"):
        from app.agent.reply import (
            EXECUTION_STOP_REPLY, EXECUTION_TIMEOUT_REPLY, EXECUTION_BUSY_REPLY,
            EXECUTION_POLICY_STOP_REPLY,
        )
        # A program branch plus exact reviewed text, never a model-declared type.
        execution = runtime["last_result"]["execution"]
        trusted = execution.get("exit_reply", EXECUTION_STOP_REPLY)
        if trusted not in (EXECUTION_STOP_REPLY, EXECUTION_TIMEOUT_REPLY, EXECUTION_BUSY_REPLY,
                           EXECUTION_POLICY_STOP_REPLY):
            trusted = EXECUTION_STOP_REPLY
        verified = text == trusted
        return {"decision": "unchanged" if verified else "bounded", "text": trusted,
                "verified": verified, "topics": [], "reply_kind": "execution_failure"}
    state = runtime["memory"].get_state()
    skus = (state.current_sku,) if state.current_sku else state.comparison_skus
    facts = current_tool_facts(runtime)
    if reply_kind == "clarification" and not verify_only:
        # The application chooses this branch, never a model's claimed type.
        # Audit content, not the user's unanswered attribute request. Only exact
        # candidate names from the fresh lookup may be quoted without selection.
        names = ()
        if clarification_query is not None and state.name_query == clarification_query:
            names = tuple(f for f in runtime["memory"].facts.valid_query_facts(
                "search_product", {"query": clarification_query},
            ) if f.field == "name")
        residue = _boundary_residue(text, names)
        if (not _unsupported_product_claim(residue, facts, skus)
                and not any(re.search(pattern, residue) for pattern in BUSINESS_TOPICS.values())):
            return {"decision": "unchanged", "text": text, "topics": [], "reply_kind": "clarification"}
        # Restore a plain program-authored question after a filter adds an
        # unsupported claim. It still traverses the normal replacement reviews.
        if state.clarification:
            trusted = _boundary_residue(state.clarification, facts)
            if (not _unsupported_product_claim(trusted, facts, skus)
                    and not any(re.search(pattern, trusted) for pattern in BUSINESS_TOPICS.values())):
                return {"decision": "bounded", "text": state.clarification,
                        "topics": [], "reply_kind": "clarification"}
    if verify_only:
        residue = _boundary_residue(text, facts)
        verified = (not _unsupported_product_claim(residue, facts, skus)
                    and not any(re.search(pattern, residue) for pattern in BUSINESS_TOPICS.values()))
        ingredient_topic = (_ingredient_request(runtime, user_input, facts)
                            or MISSING_INGREDIENT_NOTICE in runtime.get("last_fact_check", {}).get("text", ""))
        if ingredient_topic and _INGREDIENT_AFFIRMATION.fullmatch(_normal(text)):
            verified = False
        if (comparison_guidance_requested(user_input, state)
                and text not in (DISCLAIMER_MESSAGE, SAFETY_FALLBACK_MESSAGE)):
            # Rebuild from freshly resolved evidence: identity, fields, source
            # expiry, scene association and required sections must still agree.
            verified = verified and text == render_comparison_guidance(user_input, state, facts)
        return {"decision": "unchanged" if verified else "rejected", "text": text, "verified": verified}
    if comparison_guidance_requested(user_input, state):
        return {"decision": "bounded", "text": render_comparison_guidance(user_input, state, facts),
                "topics": [], "reply_kind": "product_comparison"}
    topics = {key for key, pattern in BUSINESS_TOPICS.items()
              if re.search(pattern, _normal(user_input) + "\n" + _normal(text))}
    unsupported = _unsupported_product_claim(text, facts, skus)
    ingredients = _ingredient_request(runtime, user_input, facts) or _ingredient_claim(text, facts)
    # These requests need exact local attributes, not guessed paraphrases. Some
    # fields (e.g. ingredients/origin) have no source in the current tool schema.
    attribute_request = bool(re.search(r"功效|作用|成分|材质|产地|保质期|适用范围|适用人群|适合谁|适合什么人", user_input))
    if not topics and not unsupported and not attribute_request and not ingredients:
        return {"decision": "unchanged", "text": text, "topics": []}

    parts = []
    if ingredients:
        parts.append(MISSING_INGREDIENT_NOTICE)
    if "order" in topics:
        parts.append(BUSINESS_NOTICES["order"])
    if "stock" in topics:
        parts.append(BUSINESS_NOTICES["stock"])
    if "promotion" in topics:
        parts.append(BUSINESS_NOTICES["promotion"])
    if "delivery" in topics and "order" not in topics:
        parts.append(BUSINESS_NOTICES["delivery"])
    if "platform" in topics:
        parts.append(BUSINESS_NOTICES["platform"])

    # Include only asked-for fields, in the selected/comparison order. No broad
    # catalog inference and no fallback to older scopes after a failed lookup.
    fields = [f for f, terms in FIELD_TERMS.items() if any(t in user_input for t in terms)]
    if re.search(r"出库|送达|到货|配送说明", user_input) and "delivery" not in fields:
        fields.append("delivery")
    if re.search(r"现货", user_input) and "stock" not in fields:
        fields.append("stock")
    if re.search(r"功效|作用|商品描述|介绍", user_input):
        fields.append("product_desc")
    if not fields and not topics and not ingredients and state.query_field:
        fields.append(state.query_field)
    for sku in skus:
        for field in fields:
            fact = next((f for f in reversed(facts) if f.sku == sku and f.field == field), None)
            if fact:
                label = FIELD_LABELS.get(field, {"stock": "库存", "delivery": "配送"}.get(field, field))
                parts.append(f"{sku}本地资料中的{label}：{fact.value}。")
    if not parts:
        parts.append("当前无法确认该商品信息，缺少对应商品的有效工具依据。")
    # These qualifications apply to all included local fields, including a
    # delivery field that happens to contain a platform name or a time range.
    if any(f in fields for f in ("stock", "delivery")) and any(f.field in fields for f in facts):
        parts.append(LOCAL_QUALIFICATION)
    return {"decision": "bounded", "text": "".join(parts), "topics": sorted(topics),
            "unsupported_product_claim": unsupported}
