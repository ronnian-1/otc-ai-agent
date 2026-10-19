"""Rule-based OTC safety guard interfaces.

Input intent and output claims have separate checks. These conservative rules
are not a general natural-language safety classifier. Label dosing is accepted
only against evidence supplied by the caller, never a model's claimed citation.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any


FORBIDDEN_MEDICAL_TERMS = (
    "治疗",
    "治好",
    "治愈",
    "根治",
    "治病",
    "疗效",
    "对症",
    "降压",
    "缓解疾病",
    "改善病情",
    "治疗效果",
)

DIAGNOSIS_INTENT_TERMS = (
    "诊断",
    "确诊",
    "是不是得了",
    "我得了什么病",
)

PERSONAL_MEDICATION_TERMS = (
    "怎么吃药",
    "用什么药",
    "该吃什么",
    "能不能吃",
    "适合我吃",
)

PRODUCT_DISCLAIMER = (
    "本产品为膳食补充剂 / OTC 外用产品，不具备治疗疾病功效，"
    "不能替代药物，身体不适建议咨询医师。"
)
DISCLAIMER_MESSAGE = (
    "该问题可能涉及疾病诊断、治疗效果或个性化用药建议。"
    "请咨询医生或药师，AI 导购不能替代专业医疗建议。"
    + PRODUCT_DISCLAIMER + "个人疾病问题请联系人工客服，由专业人士协助核实。"
)
SAFETY_FALLBACK_MESSAGE = (
    "该回答可能涉及医疗功效承诺或不合规表达。"
    "请以商品页面信息为准，涉及疾病、用药或治疗问题请咨询医生或药师。"
    + PRODUCT_DISCLAIMER + "个人疾病问题请联系人工客服，由专业人士协助核实。"
)

OUTPUT_BLOCK_MESSAGE = "AI 输出包含医疗功效承诺或禁止宣传表达，需要重新生成。"

# Whole clauses only. A negative word anywhere in a paragraph is NOT a waiver.
# Unrecognized negative constructions remain subject to the medical-term check.
_LIMITATION = re.compile(
    r"(?:S\d{3,}(?:的)?[：:]?)?"
    r"(?:(?:禁忌|商品资料|标签(?:说明)?|说明书|资料)(?:中)?"
    r"(?:注明|标注|写明|写着|说明|标明)?[：:]?)?"
    r"(?:虽然)?(?:本品|本产品|该商品|该产品|此商品|这款商品|这款产品)?"
    r"(?:(?:不可|不能|不得|不应)用于疾病治疗|"
    r"(?:不能|无法|不)保证(?:根治|治愈|疗效|有效|见效)|不承诺(?:治愈|疗效))",
    re.I,
)
_DOSE_WARNING = re.compile(
    r"(?:请)?(?:不要|不可|不能|不得|不应|勿)自行"
    r"(?:加量|减量|加倍|增减用量|调整(?:剂量|用量)|增加(?:剂量|用量))"
)
_ABSOLUTE_EFFICACY = re.compile(
    r"(?:保证|确保|包管|包你|保准|一定|肯定|必定|必然|绝对|百分之百|百分百|100%|当然)"
    r".{0,24}?(?:有效|见效|奏效|管用|有用|治好|治愈|根治|康复|痊愈|好起来)|"
    r"(?:难道|岂会|怎么会|怎么可能).{0,24}?(?:无效|没效|不管用|不见效|治不好|不能治好)|"
    r"药到病除|包治|除根|十拿九稳|稳操胜券"
)
_NUMBER = r"(?:\d+(?:\.\d+)?|[一二两三四五六七八九十百半]+)"
_BASE_UNIT = r"(?:片|粒|丸|袋|滴|喷|次|毫克|毫升|克|mg|ml|g)"
_UNIT = rf"(?:{_BASE_UNIT}|支|管|瓶|贴|包)"
_FREQUENCY = rf"(?:每天|每日|一天|每周|每星期|每月|隔天|隔日|每晚|每早|每次|一次|早晚|睡前|饭前|饭后|每隔{_NUMBER}(?:小时|天|日|周|月))"
_DOSE = re.compile(
    # Preserve existing frequency detection, including intervening dose advice.
    rf"{_FREQUENCY}[^，,。！？!?；;]{{0,12}}?{_NUMBER}{_BASE_UNIT}|"
    # New container units require a regimen form, not '每月购买一瓶'.
    rf"{_FREQUENCY}(?:建议|可|至少|最多|不超过|约)*(?:使用|服用|口服|涂抹|涂|擦|抹|滴|喷|贴|用|服|吃)?{_NUMBER}(?:支|管|瓶|贴|包)|"
    rf"(?:服用|口服|吞服|吃|服|涂抹|外涂|涂|擦|抹|滴|喷|贴|注射|用)"
    rf"(?:上|至|到|为|成)?{_NUMBER}{_UNIT}|"
    rf"(?:连用|连续服用|连续使用){_NUMBER}(?:天|日|周)|"
    rf"(?:增加|减少|调整|加|减)(?:到|至|为)?{_NUMBER}{_UNIT}|"
    r"加量(?!不加价)|减量|酌情增减|多吃|多服|多涂|少吃|少服|少涂|"
    r"薄涂|厚涂|少量涂抹|涂抹适量|酌情增加|酌情减少|"
    r"(?:增加|减少|调整)(?:剂量|用量)|(?:剂量|用量)(?:增加|减少|调整)",
    re.I,
)
_DOSE_TOPIC = re.compile(r"剂量|用量|用法|服法|加量|减量|加倍|翻倍")
_DOSE_QUESTION = re.compile(
    r"(?:吃|服|涂|抹|用)(?:药|膏)?(?:多少|几|多大|多厚|多久)|"
    rf"{_FREQUENCY}.{{0,8}}(?:多少|几次|几粒|几片|几支|几贴)|"
    r"(?:该|应该|能否|能不能|可以|要不要).{0,8}(?:加量|减量|加倍|调整用量)"
)
_PERSONAL_CONTEXT = re.compile(
    r"(?:我|本人|你|您|孩子|宝宝).{0,8}(?:患|得了|有|病|症状|体重|怀孕|哺乳)|"
    r"皮炎|湿疹|糖尿病|高血压|肝病|肝功能|肾功能|怀孕|孕妇|哺乳|过敏"
)
_PERSONAL_ADVICE = re.compile(
    r"(?:针对|根据|考虑到|结合|按照).{0,16}(?:你|您|病情|症状|体重|肝功能|肾功能)|"
    r"(?:你|您)(?:应|可以|需要|要|就|适合|每天|每次)|适合你|为你|给你|你的剂量|你的用量|"
    r"(?:你|您)的(?:体重|症状|病情|身体|肝功能|肾功能)"
)
_LABEL_PREFIX = re.compile(
    r"(?:S\d{3,}(?:的)?[：:]?)?(?:商品)?(?:标签|说明书)"
    r"(?:用法|用量|说明)?(?:记载|标注|写明|写着|注明)?[：:]",
    re.I,
)
_DOSE_MULTIPLIER = re.compile(r"加倍|翻倍")
_MEDICATION_CONTEXT = re.compile(r"剂量|用量|服用|吃药|服药|涂抹|药膏|薄涂")
# User-declared medical background is a safety signal, never product evidence.
# Unlike the dose context, '我有预算' is not a medical background.
MEDICAL_CONTEXT_MARKER = "【个人医疗咨询已拦截】"
_MEDICAL_BACKGROUND = re.compile(
    r"皮炎|湿疹|糖尿病|高血压|肝病|肝功能|肾功能|怀孕|孕妇|哺乳|过敏|"
    r"处方药|医生(?:给[你您我])?开的药|医师开的药|"
    r"(?:我|本人|你|您|孩子|宝宝)(?:的)?(?:病情|疾病|症状|身体情况)|"
    r"(?:我|本人|你|您|孩子|宝宝)(?:患有|患|得了|有).{0,12}(?:病|症|炎)|"
    + re.escape(MEDICAL_CONTEXT_MARKER)
)
_USE_QUESTION = re.compile(
    r"(?:适合|适用)(?:我|本人|你|您)|"
    r"(?:能不能|可不可以|是否可以|可以|能否|能|该不该|应不应该)"
    r"(?:直接|继续|自行|正常|给我)?(?:使用|服用|食用|用|吃|涂)|"
    r"(?:使用|服用|用|吃)(?:这款|它|S\d{3,})?(?:行吗|可以吗|合适吗)"
)
_PERSONAL_USE = re.compile(
    r"(?:你|您)(?:可以|可|能|应当|应该|适合|需要)(?:直接|继续|自行)?(?:使用|服用|用|吃|涂)|"
    r"(?:建议|推荐)(?:你|您)(?:使用|服用|用|吃|涂)|"
    r"(?:适合|适用于)(?:你|您)|"
    r"(?:可以|能)(?:使用|服用|用|吃|涂)"
)

# Treatment substitution and nonrecurrence are medical claims even without
# '治疗/根治'. Review each clause; a warning elsewhere cannot waive a promise.
_DRUG_REFERENCE = r"(?:处方药|医生(?:给[你您我])?开的药|医师开的药|药物|药品)"
_SUBSTITUTION = re.compile(
    rf"(?:替代|代替|取代|换掉).{{0,12}}{_DRUG_REFERENCE}|"
    rf"{_DRUG_REFERENCE}.{{0,12}}(?:替代|代替|取代|换成|换为|改用)"
)
_RECURRENCE = re.compile(
    r"(?:不再|不会再|再也不(?:会)?|永不)(?:复发|反复|发作)|"
    r"(?:皮炎|湿疹|疾病|病情|症状|高血压|糖尿病).{0,16}"
    r"(?:不再|不会再|再也不(?:会)?)(?:来|出现|发生|回来)"
)
_REPLACEMENT_ADVICE = re.compile(r"(?:可以|可|能|建议|直接|即可).{0,8}(?:换成|换为|改用)|换药")
_TREATMENT_WARNING = re.compile(
    rf"(?:本品|本产品|该商品|该产品|这款|S\d{{3,}})?"
    rf"(?:不能|不可|不得|不应|无法)(?:替代|代替|取代){_DRUG_REFERENCE}|"
    r"(?:本品|本产品|该商品|该产品|这款|S\d{3,})?"
    r"(?:不能|无法|不)保证(?:皮炎|湿疹|疾病|病情|症状)?"
    r"(?:不再|不会再|再也不(?:会)?|永不)(?:复发|反复|发作|来|出现|发生|回来)",
    re.I,
)


def _treatment_risks(text: str, context: str = "") -> bool:
    context_is_medical = bool(re.search(_DRUG_REFERENCE, _rule_text(context))
                              or has_medical_background(context))
    return any(
        not _TREATMENT_WARNING.fullmatch(clause)
        and (_SUBSTITUTION.search(clause) or _RECURRENCE.search(clause)
             or (context_is_medical and _REPLACEMENT_ADVICE.search(clause)))
        for clause in _clauses(text)
    )


def has_medical_background(text: str) -> bool:
    """Recognize bounded personal medical context for intent/output checks."""
    return bool(_MEDICAL_BACKGROUND.search(_rule_text(text)))


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    """Return whether text contains any configured risk term."""
    return any(term in text for term in terms)


def _normalize_text(text: str | None) -> str:
    """Normalize optional text input for rule checks."""
    return text or ""


def _rule_text(text: str) -> str:
    """Normalize width, spacing and common invisible separators for detection."""
    return re.sub(r"[\s\u200b-\u200f\u2060\ufeff]+", "", unicodedata.normalize("NFKC", text))


def _clauses(text: str) -> list[str]:
    # Keep punctuation boundaries until after splitting, including newlines.
    return [_rule_text(c).strip('"\'“”「」') for c in re.split(r"[，,。！？!?；;\n]+", text) if c.strip()]


def _reviewable_clauses(text: str) -> list[str]:
    return [c for c in _clauses(text)
            if not _LIMITATION.fullmatch(c) and not _DOSE_WARNING.fullmatch(c)]


def _verified_label_clauses(texts: tuple[str, ...]) -> set[str]:
    """Only explicit label sections in already verified product descriptions.

    This does not establish provenance; the Agent/caller must obtain these texts
    from current-SKU, fresh tool facts. Exact clause matching is deliberate.
    """
    allowed: set[str] = set()
    for text in texts:
        clauses = _clauses(text)
        if clauses and _LABEL_PREFIX.match(clauses[0]):
            allowed.update(_LABEL_PREFIX.sub("", c, count=1) for c in clauses if _DOSE.search(c))
    return allowed


def _dosing_clauses(clauses: list[str]) -> list[str]:
    """Multipliers alone may describe prices/specifications, not medication."""
    concrete = [c for c in clauses if _DOSE.search(c)]
    has_context = bool(concrete or _MEDICATION_CONTEXT.search("；".join(clauses)))
    return [c for c in clauses if c in concrete or (has_context and _DOSE_MULTIPLIER.search(c))]


def check_user_input(question: str | None, *, medical_context: str = "") -> dict[str, Any]:
    """Check whether a user question contains OTC medical risk intent."""
    normalized_question = "；".join(_reviewable_clauses(_normalize_text(question)))

    if _contains_any(normalized_question, DIAGNOSIS_INTENT_TERMS):
        return {
            "risk": True,
            "risk_type": "disease_diagnosis",
            "message": DISCLAIMER_MESSAGE,
        }

    if ((has_medical_background(normalized_question + medical_context)
         and _USE_QUESTION.search(normalized_question))
            or _contains_any(normalized_question, PERSONAL_MEDICATION_TERMS)
            or _DOSE_QUESTION.search(normalized_question)
            or (_DOSE.search(normalized_question)
                and re.search(r"可以吗|行吗|合适吗|能否|能不能|可不可以|是否可以|该不该", normalized_question))
            or (_PERSONAL_CONTEXT.search(normalized_question)
                and (_DOSE_TOPIC.search(normalized_question) or _DOSE.search(normalized_question)))):
        return {
            "risk": True,
            "risk_type": "personal_medication_advice",
            "message": DISCLAIMER_MESSAGE,
        }

    if (_contains_any(normalized_question, FORBIDDEN_MEDICAL_TERMS)
            or _ABSOLUTE_EFFICACY.search(normalized_question)
            or _treatment_risks(_normalize_text(question), medical_context)):
        return {
            "risk": True,
            "risk_type": "medical_consultation",
            "message": DISCLAIMER_MESSAGE,
        }

    return {"risk": False, "risk_type": "", "message": ""}


def check_ai_output(
    text: str | None, *, verified_label_texts: tuple[str, ...] = (),
    response_kind: str = "candidate", user_question: str = "", medical_context: str = "",
) -> dict[str, Any]:
    """Inspect all clauses and all risk categories; citations never waive claims.

    Missing evidence for an executable dose is a conservative refusal/verification
    case, not proof of medical harm. Generic mentions of dosage are not blocked.
    Product comparisons exempt only the exact policy notice, never source claims;
    the caller must separately verify the complete rendered response and facts.
    """
    original = _normalize_text(text)
    review_text = original
    if response_kind == "product_comparison":
        # Application-selected factual rendering: exempt only the exact policy
        # notice on its own line. Every other clause retains candidate review.
        lines = original.splitlines()
        if lines.count(PRODUCT_DISCLAIMER) != 1:
            return {"safe": False, "reason": "商品比较缺少独立、完整的固定声明。",
                    "filtered_text": "", "risk_types": ["invalid_product_notice"], "decision": "block"}
        review_text = "\n".join(line for line in lines if line != PRODUCT_DISCLAIMER)
        response_kind = "candidate"
    # Only the application can select this mode after rejecting input/output.
    # Exact complete constants prevent appended claims or claimed types escaping.
    if response_kind != "candidate":
        if response_kind == "system_refusal" and original in (DISCLAIMER_MESSAGE, SAFETY_FALLBACK_MESSAGE):
            return {"safe": True, "reason": "", "filtered_text": original,
                    "risk_types": [], "decision": "allow_system_refusal"}
        return {"safe": False, "reason": "文本不是已登记的固定安全拒答。",
                "filtered_text": "", "risk_types": ["invalid_system_refusal"], "decision": "block"}
    clauses = _reviewable_clauses(review_text)
    remaining = "；".join(clauses)
    risks: list[str] = []
    personal_question = (has_medical_background(user_question + medical_context)
                         and _USE_QUESTION.search(_rule_text(user_question)))
    personal_claim = (_PERSONAL_USE.search(remaining)
                      and ((has_medical_background(remaining) and re.search(r"你|您", remaining))
                           or (re.search(r"你|您", remaining)
                               and re.search(r"S\d{3,}|商品|产品|这款|药", remaining, re.I))))
    if personal_question or personal_claim:
        risks.append("personal_medication_advice")
    if _contains_any(remaining, FORBIDDEN_MEDICAL_TERMS):
        risks.append("medical_claim")
    if _ABSOLUTE_EFFICACY.search(remaining):
        risks.append("absolute_efficacy")
    if _treatment_risks(review_text, user_question + medical_context):
        risks.append("treatment_substitution_or_nonrecurrence")

    dosing = _dosing_clauses(clauses)
    if dosing and (_PERSONAL_ADVICE.search(remaining)
                   or _contains_any(remaining, FORBIDDEN_MEDICAL_TERMS)):
        risks.append("personalized_or_treatment_dosage")
    allowed = _verified_label_clauses(verified_label_texts)
    cited = any(_LABEL_PREFIX.match(c) for c in clauses)
    if any(not cited or _LABEL_PREFIX.sub("", c, count=1) not in allowed for c in dosing):
        risks.append("unverified_dosage")

    if risks:
        needs_evidence = risks == ["unverified_dosage"]
        return {
            "safe": False,
            "reason": ("用量说明缺少可核验的当前商品标签来源，请核对说明书或咨询医生或药师。"
                       if needs_evidence else OUTPUT_BLOCK_MESSAGE),
            "filtered_text": "",
            "risk_types": risks,
            "decision": "clarify" if needs_evidence else "block",
        }

    return {"safe": True, "reason": "", "filtered_text": original, "risk_types": [], "decision": "allow"}


def check_safety(text: str) -> dict[str, Any]:
    """Backward-compatible wrapper for basic AI output safety checks."""
    result = check_ai_output(text)
    return {"safe": result["safe"], "risks": [] if result["safe"] else [result["reason"]]}

